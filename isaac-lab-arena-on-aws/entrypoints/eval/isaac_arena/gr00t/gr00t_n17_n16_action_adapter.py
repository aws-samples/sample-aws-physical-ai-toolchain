"""GR00T N1.7<->N1.6 Arena action-format adapter.

(Renamed from gr00t_action_compat; identity is now explicit. Hardened + fails
loud. Works generally for the Arena GR1 arms-only action space.)

TODO(port): a .pth auto-import that monkeypatches gr00t_core through a
sys.meta_path hook inside a vendored interpreter is not something to upstream
quietly. Either pin an Arena client that speaks the N1.7 action format, or make
this an explicit adapter module the eval entry imports. The --gr00t-version n16
path avoids the shim entirely (native GR1 head). See README.md#gr00t-remote-policy-protocol.

Our GR00T server (Isaac-GR00T@376ba890, Gr00tN1d7Pipeline -- the only commit that
loads our fine-tuned N1.7 checkpoint) returns actions as a per-modality DICT. The
baked N1.6-era Arena client feeds that dict straight into
remap_policy_joints_to_sim_joints_np, which does `joint_pos.shape` ->
"'dict' object has no attribute 'shape'" (joints_conversion.py:165).

FIX: patch remap_policy_joints_to_sim_joints_np at its DEFINITION module
(isaaclab_arena_gr00t.utils.joints_conversion). gr00t_core imports it via
`from ..utils.joints_conversion import remap_policy_joints_to_sim_joints_np`
AFTER joints_conversion executes, so patching the definition here means gr00t_core
binds the patched version. When joint_pos is a dict, concatenate the per-modality
arrays into the policy-joint array (gr00t_26dof order) before the original runs.

DELIVERY: auto-imported at interpreter startup via a companion .pth in site-packages
so it loads in the policy_runner SUBPROCESS. A sys.meta_path hook patches the target
module lazily on import (never imports isaaclab_arena eagerly -> no AppLauncher
segfault).

DIAGNOSTICS: the first time a dict joint_pos is seen, logs its keys + each value's
shape/dtype, so the true wire structure is captured in the SimEval log.
"""
import importlib.abc
import importlib.util
import sys

_TARGET = "isaaclab_arena_gr00t.utils.joints_conversion"
_FUNC = "remap_policy_joints_to_sim_joints_np"
# gr00t_26dof_joint_space.yaml order: left_arm(7)+right_arm(7)+left_hand(6)+right_hand(6)=26
_ACTION_KEY_ORDER = ["left_arm", "right_arm", "left_hand", "right_hand"]
_logged = {}


def _to_np(x):
    import numpy as np
    if hasattr(x, "detach"):        # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _decode_value(v):
    """Faithfully decode ONE modality action value to its native ndarray, WITHOUT
    concatenating (the original remap iterates the per-modality dict itself).
    Handles: msgpack-numpy dict {b'nd',b'type',b'shape',b'data'}, torch tensor,
    0-d object-array box, or an already-numeric ndarray (pass-through)."""
    import numpy as np
    if hasattr(v, "detach"):                 # torch tensor
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        if v.dtype == object and v.ndim == 0:  # 0-d object box -> unwrap once
            return _decode_value(v.item())
        return v                              # numeric ndarray as-is
    if isinstance(v, dict):
        keys = set(v.keys())
        if b"data" in keys or "data" in keys:  # msgpack-numpy encoded ndarray
            data = v.get(b"data", v.get("data"))
            typ = v.get(b"type", v.get("type", "<f4"))
            shp = v.get(b"shape", v.get("shape"))
            if isinstance(typ, bytes):
                typ = typ.decode()
            arr = np.frombuffer(data, dtype=np.dtype(typ))
            if shp is not None:
                arr = arr.reshape(tuple(int(s) for s in shp))
            return arr
    return np.asarray(v)


# --- GR1 action contract -----------------------------------------------------------
#
# Upstream's remap_policy_joints_to_sim_joints_np() allocates a ZERO array and fills only
# the joints whose group AND joint-name lookup both succeed. A missing action group
# therefore leaves zeros in the command array, no finiteness check intervenes, and the
# run can still be labelled policy_type=checkpoint. The active wrapper below
# checks every required group before remapping.
#
# A valid policy CAN emit zero for any active joint, including an entirely zero chunk.
# Presence and correct interpretation are what distinguish valid zero data from a missing
# action; nonzero magnitude does not. So these checks are about STRUCTURE, never about
# whether values look "big enough".
GR1_ACTION_GROUPS = {
    # group: number of joints, in policy order
    "left_arm": 7,    # shoulder pitch/roll/yaw, elbow pitch, wrist yaw/roll/pitch
    "right_arm": 7,   # same order, right arm
    "left_hand": 6,   # pinky/ring/middle/index proximal, thumb proximal yaw/pitch
    "right_hand": 6,  # corresponding R_ joints in the same order
}
GR1_POLICY_DOF = sum(GR1_ACTION_GROUPS.values())   # 26
GR1_SIM_DOF = 36
# The simulator slots the 26-DoF policy map intentionally leaves unmapped: upstream
# labels them mimic joints. They stay zero in the REMAPPED COMMAND ARRAY -- which is not
# a claim that their physical positions stay zero, nor that arbitrary omitted joints are
# passive.
GR1_UNMAPPED_SIM_SLOTS = GR1_SIM_DOF - GR1_POLICY_DOF   # 10
# The Arena GR1 client/remap tag. This is NOT the N1.7 server head tag
# (`new_embodiment`): NEW_EMBODIMENT remapping does not recognise GR1's L_/R_ hand
# prefixes, so using the server tag here zeroes every hand slot while the run still
# reports a checkpoint policy. The two tags are independent and must not be conflated.
GR1_REMAP_TAG = "GR1"
_KNOWN_REMAP_TAGS = {"GR1", "NEW_EMBODIMENT", "new_embodiment", "GR1T2"}
# The policy-index -> simulator-index mapping, learned from the pinned remapper on the
# first call rather than hardcoded from the embodiment YAMLs (which this code cannot
# verify). Module-level so it is learned once per process, not once per action.
# I4: keyed by the remap configuration, NOT global. A single unkeyed slot meant a
# mapping learned under one remap tag / group shape was reused for another, so a later
# call was verified against a permutation that never described it.
_PERMUTATION = {}


def _permutation_key(tag, resolved, policy_config, sim_config):
    """Identify the configuration a derived mapping is valid for.

    Keyed by the CONTENT that determines the mapping, not merely its shape. An earlier key of
    (tag, group widths) was insufficient: changing simulator destination indices while keeping
    the same tag and dimensions reused the stale permutation, so a remapper still producing
    the old placement was accepted. The mapping is a function of the joint NAMES and the
    columns they map to, so those are what the key must capture.

    The tag comes from the single extraction point so this cannot drift from upstream's actual
    keyword, which is `embodiment_tag` -- reading a differently named key here silently
    produced a None tag.
    """
    import numpy as np

    shape = tuple(sorted(
        (group, int(np.asarray(resolved[group]).shape[-1])) for group in resolved))
    # Ordered names per group. Order IS the placement, so a reordering must be a new key.
    if isinstance(policy_config, dict):
        names = tuple(sorted(
            (group, tuple(members) if isinstance(members, (list, tuple)) else None)
            for group, members in policy_config.items()))
    else:
        names = None
    # Destination columns for exactly the joints this configuration names.
    if isinstance(sim_config, dict) and isinstance(policy_config, dict):
        referenced = {name for members in policy_config.values()
                      if isinstance(members, (list, tuple)) for name in members}
        destinations = tuple(sorted(
            (name, sim_config.get(name)) for name in referenced))
    else:
        destinations = None
    return (tag, shape, names, destinations)


def _validate_policy_action(joint_pos):
    """Require the full GR1 action contract before upstream remapping.

    Raises RuntimeError (never `assert`, which vanishes under -O) naming exactly what was
    expected and what arrived.
    """
    import numpy as np

    if not isinstance(joint_pos, dict):
        raise RuntimeError(
            f"gr00t action contract: expected a per-group dict, got "
            f"{type(joint_pos).__name__}. Upstream remapping fills zeros for groups it "
            f"cannot find, so a non-dict cannot be accepted.")

    # Resolve each group, rejecting simultaneous aliases rather than silently preferring
    # one. The native server returns bare group names; `action.<group>` is a
    # compatibility spelling.
    resolved = {}
    for group in GR1_ACTION_GROUPS:
        prefixed, bare = f"action.{group}", group
        present = [key for key in (prefixed, bare) if key in joint_pos]
        if len(present) > 1:
            raise RuntimeError(
                f"gr00t action contract: group {group!r} supplied under BOTH "
                f"{prefixed!r} and {bare!r}. Refusing to choose one silently -- the two "
                f"could carry different actions.")
        if not present:
            raise RuntimeError(
                f"gr00t action contract: required action group {group!r} is missing "
                f"(got keys {sorted(map(str, joint_pos))}). Upstream remapping would "
                f"leave this group's simulator slots at ZERO and the run would still be "
                f"labelled a checkpoint policy.")
        resolved[group] = joint_pos[present[0]]

    # No unexpected groups: an unrecognised key means the protocol is not the one this
    # contract describes.
    expected_keys = set()
    for group in GR1_ACTION_GROUPS:
        expected_keys.update((group, f"action.{group}"))
    unexpected = sorted(str(k) for k in joint_pos if k not in expected_keys)
    if unexpected:
        raise RuntimeError(
            f"gr00t action contract: unexpected action groups {unexpected}. Expected "
            f"exactly {sorted(GR1_ACTION_GROUPS)}.")

    horizon = None
    for group, joints in GR1_ACTION_GROUPS.items():
        array = np.asarray(resolved[group])
        if array.dtype.kind != "f":
            raise RuntimeError(
                f"gr00t action contract: group {group!r} has dtype {array.dtype!r}; "
                f"absolute joint positions must be floating point (object, boolean and "
                f"complex data indicate a decode failure, not an action).")
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != joints:
            raise RuntimeError(
                f"gr00t action contract: group {group!r} has shape {array.shape}, "
                f"expected (1, H, {joints}).")
        if horizon is None:
            horizon = array.shape[1]
        elif array.shape[1] != horizon:
            raise RuntimeError(
                f"gr00t action contract: group {group!r} has horizon {array.shape[1]} "
                f"but an earlier group had {horizon}. The groups describe one action "
                f"chunk and must agree.")
        if not np.isfinite(array).all():
            raise RuntimeError(
                f"gr00t action contract: group {group!r} contains non-finite values "
                f"(NaN/Inf). These would be sent to the simulator as joint positions.")
    return resolved, horizon


def _derive_permutation(policy_joints_config, sim_joints_config, tag):
    """Build the policy-index -> simulator-index mapping from CONFIGURATION.

    This replaces learning the mapping by probing the remapper with a canary action. That
    was circular: the expected mapping came from the very function being verified, so a
    consistently wrong remapper became the accepted reference. A probe that reversed all 36
    output positions on the first call was accepted, and the rejection tests only worked
    because they calibrated against a correct remapper first.

    Upstream maps by joint NAME, and both orders it maps between arrive as arguments:

      policy_joints_config: dict[str, list[str]] -- group -> joint names, ordered by that
                            group's column in the action array
      sim_joints_config:    dict[str, int]       -- joint name -> output column index

    So the expected destination of each policy joint is a name lookup and involves no call
    to the remapper. NOTE: upstream's own annotation and docstring for policy_joints_config
    describe a nested name-to-index mapping, which is wrong -- the YAML supplies lists and
    the loader preserves them. The lists are what this reads.

    Returns a tuple of GR1_POLICY_DOF simulator indices. Raises RuntimeError naming the
    cause if the configuration is absent, misshapen, or does not resolve, since a mapping
    that cannot be established independently must not be substituted with a guess.
    """
    if tag != GR1_REMAP_TAG:
        raise RuntimeError(
            f"gr00t action contract: cannot derive the joint mapping for remap tag "
            f"{tag!r}; only {GR1_REMAP_TAG!r} is supported here.")
    if not isinstance(policy_joints_config, dict) or not policy_joints_config:
        raise RuntimeError(
            f"gr00t action contract: policy_joints_config must be a non-empty dict of "
            f"group -> ordered joint names, got {type(policy_joints_config).__name__}. "
            f"Without it the expected joint placement cannot be established "
            f"independently of the remapper being checked.")
    if not isinstance(sim_joints_config, dict) or not sim_joints_config:
        raise RuntimeError(
            f"gr00t action contract: sim_joints_config must be a non-empty dict of joint "
            f"name -> output index, got {type(sim_joints_config).__name__}.")
    if set(policy_joints_config) != set(GR1_ACTION_GROUPS):
        raise RuntimeError(
            f"gr00t action contract: policy_joints_config declares groups "
            f"{sorted(policy_joints_config)} but this contract covers "
            f"{sorted(GR1_ACTION_GROUPS)}.")
    permutation = []
    for group, expected_width in GR1_ACTION_GROUPS.items():
        names = policy_joints_config[group]
        if isinstance(names, dict) or not isinstance(names, (list, tuple)):
            raise RuntimeError(
                f"gr00t action contract: policy_joints_config[{group!r}] must be an "
                f"ORDERED list of joint names, got {type(names).__name__}. Upstream's "
                f"annotation claims a mapping here, but the pinned configuration supplies "
                f"a list and its order is the placement.")
        if len(names) != expected_width:
            raise RuntimeError(
                f"gr00t action contract: policy_joints_config[{group!r}] lists "
                f"{len(names)} joints but the contract expects {expected_width}.")
        for joint_name in names:
            if joint_name not in sim_joints_config:
                raise RuntimeError(
                    f"gr00t action contract: policy joint {joint_name!r} in group "
                    f"{group!r} has no entry in sim_joints_config, so its destination "
                    f"column is unknown.")
            sim_index = sim_joints_config[joint_name]
            if not isinstance(sim_index, int) or isinstance(sim_index, bool):
                raise RuntimeError(
                    f"gr00t action contract: sim_joints_config[{joint_name!r}] is "
                    f"{sim_index!r}, which is not an output column index.")
            if not 0 <= sim_index < GR1_SIM_DOF:
                raise RuntimeError(
                    f"gr00t action contract: sim_joints_config[{joint_name!r}] is "
                    f"{sim_index}, outside the {GR1_SIM_DOF} simulator action columns.")
            permutation.append(sim_index)
    if len(set(permutation)) != len(permutation):
        duplicated = sorted({i for i in permutation if permutation.count(i) > 1})
        raise RuntimeError(
            f"gr00t action contract: simulator columns {duplicated} are the destination "
            f"of more than one policy joint, so the placement is not an injection and "
            f"one joint's command would overwrite another's.")
    if len(permutation) != GR1_POLICY_DOF:
        raise RuntimeError(
            f"gr00t action contract: derived {len(permutation)} destinations for "
            f"{GR1_POLICY_DOF} policy joints.")
    return tuple(permutation)


def _validate_remapped_action(remapped, resolved, horizon, permutation):
    """Require every policy joint to land at ITS OWN simulator index.

    This used to sort both arrays and compare them, so any permutation preserving the 26
    values and ten zeros passed -- including values moved into the wrong arm or into an
    unmapped slot. A probe that REVERSED all 36 output positions was accepted. Upstream
    maps by joint name to a configured destination index, so position is the contract.

    The permutation is derived from the supplied joint configurations (see
    _derive_permutation), independently of the remapper being checked,
    so this checks placement without hardcoding an embodiment table this code cannot verify.
    """
    import numpy as np

    array = np.asarray(remapped)
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise RuntimeError(
            f"gr00t action contract: remapped action is not finite floating point "
            f"(dtype={array.dtype!r}). Non-finite joint targets must never reach the "
            f"simulator.")
    if array.shape != (1, horizon, GR1_SIM_DOF):
        raise RuntimeError(
            f"gr00t action contract: remapped action has shape {array.shape}, expected "
            f"(1, {horizon}, {GR1_SIM_DOF}).")
    supplied = np.concatenate(
        [np.asarray(resolved[group]) for group in GR1_ACTION_GROUPS], axis=2)
    unmapped = [index for index in range(GR1_SIM_DOF) if index not in set(permutation)]
    for step in range(horizon):
        for policy_index, sim_index in enumerate(permutation):
            expected = supplied[0, step, policy_index]
            actual = array[0, step, sim_index]
            if not np.array_equal(actual, expected):
                raise RuntimeError(
                    f"gr00t action contract: policy joint {policy_index} should occupy "
                    f"simulator index {sim_index} at step {step}, but that slot holds "
                    f"{actual!r} instead of {expected!r}. A value landing in the wrong "
                    f"joint commands a different motion than the policy produced.")
        for sim_index in unmapped:
            if array[0, step, sim_index] != 0.0:
                raise RuntimeError(
                    f"gr00t action contract: simulator index {sim_index} is not mapped by "
                    f"the 26-DoF policy and must stay zero in the command array, but it "
                    f"holds {array[0, step, sim_index]!r} at step {step}.")
    return array


def _remap_call_configuration(args, kwargs):
    """Recover (policy_joints_config, sim_joints_config, tag) from the call arguments.

    Upstream's signature is
      remap_policy_joints_to_sim_joints_np(policy_joints, policy_joints_config,
                                           sim_joints_config, embodiment_tag=...)
    so after the wrapper's own first parameter the two configurations are the next two
    positional arguments. Keywords are honoured if used.
    """
    policy_config = kwargs.get("policy_joints_config")
    sim_config = kwargs.get("sim_joints_config")
    positional = [value for value in args if isinstance(value, dict)]
    if policy_config is None and positional:
        policy_config = positional[0]
    if sim_config is None and len(positional) > 1:
        sim_config = positional[1]
    tag = kwargs.get("embodiment_tag")
    if tag is None:
        tag = next((value for value in args
                    if isinstance(value, str) and value in _KNOWN_REMAP_TAGS), None)
    if tag is None:
        tag = GR1_REMAP_TAG
    return policy_config, sim_config, str(tag).upper()


def _validate_remap_tag(args, kwargs):
    """Reject a non-GR1 remap tag if one is present among the call arguments.

    The tag's parameter position is not assumed: any argument whose value is a known
    embodiment tag is checked. NEW_EMBODIMENT remapping does not recognise GR1's L_/R_
    hand prefixes, so the wrong tag zeroes every hand slot silently.
    """
    candidates = [value for value in list(args) + list(kwargs.values())
                  if isinstance(value, str) and value in _KNOWN_REMAP_TAGS]
    for tag in candidates:
        if tag.upper() != GR1_REMAP_TAG:
            raise RuntimeError(
                f"gr00t action contract: remap tag {tag!r} is not {GR1_REMAP_TAG!r}. "
                f"That remapping does not recognise GR1's L_/R_ hand joint prefixes and "
                f"would zero every hand slot while still reporting a checkpoint policy. "
                f"Note this is the CLIENT/remap tag, not the server head tag.")
    return candidates


def _patch_module(module):
    orig = getattr(module, _FUNC, None)
    if orig is not None and getattr(orig, "_gr00t_n17_n16_action_adapter", False):
        return  # already patched -- idempotent, not a failure
    if orig is None:
        # The adapter exists because the N1.7 server delivers msgpack-numpy dicts the
        # N1.6-era remap cannot decode. If the function it must wrap is absent, the
        # adapter is NOT installed and the rollout would run with undecoded actions --
        # so fail loudly rather than returning quietly and letting the eval proceed as
        # though the adapter were active.
        raise RuntimeError(
            f"gr00t_n17_n16_action_adapter: {_TARGET}.{_FUNC} not found, so the N1.7 "
            f"action adapter cannot be installed. Refusing to continue silently: the "
            f"evaluation would run without the decode/remap this adapter provides. "
            f"Check the pinned Arena/GR00T versions against this adapter's target.")

    def patched(joint_pos, *args, **kwargs):
        # The ORIGINAL remap iterates the per-modality DICT (`for _, v in
        # policy_joints.items()`) and expects each value to be a numeric ndarray.
        # The N1.7 server delivers each modality value as a msgpack-numpy dict
        # ({b'nd',b'type',b'shape',b'data'}) that the N1.6-era client never decodes.
        # So: PRESERVE the dict, DECODE each value -> ndarray, and let the original
        # do its own concat/joint-remap. (Earlier versions wrongly concatenated the
        # dict into an array before calling orig -> 'ndarray has no attribute items'.)
        if isinstance(joint_pos, dict):
            joint_pos = {k: _decode_value(v) for k, v in joint_pos.items()}
            if not _logged.get("decoded"):
                try:
                    shapes = {k: getattr(a, "shape", type(a).__name__)
                              for k, a in joint_pos.items()}
                    print("[PATCH] gr00t_n17_n16_action_adapter decoded modality dict %s" % shapes,
                          flush=True)
                except Exception:
                    pass
                _logged["decoded"] = True
        # Reject missing groups before upstream remapping can turn them into zeros.
        _validate_remap_tag(args, kwargs)
        resolved, horizon = _validate_policy_action(joint_pos)
        # Derive the expected joint placement from the CONFIGURATION upstream maps by,
        # not by probing the remapper. Probing was circular: the expected mapping came
        # from the function under verification, so a remapper that reversed all 36
        # output positions became its own accepted reference.
        policy_config, sim_config, tag = _remap_call_configuration(args, kwargs)
        permutation_key = _permutation_key(
            tag, resolved, policy_config, sim_config)
        if permutation_key not in _PERMUTATION:
            _PERMUTATION[permutation_key] = _derive_permutation(
                policy_config, sim_config, tag)
            # The key now carries every joint name and destination, so log a DIGEST of it.
            # Printing the key itself is thousands of characters and buries the permutation
            # it is meant to explain.
            import hashlib as _hashlib
            _key_digest = _hashlib.sha256(
                repr(permutation_key).encode("utf-8")).hexdigest()[:12]
            print("[PATCH] gr00t_n17_n16_action_adapter derived joint permutation from "
                  f"configuration (tag={permutation_key[0]!r} "
                  f"config_sha256={_key_digest}): {_PERMUTATION[permutation_key]}",
                  flush=True)
        # Pass the CANONICALISED groups. _validate_policy_action resolves the
        # `action.<group>` alias spelling, but the original dict was being handed to
        # upstream -- so on an aliased payload upstream saw keys it does not map and
        # zero-filled every group while validation had already passed.
        remapped = orig(resolved, *args, **kwargs)
        return _validate_remapped_action(
            remapped, resolved, horizon, _PERMUTATION[permutation_key])

    patched._gr00t_n17_n16_action_adapter = True
    setattr(module, _FUNC, patched)
    print("[PATCH] gr00t_n17_n16_action_adapter: %s dict->array shim active" % _FUNC, flush=True)


class _CompatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        real_exec = spec.loader.exec_module

        def exec_module(mod):
            real_exec(mod)
            # Do NOT swallow: a failed patch means the rollout would run without the
            # N1.7 decode/remap while still being reported as a checkpoint evaluation.
            _patch_module(mod)

        spec.loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _CompatFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _CompatFinder())
    if _TARGET in sys.modules:
        # Already-imported target: patch it now, and let a failure propagate for the
        # same reason as above.
        _patch_module(sys.modules[_TARGET])
