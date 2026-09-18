"""The GR1 action contract must be enforced before upstream remapping.

Raised as C3 in one review cycle and as I4 in a later one; cross-cycle finding
IDs are not stable, so the defect is described rather than numbered.

Upstream `remap_policy_joints_to_sim_joints_np()` allocates a ZERO array and fills only
the joints whose group AND joint-name lookup both succeed. A missing action group
therefore leaves zeros in the command array, no finiteness check intervenes, and the run
is still labelled `policy_type=checkpoint`. This module already contained a
`_coerce_dict_to_array()` with a missing-key check, but the ACTIVE wrapper never called
it, so nothing enforced the contract on the path that runs.

The stand-in remapper below reproduces the pinned upstream behaviour that matters: fill
what you can find by name, leave everything else zero. Tests drive the real patched
function, not a copy of it.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_ADAPTER = (pathlib.Path(__file__).resolve().parents[1]
            / "entrypoints/eval/isaac_arena/gr00t/gr00t_n17_n16_action_adapter.py")

HORIZON = 16

_PINNED_CONFIG_DIR = pathlib.Path(__file__).with_name("data") / "arena_gr1"


def _pinned_configs():
    """The REAL pinned Arena joint configurations.

    Verbatim from isaac-sim/IsaacLab-Arena at 8b4a3a47fc53de23e8205089d71109a2e2348acd,
    isaaclab_arena_gr00t/embodiments/gr1/. Fixtures were previously invented, and the tag
    was passed in the position upstream uses for policy_joints_config -- so the suite
    exercised a signature that does not exist.
    """
    yaml = pytest.importorskip("yaml")
    policy = yaml.safe_load(
        (_PINNED_CONFIG_DIR / "gr00t_26dof_joint_space.yaml").read_text())["joints"]
    sim = yaml.safe_load(
        (_PINNED_CONFIG_DIR / "36dof_joint_space.yaml").read_text())["joints"]
    return policy, sim


# The mapping those configurations imply, computed from them rather than asserted by hand.
EXPECTED_PERMUTATION = (
    0, 2, 4, 6, 8, 10, 12,
    1, 3, 5, 7, 9, 11, 13,
    16, 17, 15, 14, 18, 28,
    21, 22, 20, 19, 23, 33,
)


def _remap_args(tag=None):
    """The positional/keyword arguments upstream actually passes after policy_joints."""
    policy, sim = _pinned_configs()
    return (policy, sim), {"embodiment_tag": tag or "GR1"}



def _adapter():
    spec = importlib.util.spec_from_file_location("gr1_action_contract_test", _ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _UpstreamModule:
    """Stands in for the Arena module the adapter patches.

    `remap_policy_joints_to_sim_joints_np` mirrors the pinned upstream contract: a zero
    array of simulator width, filled only where a group is found. Groups it cannot find
    stay zero -- which is precisely the behaviour C3 is about.
    """

    def __init__(self, groups, sim_dof=36):
        self._groups = list(groups)
        self._sim_dof = sim_dof

    def remap_policy_joints_to_sim_joints_np(self, policy_joints, policy_joints_config=None,
                                             sim_joints_config=None, embodiment_tag="GR1",
                                             **kwargs):
        """Place each group's columns by JOINT NAME, as pinned upstream does.

        This previously concatenated the found groups into columns 0..25 in order, which is
        not what upstream does -- the real 36-column layout interleaves left and right arms.
        A fake that places differently from upstream cannot show that a corruption of the
        real placement is caught.
        """
        out = np.zeros((1, HORIZON, self._sim_dof), dtype=np.float32)
        horizon = None
        for group in self._groups:
            value = policy_joints.get(group, policy_joints.get(f"action.{group}"))
            if value is None:
                continue
            array = np.asarray(value)
            horizon = array.shape[1]
            if policy_joints_config is None or sim_joints_config is None:
                raise AssertionError(
                    "the fake upstream needs the joint configurations, as the real one does")
            for column, joint_name in enumerate(policy_joints_config[group]):
                out[0, :, sim_joints_config[joint_name]] = array[0, :, column]
        if horizon is None:
            return np.zeros((1, HORIZON, self._sim_dof), dtype=np.float32)
        return out[:, :horizon, :]


def _remap(upstream, action, tag="GR1"):
    """Call the patched remapper the way upstream does, with real configs."""
    policy, sim = _pinned_configs()
    return upstream.remap_policy_joints_to_sim_joints_np(
        action, policy, sim, embodiment_tag=tag)


def _install(mod, groups=None, sim_dof=36):
    upstream = _UpstreamModule(groups or list(mod.GR1_ACTION_GROUPS), sim_dof)
    mod._patch_module(upstream)
    return upstream


def _action(mod, omit=None, horizon=HORIZON, value=None):
    """A well-formed action chunk, with a distinctive value per joint by default."""
    action, cursor = {}, 0.0
    for group, joints in mod.GR1_ACTION_GROUPS.items():
        if omit and group in omit:
            continue
        if value is None:
            block = np.arange(cursor, cursor + joints, dtype=np.float32) + 1.0
            arr = np.tile(block, (horizon, 1))[None, :, :]
            cursor += joints
        else:
            arr = np.full((1, horizon, joints), value, dtype=np.float32)
        action[group] = arr
    return action


def test_a_wellformed_action_passes_and_preserves_every_joint():
    """Positive control: 26 distinct values survive, 10 unmapped slots are zero."""
    mod = _adapter()
    upstream = _install(mod)
    out = _remap(upstream, _action(mod))
    assert out.shape == (1, HORIZON, 36)
    assert np.count_nonzero(out[0, 0]) == mod.GR1_POLICY_DOF
    assert (out[0, 0] == 0).sum() == mod.GR1_UNMAPPED_SIM_SLOTS


def test_an_entirely_zero_action_is_valid():
    """A valid policy can emit zero for any active joint, including a whole chunk.

    Presence and interpretation distinguish valid zero data from a missing action;
    magnitude does not. A magnitude-based check would reject this.
    """
    mod = _adapter()
    upstream = _install(mod)
    out = _remap(upstream, _action(mod, value=0.0))
    assert out.shape == (1, HORIZON, 36)
    assert np.count_nonzero(out) == 0


@pytest.mark.parametrize("group", ["left_arm", "right_arm", "left_hand", "right_hand"])
def test_a_missing_group_is_rejected(group):
    """The reviewer's probe: a missing group became zeros and the run still passed."""
    mod = _adapter()
    upstream = _install(mod)
    with pytest.raises(RuntimeError, match=f"action group '{group}' is missing"):
        _remap(upstream, _action(mod, omit={group}))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_rejected(bad):
    """The reviewer's probe: NaN survived to the simulator."""
    mod = _adapter()
    upstream = _install(mod)
    action = _action(mod)
    action["left_hand"][0, 3, 2] = bad
    with pytest.raises(RuntimeError, match="non-finite"):
        _remap(upstream, action)


@pytest.mark.parametrize("group,shape", [
    ("left_arm", (1, HORIZON, 6)),      # wrong joint count
    ("right_hand", (1, HORIZON, 7)),    # wrong joint count
    ("left_hand", (2, HORIZON, 6)),     # batch must be 1
    ("right_arm", (HORIZON, 7)),        # missing batch dimension
])
def test_wrong_dimensions_are_rejected(group, shape):
    mod = _adapter()
    upstream = _install(mod)
    action = _action(mod)
    action[group] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(RuntimeError, match="expected"):
        _remap(upstream, action)


def test_a_disagreeing_horizon_is_rejected():
    mod = _adapter()
    upstream = _install(mod)
    action = _action(mod)
    action["right_hand"] = np.zeros((1, HORIZON - 1, 6), dtype=np.float32)
    with pytest.raises(RuntimeError, match="horizon"):
        _remap(upstream, action)


@pytest.mark.parametrize("dtype", [np.int32, bool, np.complex64, object])
def test_nonfloat_data_is_rejected(dtype):
    """Object, boolean and complex data indicate a decode failure, not an action."""
    mod = _adapter()
    upstream = _install(mod)
    action = _action(mod)
    action["left_arm"] = np.zeros((1, HORIZON, 7), dtype=dtype)
    with pytest.raises(RuntimeError, match="floating point|non-finite"):
        _remap(upstream, action)


def test_simultaneous_aliases_are_rejected():
    """Two spellings could carry different actions; refuse to prefer one silently."""
    mod = _adapter()
    upstream = _install(mod)
    action = _action(mod)
    action["action.left_arm"] = action["left_arm"]
    with pytest.raises(RuntimeError, match="supplied under BOTH"):
        _remap(upstream, action)


def test_an_unexpected_group_is_rejected():
    mod = _adapter()
    upstream = _install(mod)
    action = _action(mod)
    action["neck"] = np.zeros((1, HORIZON, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="unexpected action groups"):
        _remap(upstream, action)


def test_the_wrong_remap_tag_is_rejected():
    """NEW_EMBODIMENT remapping does not recognise GR1's L_/R_ hand prefixes.

    This is the CLIENT/remap tag, which is independent of the N1.7 server head tag --
    conflating them zeroes every hand slot while still reporting a checkpoint policy.
    """
    mod = _adapter()
    upstream = _install(mod)
    with pytest.raises(RuntimeError, match="not 'GR1'"):
        upstream.remap_policy_joints_to_sim_joints_np(_action(mod), *_remap_args("NEW_EMBODIMENT")[0], **_remap_args("NEW_EMBODIMENT")[1])


def test_the_correct_remap_tag_passes():
    mod = _adapter()
    upstream = _install(mod)
    out = _remap(upstream, _action(mod), "GR1")
    assert out.shape == (1, HORIZON, 36)


def test_a_dropped_group_after_remapping_is_caught():
    """Upstream silently dropping a group must not pass.

    The contract check passes -- every group is present and well formed -- but the remapper
    ignores right_hand, exactly as upstream does for a group it cannot map by name. This is
    now caught while LEARNING the permutation: only 20 of 26 joints get placed, so the
    mapping cannot be established and evaluation must not proceed against an unknown one.
    """
    mod = _adapter()
    upstream = _install(mod, groups=["left_arm", "right_arm", "left_hand"])
    with pytest.raises(RuntimeError, # The placement check now names the exact joint and slot, which the
        # learner-derived message could not.
        match="policy joint 20 should occupy simulator index 21"):
        _remap(upstream, _action(mod))


def test_a_wrong_simulator_width_is_rejected():
    mod = _adapter()
    upstream = _install(mod, sim_dof=54)
    with pytest.raises(RuntimeError, match=r"remapped action has shape \(1, 16, 54\)"):
        _remap(upstream, _action(mod))


def test_a_reversed_output_is_rejected():
    """I4: the guard compared a sorted multiset, so reversing all 36 positions passed.

    Upstream maps each joint by NAME to a configured destination index, so position is the
    contract: a value landing in the wrong joint commands a different motion than the
    policy produced.
    """
    mod = _adapter()
    upstream = _install(mod)
    # No calibration: the expected mapping comes from CONFIGURATION, so the very
    # first call is checked. Previously these tests had to call a real remapper
    # first, which is exactly why a remapper wrong from the start was accepted.

    class _Reversing:
        def remap_policy_joints_to_sim_joints_np(self, policy_joints, *args, **kwargs):
            out = _UpstreamModule(list(mod.GR1_ACTION_GROUPS)) \
                .remap_policy_joints_to_sim_joints_np(policy_joints, *args, **kwargs)
            return out[:, :, ::-1]

    reversing = _Reversing()
    mod._patch_module(reversing)
    with pytest.raises(RuntimeError, match="should occupy simulator index"):
        _remap(reversing, _action(mod))


def test_swapped_arms_are_rejected():
    """Both arms are 7 joints, so a swap preserves the multiset exactly."""
    mod = _adapter()
    _install(mod)

    class _SwappingArms:
        def remap_policy_joints_to_sim_joints_np(self, policy_joints, *args, **kwargs):
            swapped = dict(policy_joints)
            swapped["left_arm"], swapped["right_arm"] = (
                policy_joints["right_arm"], policy_joints["left_arm"])
            return _UpstreamModule(list(mod.GR1_ACTION_GROUPS)) \
                .remap_policy_joints_to_sim_joints_np(swapped, *args, **kwargs)

    swapping = _SwappingArms()
    mod._patch_module(swapping)
    with pytest.raises(RuntimeError, match="should occupy simulator index"):
        _remap(swapping, _action(mod))


def test_a_value_moved_into_an_unmapped_slot_is_rejected():
    """An unmapped mimic slot must stay zero in the command array."""
    mod = _adapter()
    _install(mod)

    class _Leaking:
        def remap_policy_joints_to_sim_joints_np(self, policy_joints, *args, **kwargs):
            out = _UpstreamModule(list(mod.GR1_ACTION_GROUPS)) \
                .remap_policy_joints_to_sim_joints_np(policy_joints, *args, **kwargs)
            out[0, :, mod.GR1_SIM_DOF - 1] = 9.0     # an unmapped slot
            return out

    leaking = _Leaking()
    mod._patch_module(leaking)
    with pytest.raises(RuntimeError, match="must stay zero"):
        _remap(leaking, _action(mod))


def test_prefixed_group_names_reach_upstream_canonicalised():
    """The alias spelling was validated then the ORIGINAL dict was passed on.

    Upstream does not map `action.<group>` keys, so an aliased payload would have been
    zero-filled after validation had already passed.
    """
    mod = _adapter()
    seen = {}

    class _Recording:
        def remap_policy_joints_to_sim_joints_np(self, policy_joints, *args, **kwargs):
            seen["keys"] = sorted(policy_joints)
            return _UpstreamModule(list(mod.GR1_ACTION_GROUPS)) \
                .remap_policy_joints_to_sim_joints_np(policy_joints, *args, **kwargs)

    recording = _Recording()
    mod._patch_module(recording)
    aliased = {f"action.{group}": value for group, value in _action(mod).items()}
    _remap(recording, aliased)
    assert seen["keys"] == sorted(mod.GR1_ACTION_GROUPS), seen["keys"]


def test_the_contract_matches_the_pinned_embodiment_widths():
    """Guards the constants against silent drift."""
    mod = _adapter()
    assert mod.GR1_ACTION_GROUPS == {
        "left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6}
    assert mod.GR1_POLICY_DOF == 26
    assert mod.GR1_SIM_DOF == 36
    assert mod.GR1_UNMAPPED_SIM_SLOTS == 10
    assert mod.GR1_REMAP_TAG == "GR1"


def test_the_patch_stays_idempotent():
    mod = _adapter()
    upstream = _install(mod)
    first = upstream.remap_policy_joints_to_sim_joints_np
    mod._patch_module(upstream)
    assert upstream.remap_policy_joints_to_sim_joints_np is first


def test_the_permutation_cache_is_keyed_by_configuration_not_global():
    """I4: the mapping was cached in ONE global slot keyed on nothing.

    Tag variation is NOT the reachable path -- _validate_remap_tag rejects any tag but GR1
    before learning happens, so a second tag cannot install a second mapping. What the
    unkeyed slot did allow was a mapping learned under one configuration being reused to
    verify calls it never described, if the guarded set is ever widened. The key records the
    configuration the mapping was learned for so that conflation cannot happen silently.
    """
    mod = _adapter()
    upstream = _install(mod)
    upstream.remap_policy_joints_to_sim_joints_np(_action(mod), *_remap_args()[0], **_remap_args()[1])
    assert len(mod._PERMUTATION) == 1
    tag, shape, names, destinations = next(iter(mod._PERMUTATION))
    assert tag == mod.GR1_REMAP_TAG, "the key must record the tag it was learned under"
    assert dict(shape) == dict(mod.GR1_ACTION_GROUPS), (
        "the key must record the group shape it was learned for")


def test_a_rejected_remap_tag_never_reaches_permutation_learning():
    """The tag guard runs first, so no mapping is learned for a tag we refuse."""
    mod = _adapter()
    upstream = _install(mod)
    with pytest.raises(RuntimeError, match="is not 'GR1'"):
        upstream.remap_policy_joints_to_sim_joints_np(_action(mod), *_remap_args("NEW_EMBODIMENT")[0], **_remap_args("NEW_EMBODIMENT")[1])
    assert mod._PERMUTATION == {}


def test_the_same_configuration_learns_only_once():
    """The cache must still avoid re-probing the remapper on every action."""
    mod = _adapter()
    upstream = _install(mod)
    action = _action(mod)
    for _ in range(3):
        upstream.remap_policy_joints_to_sim_joints_np(action, *_remap_args()[0], **_remap_args()[1])
    assert len(mod._PERMUTATION) == 1


def test_the_cache_key_records_the_group_shape():
    """A different group width is a different mapping even under the same tag."""
    mod = _adapter()
    upstream = _install(mod)
    upstream.remap_policy_joints_to_sim_joints_np(_action(mod), *_remap_args()[0], **_remap_args()[1])
    tag, shape, names, destinations = next(iter(mod._PERMUTATION))
    assert tag == mod.GR1_REMAP_TAG
    assert dict(shape) == dict(mod.GR1_ACTION_GROUPS)


def test_the_expected_mapping_matches_the_pinned_configuration():
    """The derivation must reproduce the real pinned Arena layout.

    Computed from the vendored upstream YAMLs, so this fails if the derivation stops
    following configuration order or the pinned configuration changes.
    """
    mod = _adapter()
    policy, sim = _pinned_configs()
    assert mod._derive_permutation(policy, sim, "GR1") == EXPECTED_PERMUTATION


def test_a_remapper_wrong_from_the_very_first_call_is_rejected():
    """I4's core: the expected mapping must not come from the function under test.

    _learn_permutation obtained it by probing the remapper with a canary, so a remapper that
    reversed all 36 output positions on its FIRST call became its own accepted reference.
    Astra's probe demonstrated exactly that. There is no calibration step here.
    """
    mod = _adapter()

    class _WrongFromTheStart:
        def remap_policy_joints_to_sim_joints_np(self, policy_joints, *args, **kwargs):
            out = _UpstreamModule(list(mod.GR1_ACTION_GROUPS)) \
                .remap_policy_joints_to_sim_joints_np(policy_joints, *args, **kwargs)
            return out[:, :, ::-1]

    wrong = _WrongFromTheStart()
    mod._patch_module(wrong)
    with pytest.raises(RuntimeError, match="should occupy simulator index"):
        _remap(wrong, _action(mod))
    assert mod._PERMUTATION, "the mapping must still have been derived from configuration"


def test_an_absent_configuration_fails_loudly():
    """Without configuration the mapping cannot be established; do not fall back."""
    mod = _adapter()
    with pytest.raises(RuntimeError, match="policy_joints_config must be a non-empty dict"):
        mod._derive_permutation(None, {"a": 0}, "GR1")
    with pytest.raises(RuntimeError, match="sim_joints_config must be a non-empty dict"):
        mod._derive_permutation({g: [] for g in mod.GR1_ACTION_GROUPS}, None, "GR1")


def test_a_name_missing_from_the_simulator_configuration_is_named():
    mod = _adapter()
    policy, sim = _pinned_configs()
    broken = {k: v for k, v in sim.items() if k != "left_elbow_pitch_joint"}
    with pytest.raises(RuntimeError, match="left_elbow_pitch_joint"):
        mod._derive_permutation(policy, broken, "GR1")


def test_two_joints_sharing_a_destination_is_rejected():
    """A non-injective placement means one joint's command overwrites another's."""
    mod = _adapter()
    policy, sim = _pinned_configs()
    collided = dict(sim)
    collided["left_shoulder_roll_joint"] = collided["left_shoulder_pitch_joint"]
    with pytest.raises(RuntimeError, match="destination of more than one policy joint"):
        mod._derive_permutation(policy, collided, "GR1")


def test_a_nested_mapping_where_a_list_is_required_is_rejected():
    """Upstream's own annotation claims a mapping here; the pinned YAML supplies a list.

    Implementing against the annotation would read the wrong structure, so the guard rejects
    it rather than silently accepting dict iteration order as joint order.
    """
    mod = _adapter()
    policy, sim = _pinned_configs()
    misread = dict(policy)
    misread["left_arm"] = {name: index for index, name in enumerate(policy["left_arm"])}
    with pytest.raises(RuntimeError, match="must be an ORDERED list"):
        mod._derive_permutation(misread, sim, "GR1")


def test_a_group_of_the_wrong_width_is_rejected():
    mod = _adapter()
    policy, sim = _pinned_configs()
    short = dict(policy)
    short["left_arm"] = policy["left_arm"][:-1]
    with pytest.raises(RuntimeError, match="lists 6 joints but the contract expects 7"):
        mod._derive_permutation(short, sim, "GR1")


def _placing_remapper(mod, policy_config, sim_config):
    """A remapper that places by name using the GIVEN configuration."""
    class _Module:
        def remap_policy_joints_to_sim_joints_np(self, joint_pos, *args, **kwargs):
            out = np.zeros((1, HORIZON, mod.GR1_SIM_DOF), dtype=np.float32)
            for group in mod.GR1_ACTION_GROUPS:
                array = np.asarray(joint_pos[group])
                for column, name in enumerate(policy_config[group]):
                    out[0, :, sim_config[name]] = array[0, :, column]
            return out
    return _Module()


def test_a_changed_destination_does_not_reuse_the_stale_permutation():
    """Important 9: the key was (tag, group widths), which the configuration outlives.

    Changing simulator destination indices while keeping the same tag and the same dimensions
    reused the previously derived permutation, so a remapper still producing the OLD placement
    was accepted. The mapping is a function of the joint names and their destination columns,
    so the key has to capture those.
    """
    mod = _adapter()
    policy, sim = _pinned_configs()
    stale = _placing_remapper(mod, policy, sim)
    mod._patch_module(stale)
    # Valid first call under the original configuration.
    stale.remap_policy_joints_to_sim_joints_np(
        _action(mod), policy, sim, embodiment_tag="GR1")
    assert len(mod._PERMUTATION) == 1

    # Same tag, same dimensions, different destinations.
    swapped = dict(sim)
    swapped["left_shoulder_pitch_joint"], swapped["left_shoulder_roll_joint"] = (
        sim["left_shoulder_roll_joint"], sim["left_shoulder_pitch_joint"])
    with pytest.raises(RuntimeError, match="should occupy simulator index"):
        stale.remap_policy_joints_to_sim_joints_np(
            _action(mod), policy, swapped, embodiment_tag="GR1")
    assert len(mod._PERMUTATION) == 2, "a different configuration must derive its own mapping"


def test_a_reordered_group_is_a_different_configuration():
    """Order IS the placement, so reordering names must not reuse the mapping."""
    mod = _adapter()
    policy, sim = _pinned_configs()
    remapper = _placing_remapper(mod, policy, sim)
    mod._patch_module(remapper)
    remapper.remap_policy_joints_to_sim_joints_np(
        _action(mod), policy, sim, embodiment_tag="GR1")
    reordered = {group: list(names) for group, names in policy.items()}
    reordered["left_arm"] = list(reversed(reordered["left_arm"]))
    with pytest.raises(RuntimeError, match="should occupy simulator index"):
        remapper.remap_policy_joints_to_sim_joints_np(
            _action(mod), reordered, sim, embodiment_tag="GR1")
    assert len(mod._PERMUTATION) == 2


def test_an_identical_configuration_still_derives_only_once():
    """The cache must still avoid re-deriving on every action."""
    mod = _adapter()
    policy, sim = _pinned_configs()
    remapper = _placing_remapper(mod, policy, sim)
    mod._patch_module(remapper)
    for _ in range(3):
        remapper.remap_policy_joints_to_sim_joints_np(
            _action(mod), policy, sim, embodiment_tag="GR1")
    assert len(mod._PERMUTATION) == 1
