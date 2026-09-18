"""CPU tests for gr00t_n17_n16_action_adapter -- the N1.7->N1.6 wire shim.

Tests adapter decoding, patching, malformed input handling, and the lazy
import hook. No GPU, no Isaac Sim, no Arena imports.
"""
from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
import pytest

_ADAPTER_PATH = (pathlib.Path(__file__).resolve().parents[1]
                 / "entrypoints/eval/isaac_arena/gr00t"
                 / "gr00t_n17_n16_action_adapter.py")


@pytest.fixture()
def adapter():
    """Load the adapter module fresh (no Isaac deps required)."""
    spec = importlib.util.spec_from_file_location(
        "gr00t_n17_n16_action_adapter", _ADAPTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDecodeValue:
    def test_ndarray_passthrough(self, adapter):
        arr = np.ones((1, 10, 7), dtype=np.float32)
        result = adapter._decode_value(arr)
        np.testing.assert_array_equal(result, arr)

    def test_msgpack_dict_bytes_keys(self, adapter):
        raw = np.arange(7, dtype=np.float32)
        encoded = {b"nd": True, b"type": b"<f4",
                   b"shape": [1, 7], b"data": raw.tobytes()}
        result = adapter._decode_value(encoded)
        assert result.shape == (1, 7)
        np.testing.assert_allclose(result, raw.reshape(1, 7))

    def test_msgpack_dict_str_keys(self, adapter):
        raw = np.zeros((1, 10, 6), dtype=np.float32)
        encoded = {"nd": True, "type": "<f4",
                   "shape": [1, 10, 6], "data": raw.tobytes()}
        result = adapter._decode_value(encoded)
        assert result.shape == (1, 10, 6)

    def test_one_dim_object_array(self, adapter):
        """A 1-D object array, which is what np.array(<1-D>, dtype=object) produces.

        I20: this was named test_zero_dim_object_array, but np.array(np.ones(7),
        dtype=object) has shape (7,) -- the name described a case it never built. The
        genuine 0-D box is exercised separately below.
        """
        inner = np.ones(7, dtype=np.float32)
        boxed = np.array(inner, dtype=object)
        assert boxed.shape == (7,)
        result = adapter._decode_value(boxed)
        np.testing.assert_array_equal(result, inner)

    def test_zero_dim_object_array(self, adapter):
        """A true 0-D object array wrapping an ndarray, as msgpack decoding can yield."""
        inner = np.ones(7, dtype=np.float32)
        boxed = np.empty((), dtype=object)
        boxed[()] = inner
        assert boxed.shape == ()
        result = adapter._decode_value(boxed)
        np.testing.assert_array_equal(result, inner)


class TestPatchModule:
    def test_patches_function(self, adapter):
        fake_mod = ModuleType("fake_joints")
        original_called = []

        def original_remap(joint_pos, *a, **kw):
            original_called.append(joint_pos)
            return joint_pos

        setattr(fake_mod, adapter._FUNC, original_remap)
        adapter._patch_module(fake_mod)
        patched = getattr(fake_mod, adapter._FUNC)
        assert getattr(patched, "_gr00t_n17_n16_action_adapter", False)

    def test_patch_target_missing_fails_loud(self, adapter):
        """A module without the target function means the adapter is NOT installed.

        It used to `return` quietly, so the rollout continued with undecoded N1.7
        actions while still being reported as a checkpoint evaluation. Absence of the
        wrap target is a hard error, not a no-op.
        """
        empty = ModuleType("fake_joints_without_target")
        with pytest.raises(RuntimeError, match="not found"):
            adapter._patch_module(empty)

    def test_patch_is_idempotent(self, adapter):
        """Patching twice must NOT raise -- only a MISSING target is an error."""
        fake_mod = ModuleType("fake_joints_twice")
        setattr(fake_mod, adapter._FUNC, lambda joint_pos, *a, **kw: joint_pos)
        adapter._patch_module(fake_mod)
        first = getattr(fake_mod, adapter._FUNC)
        adapter._patch_module(fake_mod)          # second call: no-op, no raise
        assert getattr(fake_mod, adapter._FUNC) is first

    def test_patched_decodes_dict_values(self, adapter):
        fake_mod = ModuleType("fake_joints")
        received = []

        def original_remap(joint_pos, policy_joints_config=None, sim_joints_config=None,
                           **kw):
            received.append(joint_pos)
            # Place by joint NAME, as pinned upstream does. Filling columns 0..25 in order
            # is not upstream's layout, so the positional check would reject it -- this
            # test is about DECODING and must not fail for the wrong reason.
            first = np.asarray(joint_pos[next(iter(adapter.GR1_ACTION_GROUPS))])
            out = np.zeros((1, first.shape[1], adapter.GR1_SIM_DOF), dtype=first.dtype)
            for group in adapter.GR1_ACTION_GROUPS:
                array = np.asarray(joint_pos[group])
                for column, name in enumerate(policy_joints_config[group]):
                    out[0, :, sim_joints_config[name]] = array[0, :, column]
            return out

        setattr(fake_mod, adapter._FUNC, original_remap)
        adapter._patch_module(fake_mod)
        patched = getattr(fake_mod, adapter._FUNC)
        adapter._logged.clear()
        # A full, contract-valid action chunk: the adapter now enforces the GR1 action
        # contract before remapping, so a two-group 1-D stub would be rejected before
        # the decode path was reached.
        horizon = 16
        action = {}
        for group, joints in adapter.GR1_ACTION_GROUPS.items():
            action[group] = np.ones((1, horizon, joints), dtype=np.float32)
        # Deliver left_arm the way the N1.7 server does: a msgpack-numpy dict.
        raw = np.ones((1, horizon, 7), dtype=np.float32)
        action["left_arm"] = {b"data": raw.tobytes(), b"type": b"<f4",
                              b"shape": list(raw.shape)}
        import yaml
        config_dir = pathlib.Path(__file__).with_name("data") / "arena_gr1"
        policy_config = yaml.safe_load(
            (config_dir / "gr00t_26dof_joint_space.yaml").read_text())["joints"]
        sim_config = yaml.safe_load(
            (config_dir / "36dof_joint_space.yaml").read_text())["joints"]
        patched(action, policy_config, sim_config, embodiment_tag="GR1")
        # ONE call reaches the original. The adapter used to probe it first with a canary
        # action to learn the permutation; that was circular and is gone -- the expected
        # mapping now comes from the configuration above (I4).
        assert len(received) == 1
        decoded = received[-1]
        assert isinstance(decoded["left_arm"], np.ndarray)
        assert decoded["left_arm"].dtype != object
        assert decoded["left_arm"].shape == (1, horizon, 7)


class TestCompatFinder:
    def test_finder_installs(self, adapter):
        finders = [f for f in sys.meta_path
                   if type(f).__name__ == "_CompatFinder"]
        assert len(finders) >= 1
