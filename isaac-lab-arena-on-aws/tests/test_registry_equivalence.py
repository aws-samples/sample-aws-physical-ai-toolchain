"""Registry resolution regression guard + launcher-consumption proof.

Originally (commit 281e324) this asserted resolve() was byte-identical to the
launchers' hardcoded maps. After the launcher rewire, those maps are GONE --
the registry is the single source of truth. This test now:
  1. pins resolve() output for every real pair to the canonical expected values
     (regression guard against manifest drift), and
  2. proves the launchers consume the registry (no resurrected hardcoded maps),
     so "adding a model = a manifest" holds.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from vla_pipeline.registry import resolve  # noqa: E402

# Canonical expected adapter identity per real pair (the values that used to be
# hardcoded in run_arena.FAMILY / run_libero._get_images).
_EXPECT = {
    ("gr00t", "isaac_arena"): dict(
        # The DERIVED connector image, in its own repo. `vla/isaac-arena` now
        # holds only the external Isaac Sim + Arena BASE image; building the
        # connector into that same repo made it its own FROM (see
        # README.md#arena-images-and-connector-contract).
        train="vla/gr00t:1.2", eval="physical-ai/isaac-lab-arena:v11",
        connector="groot", ugs="true", suite="arena_gr1_fridge",
        group="vla-arena-gr00t", delivery="baked"),
    # NOTE: (openvla, isaac_arena) and (molmoact2, isaac_arena) are intentionally
    # ABSENT -- those cells are embodiment-blocked (7-DoF Franka policies cannot
    # emit the 26-dim GR1 action) and excluded from the component (their pair
    # manifests are deleted). See the README compatibility matrix. They must NOT
    # resolve; run_arena._ARENA_FAMILIES is registry-derived so it excludes them.
    ("gr00t", "libero"): dict(
        train="vla/gr00t:1.2", eval="vla/gr00t:1.2",
        connector="groot", ugs="false", suite="libero_spatial",
        group="vla-eval-gr00t", delivery="sourcedir"),
    ("openvla", "libero"): dict(
        train="vla/openvla:smoke-v2", eval="vla/openvla:smoke-v2",
        connector="groot", ugs="false", suite="libero_spatial",
        group="vla-eval-openvla", delivery="sourcedir"),
    ("molmoact2", "libero"): dict(
        train="vla/molmoact2:smoke-v2", eval="vla/molmoact2:smoke-v2",
        connector="groot", ugs="false", suite="libero_spatial",
        group="vla-eval-molmoact2", delivery="sourcedir"),
}


def test_resolve_matches_canonical_expected():
    for (fam, sim), exp in _EXPECT.items():
        spec = resolve(fam, sim)
        assert spec.train_image_repo == exp["train"], (fam, sim)
        assert spec.eval_image_repo == exp["eval"], (fam, sim)
        assert spec.arena_connector == exp["connector"], (fam, sim)
        assert spec.use_groot_server == exp["ugs"], (fam, sim)
        assert spec.default_suite == exp["suite"], (fam, sim)
        assert spec.registry_group == exp["group"], (fam, sim)
        assert spec.delivery_mode == exp["delivery"], (fam, sim)


def test_arena_eval_image_is_not_its_own_base():
    """The connector image must not live in the same ECR repo as its base.

    Base and connector repositories remain separate build targets.
    """
    spec = resolve("gr00t", "isaac_arena")
    assert spec.eval_base_image_repo, (
        "the Arena pair must declare eval_base_image -- it is an external "
        "prerequisite this repo does not build (README.md#arena-images-and-connector-contract)")
    eval_repo = spec.eval_image_repo.rsplit(":", 1)[0]
    base_repo = spec.eval_base_image_repo.rsplit(":", 1)[0]
    assert eval_repo != base_repo, (
        f"eval_image and eval_base_image share the repo {eval_repo!r}")


def _load_script(module_name: str):
    prev = os.getcwd()
    os.chdir(_REPO_ROOT)  # launchers do sys.path.insert(0, "src") (cwd-relative)
    try:
        path = os.path.join(_REPO_ROOT, "scripts", f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(prev)


def test_launchers_consume_registry_no_hardcoded_maps():
    run_arena = _load_script("run_arena")
    run_libero = _load_script("run_libero")
    run_matrix = _load_script("run_matrix")
    # The old hardcoded maps must be gone -- registry is the single source.
    assert not hasattr(run_arena, "FAMILY"), "run_arena resurrected a hardcoded FAMILY map"
    assert not hasattr(run_libero, "_get_images"), "run_libero resurrected _get_images"
    assert not hasattr(run_matrix, "IMAGES"), "run_matrix resurrected a hardcoded IMAGES map"
    # And the registry-derived family lists match the real families.
    # Arena advertises ONLY gr00t: OpenVLA/MolmoAct2 x Arena are embodiment-blocked
    # and excluded (their pair manifests are deleted), and _ARENA_FAMILIES is
    # registry-derived, so the launcher's --family choices exclude them.
    assert set(run_arena._ARENA_FAMILIES) == {"gr00t"}
    assert set(run_libero._LIBERO_FAMILIES) == {"gr00t", "openvla", "molmoact2"}
