"""Full round-trip test for scriptify_policy.py — the one piece fully verifiable
without a GPU: rsl_rl checkpoint → TorchScript → torch.jit.load → forward."""
import pathlib

import pytest

torch = pytest.importorskip("torch")
REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_module():
    import importlib.util
    p = REPO / "training/scripts/scriptify_policy.py"
    spec = importlib.util.spec_from_file_location("scriptify_policy", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scriptify_roundtrip(tmp_path):
    import torch.nn as nn
    mod = _load_module()

    obs, act, hid = 12308, 7, [256, 256, 256]
    # Build a matching actor and save it rsl_rl-style.
    layers, prev = [], obs
    for h in hid:
        layers += [nn.Linear(prev, h), nn.ELU()]; prev = h
    layers += [nn.Linear(prev, act)]
    net = nn.Sequential(*layers)
    sd = {f"actor.{k}": v for k, v in net.state_dict().items()}
    ckpt = tmp_path / "model_49.pt"
    out = tmp_path / "scripted.pt"
    torch.save({"model_state_dict": sd}, ckpt)

    # Run the scriptify entry point.
    import sys
    argv = ["scriptify_policy.py", "--checkpoint", str(ckpt), "--output", str(out),
            "--obs-dim", str(obs), "--action-dim", str(act)]
    old = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = old

    assert out.exists(), "TorchScript model was not written"
    # Load exactly as export.py does, and run a forward pass.
    m = torch.jit.load(str(out), map_location="cpu")
    m.eval()
    y = m(torch.randn(1, obs))
    assert tuple(y.shape) == (1, act)


def test_scriptify_infers_dims_without_flags(tmp_path):
    """No --obs-dim/--action-dim/--hidden-dims: arch is read from the checkpoint."""
    import sys

    import torch.nn as nn
    mod = _load_module()

    # An Anymal-shaped actor (48 -> [128,128,128] -> 12), saved rsl_rl-style.
    obs, act, hid = 48, 12, [128, 128, 128]
    layers, prev = [], obs
    for h in hid:
        layers += [nn.Linear(prev, h), nn.ELU()]; prev = h
    layers += [nn.Linear(prev, act)]
    sd = {f"actor.{k}": v for k, v in nn.Sequential(*layers).state_dict().items()}
    ckpt = tmp_path / "model_50.pt"
    out = tmp_path / "scripted.pt"
    torch.save({"model_state_dict": sd}, ckpt)

    argv = ["scriptify_policy.py", "--checkpoint", str(ckpt), "--output", str(out)]
    old = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = old

    assert out.exists()
    m = torch.jit.load(str(out), map_location="cpu")
    m.eval()
    y = m(torch.randn(1, obs))
    assert tuple(y.shape) == (1, act)


def test_infer_arch_reads_weight_shapes():
    import torch.nn as nn
    mod = _load_module()
    obs, act, hid = 48, 12, [128, 128, 128]
    layers, prev = [], obs
    for h in hid:
        layers += [nn.Linear(prev, h), nn.ELU()]; prev = h
    layers += [nn.Linear(prev, act)]
    actor = {k: v for k, v in nn.Sequential(*layers).state_dict().items()}
    assert mod._infer_arch(actor) == (obs, act, hid)


def test_extract_state_dict_variants():
    mod = _load_module()
    sd = {"actor.net.0.weight": torch.zeros(1)}
    assert mod._extract_state_dict({"model_state_dict": sd}) is sd
    assert mod._extract_state_dict({"state_dict": sd}) is sd
    assert mod._extract_state_dict(sd) is sd
    # actor-prefix stripping
    stripped = mod._actor_weights({"actor.net.0.weight": torch.zeros(1), "critic.x": torch.zeros(1)})
    assert "net.0.weight" in stripped and "critic.x" not in str(stripped)
