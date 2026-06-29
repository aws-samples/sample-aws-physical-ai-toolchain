"""
Convert an RL checkpoint into a TorchScript module that export.py can consume.

The gap this closes: `export.py` does `torch.jit.load(...)` (it expects a
TorchScript model), but rsl_rl / Isaac Lab RL save plain checkpoints
(`model_*.pt` = a dict with 'model_state_dict', not a scripted module). Feeding
those straight to export.py fails at load.

This script loads an rsl_rl actor checkpoint, rebuilds the policy MLP, loads the
weights, and `torch.jit.script`s it to a standalone `.pt` TorchScript file. The
scripted module takes a single flattened observation tensor (matching what the
edge inference node + export.py expect) and returns the action.

Usage:
    # Inspect a checkpoint's contents (no output written):
    python training/scripts/scriptify_policy.py --checkpoint model_49.pt --inspect

    # Produce a TorchScript model for export.py (dims inferred from the checkpoint):
    python training/scripts/scriptify_policy.py \
        --checkpoint model_49.pt \
        --output policy_scripted.pt

Then:
    python training/scripts/export.py --checkpoint policy_scripted.pt \
        --output-onnx policy.onnx --output-trt policy.trt --target-device jetson-orin --fp16

NOTE: obs/action/hidden dims are read straight from the checkpoint's weight
shapes — no need to know the architecture. Override with --obs-dim /
--action-dim / --hidden-dims if needed. The activation is NOT stored in the
checkpoint and defaults to ELU (matches rsl_rl / Isaac Lab); override with
--activation if you trained with something else.
"""

import argparse
import sys

import torch
import torch.nn as nn

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class PolicyMLP(nn.Module):
    """A plain MLP actor: flattened obs -> action. Matches rsl_rl's actor head."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims, activation: str):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers, prev = [], obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers += [nn.Linear(prev, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _extract_state_dict(ckpt: dict) -> dict:
    """rsl_rl checkpoints store the actor-critic under 'model_state_dict'."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt  # assume it's already a state dict


def _actor_weights(state_dict: dict) -> dict:
    """Pull just the actor MLP weights (rsl_rl prefixes them with 'actor.')."""
    actor = {k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")}
    return actor or state_dict


def _infer_arch(actor: dict):
    """Read (obs_dim, action_dim, hidden_dims) straight from the Linear weight
    shapes so the user never has to hand-type them. rsl_rl stores the actor as a
    Sequential, so weight keys look like '<idx>.weight' with shape (out, in).
    Returns None if the layers can't be parsed (caller falls back to flags)."""
    linears = []
    for k, v in actor.items():
        if k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 2:
            try:
                idx = int(k.split(".")[0])
            except (ValueError, IndexError):
                continue
            linears.append((idx, tuple(v.shape)))
    if not linears:
        return None
    linears.sort(key=lambda t: t[0])
    shapes = [s for _, s in linears]
    obs_dim = shapes[0][1]           # in_features of first layer
    action_dim = shapes[-1][0]       # out_features of last layer
    hidden_dims = [out for out, _ in shapes[:-1]]  # out_features of all but last
    return obs_dim, action_dim, hidden_dims


def main():
    p = argparse.ArgumentParser(description="Scriptify an RL checkpoint for export.py")
    p.add_argument("--checkpoint", required=True, help="rsl_rl model_*.pt checkpoint")
    p.add_argument("--output", help="Output TorchScript .pt path (required unless --inspect)")
    p.add_argument("--obs-dim", type=int, default=None,
                   help="Override; inferred from the checkpoint if omitted")
    p.add_argument("--action-dim", type=int, default=None,
                   help="Override; inferred from the checkpoint if omitted")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=None,
                   help="Override; inferred from the checkpoint if omitted")
    p.add_argument("--activation", default="elu", choices=list(ACTIVATIONS),
                   help="Activation (not stored in the checkpoint; default elu)")
    p.add_argument("--inspect", action="store_true", help="Print checkpoint keys and exit")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)

    if args.inspect:
        print(f"  Checkpoint top-level type: {type(ckpt).__name__}")
        if isinstance(ckpt, dict):
            print(f"  Top-level keys: {list(ckpt.keys())[:20]}")
        actor = _actor_weights(state_dict)
        print(f"  Actor param tensors: {len(actor)}")
        for k, v in list(actor.items())[:8]:
            print(f"    {k}: {tuple(v.shape)}")
        return

    if not args.output:
        print("ERROR: --output is required (or use --inspect)")
        sys.exit(1)

    actor = _actor_weights(state_dict)

    # Infer architecture from the checkpoint's weight shapes; CLI flags override.
    inferred = _infer_arch(actor)
    obs_dim, action_dim, hidden_dims = args.obs_dim, args.action_dim, args.hidden_dims
    if inferred is not None:
        i_obs, i_act, i_hidden = inferred
        obs_dim = obs_dim if obs_dim is not None else i_obs
        action_dim = action_dim if action_dim is not None else i_act
        hidden_dims = hidden_dims if hidden_dims is not None else i_hidden
    if obs_dim is None or action_dim is None or hidden_dims is None:
        print("ERROR: could not infer architecture from checkpoint; pass "
              "--obs-dim/--action-dim/--hidden-dims explicitly (or use --inspect to see shapes)")
        sys.exit(1)
    print(f"  Architecture: obs={obs_dim} action={action_dim} hidden={hidden_dims} "
          f"({'inferred' if inferred else 'from flags'})")

    model = PolicyMLP(obs_dim, action_dim, hidden_dims, args.activation)
    missing, unexpected = model.net.load_state_dict(actor, strict=False)
    if missing or unexpected:
        print(f"  WARN: state_dict mismatch — missing={len(missing)} unexpected={len(unexpected)}")
        print(f"        (check --hidden-dims/--activation match how the policy was trained)")
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(args.output)
    print(f"  Wrote TorchScript model: {args.output}")
    print(f"  Next: export.py --checkpoint {args.output} --output-onnx ... --output-trt ...")


if __name__ == "__main__":
    main()
