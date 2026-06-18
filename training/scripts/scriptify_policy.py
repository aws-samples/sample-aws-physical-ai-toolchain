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

    # Produce a TorchScript model for export.py:
    python training/scripts/scriptify_policy.py \
        --checkpoint model_49.pt \
        --output policy_scripted.pt \
        --obs-dim 12308 --action-dim 7

Then:
    python training/scripts/export.py --checkpoint policy_scripted.pt \
        --output-onnx policy.onnx --output-trt policy.trt --target-device jetson-orin --fp16

NOTE: the exact actor architecture (hidden sizes/activation) must match how the
policy was trained. Defaults mirror training/configs/ppo_pick_place.yaml
(3x256, ELU). Override with --hidden-dims / --activation. Validated only at the
scripting level here; a real checkpoint round-trip needs a trained model.
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


def main():
    p = argparse.ArgumentParser(description="Scriptify an RL checkpoint for export.py")
    p.add_argument("--checkpoint", required=True, help="rsl_rl model_*.pt checkpoint")
    p.add_argument("--output", help="Output TorchScript .pt path (required unless --inspect)")
    p.add_argument("--obs-dim", type=int, default=12308)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--activation", default="elu", choices=list(ACTIVATIONS))
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

    model = PolicyMLP(args.obs_dim, args.action_dim, args.hidden_dims, args.activation)
    actor = _actor_weights(state_dict)
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
