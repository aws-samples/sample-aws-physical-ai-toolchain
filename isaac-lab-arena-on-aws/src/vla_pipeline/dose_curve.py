"""Dose-curve: an OPTIONAL, default-OFF layer over the core pipeline.

The core pipeline is train -> eval -> validate -> gate -> register on the FINAL
checkpoint; run it N times for N independent comparisons. The *dose curve* (one
training trajectory that saves intermediate checkpoints, then evaluates a subset
of them to plot success-vs-training-steps) is an opt-in extra. This module is the
single place that knowledge lives, so the core path carries no dose branching.

DEFAULT-OFF BY CONSTRUCTION
---------------------------
A dose curve is active iff the trainer is told to save intermediate checkpoints,
i.e. the save interval (``save_steps``) is strictly below the training length
(``max_steps``). The pipeline's ``TrainSaveSteps`` default is
``FINAL_ONLY_SAVE_STEPS`` (>> any real ``max_steps``), so by default exactly one
(final) checkpoint is written -- the historical single-checkpoint behaviour,
byte-for-byte. ``enabled`` is therefore ``False`` unless a caller lowers
``save_steps``.

CORRECTNESS vs PERFORMANCE
--------------------------
When dose is enabled the run needs a SINGLE GPU: intermediate ``checkpoint-<step>/``
dirs must be standalone HF checkpoints (loadable by ``from_pretrained``), but
multi-GPU training writes DeepSpeed ZeRO shards that are not standalone-loadable.
Forcing single-GPU only changes SPEED, not the RESULT, so ``validate_preconditions``
WARNS and auto-downgrades -- it never hard-fails (a hard error is reserved for
correctness failures that would change the result). A per-family escape hatch
(``supports_multi_gpu_intermediates``) is provided for future backends that can
consolidate shards.
"""
from __future__ import annotations

from dataclasses import dataclass

# The historical "final-only" save interval. ``TrainSaveSteps`` defaults to this
# (>> any real ``max_steps``) so exactly one final checkpoint is written = the
# byte-identical single-checkpoint path. A dose-curve run lowers it (e.g. 2000).
FINAL_ONLY_SAVE_STEPS = 1_000_000

# The pipeline parameter names this layer -- and ONLY this layer -- owns.
DOSE_PARAM_NAMES = ("TrainSaveSteps", "EvalDoseSteps", "VolumeSizeInGB")


@dataclass(frozen=True)
class DoseCurveConfig:
    """Immutable description of whether/how a run collects a dose curve.

    Attributes:
        max_steps: total fine-tune steps for the trajectory (== ``TrainSteps``).
        save_steps: intermediate-checkpoint save interval. Default = final-only.
        eval_dose_steps: which staged checkpoints to roll out -- ``"all"`` or a
            comma list of step counts. Non-empty (SageMaker rejects empty params).
        volume_size_gb: EBS size; a curve stages N checkpoints so it needs more.
        supports_multi_gpu_intermediates: escape hatch for backends that can
            consolidate ZeRO shards into loadable checkpoints (default False).
    """

    max_steps: int
    save_steps: int = FINAL_ONLY_SAVE_STEPS
    eval_dose_steps: str = "all"
    #: An OVERRIDE, not a default. This was an unconditional 100, and inject_dose_params() wrote it
    #: over whatever volume the caller had already resolved from the family manifest -- so enabling a
    #: dose curve silently reverted a family declaring 300 GB back to 100. A curve run legitimately
    #: needs MORE disk (it stages N intermediate checkpoints), which is the opposite of what an
    #: unconditional 100 produced.
    volume_size_gb: int | None = None
    supports_multi_gpu_intermediates: bool = False

    @property
    def enabled(self) -> bool:
        """True iff the trainer saves intermediate checkpoints (curve run)."""
        return self.save_steps < self.max_steps

    def validate_preconditions(self, requested_num_gpus: int) -> tuple[int, list[str]]:
        """Return ``(effective_num_gpus, warnings)``.

        No-op (returns the request unchanged, no warnings) when dose is disabled.
        When enabled and multi-GPU was requested without the escape hatch, WARNS
        and downgrades to a single GPU -- a result-preserving performance
        downgrade, never a hard error.
        """
        warnings: list[str] = []
        if not self.enabled:
            return requested_num_gpus, warnings
        if requested_num_gpus > 1 and not self.supports_multi_gpu_intermediates:
            warnings.append(
                f"dose-curve enabled (save_steps={self.save_steps} < "
                f"max_steps={self.max_steps}): forcing single-GPU (requested "
                f"{requested_num_gpus}) so each intermediate checkpoint is a "
                f"standalone HF checkpoint rather than an unloadable ZeRO shard. "
                f"Slower, but the RESULT is unchanged."
            )
            return 1, warnings
        return requested_num_gpus, warnings

    def inject_dose_params(self, params: dict) -> dict:
        """Return a copy of the submit-time SageMaker ``params`` dict (keys are
        pipeline parameter names) with the dose-only knobs applied.

        When dose is DISABLED this is a pure pass-through: the core submit path
        never sees a dose parameter. Only the enabled branch sets the three
        ``DOSE_PARAM_NAMES``. Values are stringified (SageMaker parameter values
        are strings).
        """
        out = dict(params)
        if not self.enabled:
            return out
        out["TrainSaveSteps"] = str(self.save_steps)
        out["EvalDoseSteps"] = self.eval_dose_steps
        # Written ONLY when explicitly set. Absence is `is None`, so the caller's resolved value
        # survives; an explicit 0 is not treated as absent.
        if self.volume_size_gb is not None:
            if isinstance(self.volume_size_gb, bool) or self.volume_size_gb <= 0:
                raise ValueError(
                    f"dose volume_size_gb must be a positive int when set, got "
                    f"{self.volume_size_gb!r}")
            out["VolumeSizeInGB"] = str(self.volume_size_gb)
        return out
