"""The four local test cells; model images and suites come from the registry."""
from __future__ import annotations

PROFILES = {
    "arena-gr1": ("gr00t", "isaac_arena", "arena_gr1_fridge"),
    "openvla-libero": ("openvla", "libero", "libero_spatial"),
    "molmoact2-libero": ("molmoact2", "libero", "libero_spatial"),
    "gr00t-libero": ("gr00t", "libero", "libero_spatial"),
}


def execution_parameters(parameters, declarations):
    """Keep empty declared defaults in the manifest, not API parameter overrides."""
    defaults = {parameter.name: parameter.default_value for parameter in declarations.values()}
    overrides = {}
    for name, value in parameters.items():
        if value == "":
            if defaults.get(name) != "":
                raise RuntimeError(f"Empty override is not the declared default for {name}")
        else:
            overrides[name] = value
    return overrides


def libero_parameters(profile, declarations, registry, resolve_digest, *,
                      train_image=None, eval_image=None):
    """Mirror run_libero.py's train-default recipe without submitting a cloud job."""
    from vla_pipeline.registry import resolve, resolve_suite

    family, simulator, suite_name = PROFILES[profile]
    if simulator != "libero":
        raise ValueError("Use the complete managed export for the Arena profile")
    spec = resolve(family, simulator)
    suite = resolve_suite(suite_name)
    values = {parameter.name: parameter.default_value
              for parameter in declarations.values()
              if parameter.default_value is not None}
    values.update(
        ModelFamily=family, ModelPackageGroupName=spec.registry_group,
        Suite=suite_name, TrainSuite="unified" if family == "molmoact2" else suite_name,
        DatasetS3Uri="__LIBERO_DEFAULT__",
        TrainImageUri=(train_image if train_image is not None
                       else f"{registry}/{spec.train_image_repo}"),
        EvalImageUri=resolve_digest(eval_image if eval_image is not None
                                    else f"{registry}/{spec.eval_image_repo}"),
        TrainInstanceType="local_gpu", EvalInstanceType="local_gpu",
        VolumeSizeInGB=spec.train_volume_gb, EvalVolumeSizeInGB=spec.eval_volume_gb,
        TrainSourceDirUri="__STAGED_LOCAL_SOURCE__",
        EvalSourceDirUri="__STAGED_LOCAL_SOURCE__",
        TrainSteps=200, EvalTrials=3, EvalSeed=1000, EvalTaskIds="all",
        SuccessThreshold=0.0, Gr00tVersion="n17",
        UseGrootServer=spec.use_groot_server, ArenaConnector=spec.arena_connector,
    )
    if suite.dataset and suite.dataset.revision:
        values["DatasetRevision"] = suite.dataset.revision
    missing = {p.name for p in declarations.values()} - values.keys()
    if missing:
        raise RuntimeError(f"Local recipe omitted pipeline parameters: {sorted(missing)}")
    return values


def expected_episodes(params):
    from vla_pipeline.registry import resolve_suite

    suite = resolve_suite(params["Suite"])
    if suite.simulator == "isaac_arena":
        return params["EvalTrials"]
    if params["EvalTaskIds"] != "all":
        raise RuntimeError("The local compatibility profiles require the complete spatial suite")
    return len(suite.canonical_task_ids) * params["EvalTrials"]


def check_training_contract(receipt, params):
    """Retain the validator's different claim strengths across model families."""
    contract = receipt["training_contract"]
    if params.get("CheckpointS3Uri"):
        if contract != {
            "train_path": "eval_only", "fields": [], "matched": [], "verified": [], "attested": [],
            "reason": "checkpoint_training_is_outside_this_execution",
        }:
            raise RuntimeError("Supplied checkpoint must not claim training lineage from this run")
        return contract
    statuses = {row["field"]: row["status"] for row in contract["fields"]}
    if statuses.get("train_steps") != "matched":
        raise RuntimeError("Validate did not match the requested training dose")
    if params["ModelFamily"] == "gr00t":
        expected = {
            "train_steps": "matched", "dataset_revision": "verified",
            "dataset_source": "attested", "dataset_subdirectory": "attested",
            "dataset_content_digest": "attested", "train_suite": "attested",
        }
        if statuses != expected or contract.get("verified") != ["dataset_revision"]:
            raise RuntimeError("Receipt did not qualify GR00T training lineage correctly")
    else:
        if contract.get("verified") or contract.get("attested"):
            raise RuntimeError("Unexpected stronger lineage claim for a legacy family")
        if statuses != {
            "train_steps": "matched", "dataset_source": "recorded_only",
            "dataset_revision": "recorded_only", "train_suite": "unavailable",
        }:
            raise RuntimeError("Receipt did not preserve the legacy training-lineage limits")
    if receipt.get("model_family") not in (None, params["ModelFamily"]):
        raise RuntimeError("Receipt belongs to a different model family")
    if receipt.get("suite") != params["Suite"]:
        raise RuntimeError("Receipt belongs to a different evaluation suite")
    if params["Suite"] == "libero_spatial":
        # The full task set is additionally enforced inside Validate against the
        # registry. This check refuses a misleading task-subset parameter.
        if params["EvalTaskIds"] != "all":
            raise RuntimeError("LIBERO spatial coverage requires all ten tasks")
    return contract
