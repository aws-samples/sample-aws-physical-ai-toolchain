"""
Cosmos 3 generation runner (text2video / video2video) — the v3 *generation* half.

Cosmos 3 (NVIDIA/cosmos-framework, image physical-ai/cosmos3) GENERATES video from
a text prompt (text2video) or restyles a video loosely from a prompt
(video2video). It does NOT do the controlled sim-to-real *transfer*
(edge/depth/seg) the toolkit uses for domain randomization — that remains Cosmos
Transfer 2.5 (see cosmos_setup.py). Use this for creating novel/varied footage.

Runtime: the cosmos-framework CLI
    python -m cosmos_framework.scripts.inference \
        --parallelism-preset {latency|throughput} \
        -i <spec.json> -o <out> --checkpoint-path {Cosmos3-Nano|Cosmos3-Super}
    (torchrun --nproc-per-node=N for multi-GPU)

We run it as a SageMaker batch Training job on the cosmos3 image (consistent with
the rest of the toolchain; self-terminating). Model weights + the guardrail are
gated HuggingFace downloads pulled at runtime via HF_TOKEN.

Account/region/role/image resolve from the caller — nothing hardcoded.

Usage:
    # Preview (no AWS writes):
    python training/scripts/cosmos3_generate.py --mode text2video \
        --prompt "a robot arm sorting parts on a conveyor, warehouse lighting" --dry-run

    # video2video from an input clip:
    python training/scripts/cosmos3_generate.py --mode video2video \
        --prompt "make it photorealistic, dusty factory" \
        --video s3://<bucket>/clips/in.mp4 --dry-run

NOTE: cosmos-framework has no release tags (built from `main`); this path is
wired correctly but NOT validated end-to-end (needs p5/H100 + HF license accept
for nvidia/Cosmos3-Nano + nvidia/Cosmos-Guardrail1). See docs/ROADMAP.md Feature 3.
"""

import argparse
import json
import os
import sys

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "physical-ai")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
INSTANCE_TYPE = os.environ.get("COSMOS3_INSTANCE_TYPE", "ml.p5.48xlarge")  # 8x H100
HF_SECRET_NAME = os.environ.get("HF_SECRET_NAME", f"{PROJECT_NAME}/hf-token")


def _account() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def _cosmos3_image() -> str:
    return f"{_account()}.dkr.ecr.{REGION}.amazonaws.com/{PROJECT_NAME}/cosmos3:latest"


def _role_arn() -> str:
    return os.environ.get(
        "SAGEMAKER_ROLE_ARN",
        f"arn:aws:iam::{_account()}:role/{PROJECT_NAME}-{ENVIRONMENT}-sagemaker-role",
    )


def _bucket() -> str:
    return os.environ.get("DATASETS_BUCKET", f"{PROJECT_NAME}-{ENVIRONMENT}-datasets-{_account()}")


def _build_spec(mode: str, prompt: str, video: str | None) -> dict:
    """cosmos-framework input spec (verified shape from inputs/omni/*.json)."""
    if mode == "video2video":
        if not video:
            print("ERROR: --video is required for video2video")
            sys.exit(1)
        return {"model_mode": "video2video", "prompt": prompt, "vision_path": video}
    # text2video
    return {"model_mode": "text2video", "name": "t2v", "prompt": prompt}


def run(mode: str, prompt: str, video: str | None, checkpoint: str, preset: str,
        nproc: int, output_s3: str, dry_run: bool):
    job_name = f"cosmos3-{mode}-{int(__import__('time').time())}"
    bucket = _bucket()
    spec = _build_spec(mode, prompt, video)

    # The container runs the cosmos-framework CLI. SageMaker writes /opt/ml/model
    # back to S3, so we point -o there. Spec is written into the container at start.
    launcher = "python -m" if nproc <= 1 else f"torchrun --nproc-per-node={nproc} -m"
    # Per cosmos-framework docs: the `latency` preset on >1 GPU additionally needs
    # --dp-shard-size=1 so ranks are free for context parallelism.
    extra = " --dp-shard-size=1" if (preset == "latency" and nproc > 1) else ""
    inner = (
        "set -e; cd /workspace; "
        f"echo '{json.dumps(spec)}' > /tmp/spec.json; "
        f"{launcher} cosmos_framework.scripts.inference "
        f"--parallelism-preset={preset}{extra} -i /tmp/spec.json "
        f"-o /opt/ml/model/out --checkpoint-path {checkpoint}"
    )

    job_request = {
        "TrainingJobName": job_name,
        "RoleArn": _role_arn(),
        "AlgorithmSpecification": {
            "TrainingImage": _cosmos3_image(),
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["bash", "-lc"],
            "ContainerArguments": [inner],
        },
        "OutputDataConfig": {"S3OutputPath": output_s3 or f"s3://{bucket}/cosmos3/output/"},
        "ResourceConfig": {"InstanceType": INSTANCE_TYPE, "InstanceCount": 1, "VolumeSizeInGB": 500},
        "StoppingCondition": {"MaxRuntimeInSeconds": 2 * 3600},
        "Environment": {"HF_HOME": "/opt/ml/input/data/hf-cache"},
    }

    print(f"{'='*60}")
    print(f"  Cosmos 3 GENERATION ({mode})  [not controlled transfer — that's Cosmos 2.5]")
    print(f"  Job:        {job_name}")
    print(f"  Checkpoint: {checkpoint}  preset={preset}  nproc={nproc}")
    print(f"  Image:      {_cosmos3_image()}")
    print(f"  Output:     {job_request['OutputDataConfig']['S3OutputPath']}")
    print(f"{'='*60}")

    if dry_run:
        print("[dry-run] spec.json:\n" + json.dumps(spec, indent=2))
        print("\n[dry-run] container command:\n  " + inner)
        print("\n[dry-run] No AWS calls made.")
        return job_name

    # HF_TOKEN injected from Secrets Manager (gated weights + guardrail download at runtime).
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        hf = boto3.client("secretsmanager", region_name=REGION).get_secret_value(
            SecretId=HF_SECRET_NAME)["SecretString"]
        job_request["Environment"]["HF_TOKEN"] = hf
    except Exception:
        print(f"  WARN: no HF token at {HF_SECRET_NAME}; gated weight download will fail.")
    sm.create_training_job(**job_request)
    print(f"  Launched. Monitor: aws sagemaker describe-training-job --training-job-name {job_name}")
    return job_name


def main():
    p = argparse.ArgumentParser(description="Cosmos 3 generation runner (text2video / video2video)")
    p.add_argument("--mode", default="text2video", choices=["text2video", "video2video"])
    p.add_argument("--prompt", required=True)
    p.add_argument("--video", default=None, help="(video2video) input clip path/URL")
    p.add_argument("--checkpoint", default="Cosmos3-Nano", choices=["Cosmos3-Nano", "Cosmos3-Super"])
    p.add_argument("--preset", default="latency", choices=["latency", "throughput"])
    p.add_argument("--nproc", type=int, default=8, help="GPUs (torchrun --nproc-per-node); 1 = single-GPU")
    p.add_argument("--output", default=None, help="S3 output prefix")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(args.mode, args.prompt, args.video, args.checkpoint, args.preset,
        args.nproc, args.output, args.dry_run)


if __name__ == "__main__":
    main()
