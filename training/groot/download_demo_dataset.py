"""
Download a demo dataset for GR00T fine-tuning.

Downloads a public LeRobot dataset from HuggingFace and saves it locally
in the format expected by the training pipeline.

Usage:
    python download_demo_dataset.py --output ./data/demo-dataset
    python download_demo_dataset.py --output ./data/demo-dataset --dataset lerobot/aloha_sim_insertion_human

WORKSHOP NOTE: Run this before uploading to S3. The demo dataset is small (~500MB)
and contains 50 episodes of a simulated manipulation task.
"""

import argparse
import os
import json
from pathlib import Path


def download_dataset(dataset_name: str, output_dir: str) -> dict:
    """Download a LeRobot dataset from HuggingFace."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Demo Dataset Download")
    print(f"  Source: huggingface.co/{dataset_name}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    try:
        from huggingface_hub import snapshot_download

        print(f"\n  Downloading from HuggingFace Hub...")
        local_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=str(output_path),
        )

        # Verify the dataset has the expected structure
        expected_dirs = ["data", "meta"]
        found = [d for d in expected_dirs if (output_path / d).exists()]
        missing = [d for d in expected_dirs if d not in found]

        if missing:
            # Some datasets have slightly different structures
            print(f"  WARNING: Missing expected directories: {missing}")
            print(f"  Found: {list(output_path.iterdir())}")

        # Count episodes
        parquet_files = list(output_path.rglob("*.parquet"))
        video_files = list(output_path.rglob("*.mp4"))

        result = {
            "status": "success",
            "dataset": dataset_name,
            "output_dir": str(output_path),
            "parquet_files": len(parquet_files),
            "video_files": len(video_files),
            "total_size_mb": sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1024 / 1024,
        }

        print(f"\n  Download complete!")
        print(f"  Parquet files: {result['parquet_files']}")
        print(f"  Video files: {result['video_files']}")
        print(f"  Total size: {result['total_size_mb']:.0f} MB")
        print(f"\n  Next step: Upload to S3")
        print(f"    aws s3 sync {output_path} s3://<DATASETS_BUCKET>/groot-data/demo/dataset/")

        return result

    except ImportError:
        print("\n  ERROR: huggingface_hub not installed.")
        print("  Install with: pip install huggingface-hub")
        print("\n  Alternative: manually download from")
        print(f"    https://huggingface.co/datasets/{dataset_name}")
        return {"status": "error", "message": "huggingface_hub not installed"}

    except Exception as e:
        print(f"\n  ERROR: {e}")
        return {"status": "error", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Download a demo dataset for GR00T training")
    parser.add_argument("--output", default="./data/demo-dataset",
                        help="Output directory for the dataset")
    parser.add_argument("--dataset", default="lerobot/aloha_sim_insertion_human",
                        help="HuggingFace dataset ID to download")
    args = parser.parse_args()

    result = download_dataset(args.dataset, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
