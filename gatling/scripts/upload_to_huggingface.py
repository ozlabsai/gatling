"""
Upload Gatling adversarial dataset to HuggingFace Hub.

This script uploads the generated dataset to HuggingFace after the repo is created.
Requires the repo to be created first at: https://huggingface.co/new-dataset

Usage:
    python scripts/upload_to_huggingface.py --repo OzLabs/gatling-adversarial-563k
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file, create_repo


def upload_dataset(repo_id: str, dataset_path: str, readme_path: str, metadata_path: str) -> None:
    """
    Upload dataset files to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'OzLabs/gatling-adversarial-563k')
        dataset_path: Path to the JSONL dataset file
        readme_path: Path to the README.md dataset card
        metadata_path: Path to the metadata JSON file
    """
    # Load HF token from environment
    load_dotenv()
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise ValueError(
            "HUGGINGFACE_API_TOKEN not found in environment. "
            "Please set it in .env file or export HUGGINGFACE_API_TOKEN=your_token"
        )

    api = HfApi(token=token)

    print(f"\n{'=' * 70}")
    print(f"📤 Uploading Gatling Dataset to {repo_id}")
    print(f"{'=' * 70}\n")

    # Create repository if it doesn't exist
    print("🔍 Checking if repository exists...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            exist_ok=True,
            private=False,
        )
        print(f"✓ Repository ready: {repo_id}\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not create/verify repository: {e}\n")

    # Check files exist
    for path in [dataset_path, readme_path, metadata_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    # Get file sizes
    dataset_size = Path(dataset_path).stat().st_size / (1024 * 1024)  # MB
    print(f"📊 Dataset file: {dataset_size:.2f} MB")

    # Upload dataset file (this will take a while for large files)
    print(f"\n1️⃣ Uploading dataset: {dataset_path}")
    print("   (This may take several minutes for large files...)")

    # Extract filename from dataset path for repo naming
    dataset_filename = Path(dataset_path).name

    try:
        upload_file(
            path_or_fileobj=dataset_path,
            path_in_repo=dataset_filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("   ✓ Dataset uploaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to upload dataset: {e}")
        raise

    # Upload README (dataset card)
    print(f"\n2️⃣ Uploading dataset card: {readme_path}")
    try:
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("   ✓ README uploaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to upload README: {e}")
        raise

    # Upload metadata
    print(f"\n3️⃣ Uploading metadata: {metadata_path}")
    try:
        upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo="metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("   ✓ Metadata uploaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to upload metadata: {e}")
        raise

    print(f"\n{'=' * 70}")
    print(f"✅ Upload Complete!")
    print(f"{'=' * 70}")
    print(f"\n📍 Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"\n💡 Load with: load_dataset('{repo_id}')")
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload Gatling dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="OzLabs/gatling-tier1-338k",
        help="HuggingFace repo ID (default: OzLabs/gatling-tier1-338k)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/tier1_free_loaders.jsonl",
        help="Path to dataset JSONL file",
    )
    parser.add_argument(
        "--readme",
        type=str,
        default="data/README.md",
        help="Path to README dataset card",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/tier1_free_loaders.metadata.json",
        help="Path to metadata JSON file",
    )

    args = parser.parse_args()

    try:
        upload_dataset(
            repo_id=args.repo,
            dataset_path=args.dataset,
            readme_path=args.readme,
            metadata_path=args.metadata,
        )
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure the repo exists: https://huggingface.co/new-dataset")
        print("   2. Check your HF_TOKEN has write permissions")
        print("   3. Verify you're a member of OzLabs organization with write access")
        exit(1)


if __name__ == "__main__":
    main()
