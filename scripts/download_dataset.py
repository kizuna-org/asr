#!/usr/bin/env python3
"""
Script to pre-download the LJSpeech dataset for Docker container
"""

import tensorflow_datasets as tfds
import os


def download_ljspeech_dataset():
    """Download the LJSpeech dataset to a specified directory."""
    print("Pre-downloading LJSpeech dataset...")

    # Set data directory
    data_dir = "/opt/datasets"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading to: {data_dir}")

    try:
        # Download the dataset
        dataset, info = tfds.load(
            "ljspeech", split="train", with_info=True, data_dir=data_dir, download=True
        )

        print("Dataset downloaded successfully!")
        print(f"Number of examples: {info.splits['train'].num_examples}")
        dataset_size_gb = info.splits["train"].num_bytes / (1024**3)
        print(f"Dataset size: {dataset_size_gb:.2f} GB")

        return True

    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        return False


if __name__ == "__main__":
    success = download_ljspeech_dataset()
    exit(0 if success else 1)
