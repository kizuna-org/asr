#!/usr/bin/env python3
"""
Script to download the LJSpeech dataset using TensorFlow Datasets
"""

import tensorflow_datasets as tfds
import os
import sys


def download_ljspeech_dataset():
    """Download the LJSpeech dataset to a specified directory."""
    print("Starting LJSpeech dataset download...")

    # Set data directory
    data_dir = "/app/data/ljspeech"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading to: {data_dir}")

    # Check if dataset already exists
    if os.path.exists(os.path.join(data_dir, "LJSpeech-1.1")):
        print("LJSpeech dataset already exists at {}/LJSpeech-1.1".format(data_dir))
        print("Skipping download.")
        return True

    try:
        # Download the dataset
        print("Downloading LJSpeech dataset using TensorFlow Datasets...")
        dataset, info = tfds.load(
            "ljspeech", 
            split="train", 
            with_info=True, 
            data_dir=data_dir, 
            download=True
        )

        print("Dataset downloaded successfully!")
        print(f"Number of examples: {info.splits['train'].num_examples}")
        dataset_size_gb = info.splits["train"].num_bytes / (1024**3)
        print(f"Dataset size: {dataset_size_gb:.2f} GB")
        
        # List the contents
        print("Dataset contents:")
        if os.path.exists(os.path.join(data_dir, "LJSpeech-1.1")):
            import subprocess
            result = subprocess.run(["ls", "-la", os.path.join(data_dir, "LJSpeech-1.1")], 
                                  capture_output=True, text=True)
            print(result.stdout)
        else:
            # List the actual downloaded directory structure
            result = subprocess.run(["find", data_dir, "-type", "d", "-maxdepth", "2"], 
                                  capture_output=True, text=True)
            print("Downloaded directory structure:")
            print(result.stdout)

        return True

    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = download_ljspeech_dataset()
    sys.exit(0 if success else 1)
