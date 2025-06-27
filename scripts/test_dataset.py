#!/usr/bin/env python3
"""
Test script to verify LJSpeech dataset loading
"""

import tensorflow_datasets as tfds
import os


def test_dataset_loading():
    """Test loading the pre-downloaded LJSpeech dataset."""
    print("Testing LJSpeech dataset loading...")

    # Check if dataset directory exists
    data_dir = "/opt/datasets"
    if not os.path.exists(data_dir):
        print(f"ERROR: Dataset directory {data_dir} does not exist!")
        return False

    print(f"Dataset directory found: {data_dir}")

    try:
        # Load the dataset
        dataset, info = tfds.load(
            "ljspeech",
            split="train",
            with_info=True,
            as_supervised=True,
            data_dir=data_dir,
        )

        print("Dataset loaded successfully!")
        print(f"Number of examples: {info.splits['train'].num_examples}")
        dataset_size_gb = info.splits["train"].num_bytes / (1024**3)
        print(f"Dataset size: {dataset_size_gb:.2f} GB")

        # Test loading a few examples
        print("\nTesting data loading...")
        for i, (text, audio) in enumerate(dataset.take(2)):
            print(f"Example {i+1}:")
            text_str = text.numpy().decode("utf-8")[:50]
            print(f"  Text: {text_str}...")
            print(f"  Audio shape: {audio.shape}")
            print(f"  Audio dtype: {audio.dtype}")

        print("\nDataset test completed successfully!")
        return True

    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return False


if __name__ == "__main__":
    success = test_dataset_loading()
    exit(0 if success else 1)
