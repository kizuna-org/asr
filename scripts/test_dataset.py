#!/usr/bin/env python3
"""
Test script to verify M-AILABS dataset loading
"""

import os
import librosa
from pathlib import Path


def parse_mailabs_metadata(metadata_path: str):
    """Parse M-AILABS metadata file."""
    metadata = []
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 2:
                        filename = parts[0]
                        text = parts[1]
                        # Extract speaker info from path
                        speaker_id = filename.split('/')[0] if '/' in filename else "unknown"
                        
                        metadata.append({
                            'filename': filename,
                            'text': text,
                            'speaker_id': speaker_id
                        })
    except Exception as e:
        print(f"Metadata parsing error: {e}")
    
    return metadata


def test_dataset_loading():
    """Test loading the pre-downloaded M-AILABS dataset."""
    print("Testing M-AILABS dataset loading...")

    # Check if dataset directory exists
    dataset_dir = "/opt/datasets/mailabs"
    audio_dir = os.path.join(dataset_dir, "en_US")
    
    if not os.path.exists(audio_dir):
        print(f"ERROR: Dataset directory {audio_dir} does not exist!")
        return False

    print(f"Dataset directory found: {audio_dir}")

    try:
        # Find metadata files
        metadata_files = []
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('metadata.csv') or file.endswith('metadata.txt'):
                    metadata_files.append(os.path.join(root, file))
        
        if not metadata_files:
            print("ERROR: No metadata files found!")
            return False
        
        print(f"Found {len(metadata_files)} metadata files")
        
        # Parse all metadata
        all_metadata = []
        speakers = set()
        
        for metadata_file in metadata_files:
            print(f"📖 Reading metadata: {metadata_file}")
            metadata = parse_mailabs_metadata(metadata_file)
            all_metadata.extend(metadata)
            
            for item in metadata:
                speakers.add(item['speaker_id'])
        
        print(f"📊 Total samples: {len(all_metadata):,}")
        print(f"🎭 Number of speakers: {len(speakers)}")
        print(f"🎭 Speakers: {sorted(list(speakers))}")

        # Test loading a few examples
        print("\nTesting audio loading...")
        samples_to_test = min(3, len(all_metadata))
        
        for i in range(samples_to_test):
            item = all_metadata[i]
            audio_path = os.path.join(audio_dir, item['filename'])
            if audio_path.endswith('.txt'):
                audio_path = audio_path.replace('.txt', '.wav')
            
            print(f"\nExample {i+1}:")
            print(f"  Speaker ID: {item['speaker_id']}")
            print(f"  Text: {item['text'][:50]}...")
            print(f"  Audio file: {audio_path}")
            
            if os.path.exists(audio_path):
                try:
                    audio, sr = librosa.load(audio_path, sr=22050)
                    print(f"  Audio shape: {audio.shape}")
                    print(f"  Sample rate: {sr}")
                    print(f"  Duration: {len(audio) / sr:.2f} seconds")
                    print("  ✅ Audio loaded successfully")
                except Exception as e:
                    print(f"  ❌ Audio loading error: {e}")
            else:
                print(f"  ❌ Audio file not found")

        print("\n✅ M-AILABS dataset test completed successfully!")
        return True

    except Exception as e:
        print(f"ERROR: Failed to test dataset: {e}")
        return False


if __name__ == "__main__":
    success = test_dataset_loading()
    exit(0 if success else 1)
