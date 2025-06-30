#!/usr/bin/env python3
"""
Script to pre-download the M-AILABS dataset for Docker container or local use
"""

import os
import requests
import tarfile
import argparse
import sys


def download_mailabs_dataset(output_dir="/opt/datasets/mailabs"):
    """Download and extract the M-AILABS dataset to a specified directory."""
    print("Pre-downloading M-AILABS dataset...")

    # Set data directory
    dataset_dir = output_dir
    os.makedirs(dataset_dir, exist_ok=True)
    
    audio_dir = os.path.join(dataset_dir, "en_US")
    dataset_path = os.path.join(dataset_dir, "en_US.tgz")
    
    # Check if already downloaded
    if os.path.exists(audio_dir) and os.listdir(audio_dir):
        print("✅ M-AILABS dataset already exists")
        return True

    print(f"Downloading to: {dataset_dir}")

    try:
        # Download the dataset
        mailabs_url = "https://ics.tau-ceti.space/data/Training/stt_tts/en_US.tgz"
        print(f"📥 Downloading from: {mailabs_url}")
        
        # Check if partial file exists for resume
        resume_header = {}
        initial_pos = 0
        mode = 'wb'
        
        if os.path.exists(dataset_path):
            initial_pos = os.path.getsize(dataset_path)
            if initial_pos > 0:
                resume_header = {'Range': f'bytes={initial_pos}-'}
                mode = 'ab'  # Append mode for resume
                print(f"📂 Resuming download from byte {initial_pos:,}")
        
        response = requests.get(mailabs_url, headers=resume_header, stream=True)
        
        # Handle range request responses
        if response.status_code == 206:  # Partial Content
            print("✅ Resume successful")
        elif response.status_code == 416:  # Range Not Satisfiable (file already complete)
            print("✅ File already downloaded completely")
            # Verify the file is complete by checking if it can be extracted
            try:
                with tarfile.open(dataset_path, 'r:gz') as tar:
                    pass  # Just check if it's a valid tarfile
                print("✅ Existing file is valid, skipping download")
                # Skip to extraction step
                total_size = initial_pos
                downloaded_size = total_size
            except:
                print("⚠️ Existing file is corrupted, restarting download")
                os.remove(dataset_path)
                initial_pos = 0
                mode = 'wb'
                response = requests.get(mailabs_url, stream=True)
        elif initial_pos > 0 and response.status_code == 200:
            print("⚠️ Server doesn't support resume, restarting download")
            os.remove(dataset_path)
            initial_pos = 0
            mode = 'wb'
        
        response.raise_for_status()
        
        # Get total file size
        if 'content-range' in response.headers:
            # Parse "bytes start-end/total" format
            total_size = int(response.headers['content-range'].split('/')[-1])
        else:
            content_length = response.headers.get('content-length', 0)
            total_size = int(content_length) + initial_pos
        
        downloaded_size = initial_pos
        
        # Only download if not already complete
        if downloaded_size < total_size:
            with open(dataset_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rProgress: {progress:.1f}% ({downloaded_size:,}/{total_size:,} bytes)", end="", flush=True)
        
        print(f"\n📦 Download completed: {dataset_path}")
        
        # Extract the dataset
        print("📂 Extracting dataset...")
        with tarfile.open(dataset_path, 'r:gz') as tar:
            tar.extractall(dataset_dir)
        
        print("✅ Dataset extraction completed")
        
        # Clean up the tar file
        os.remove(dataset_path)
        print("🗑️ Cleaned up temporary files")

        # Count files
        total_files = 0
        speakers = set()
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    total_files += 1
                    # Extract speaker from path
                    relative_path = os.path.relpath(root, audio_dir)
                    speaker = relative_path.split(os.sep)[0] if relative_path != '.' else 'unknown'
                    speakers.add(speaker)
        
        print(f"📊 Total audio files: {total_files:,}")
        print(f"🎭 Number of speakers: {len(speakers)}")
        print(f"🎭 Speakers: {sorted(list(speakers))}")

        return True

    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download the M-AILABS dataset for speech training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default Docker location
  python download_dataset.py

  # Download to local datasets directory
  python download_dataset.py --local

  # Download to custom directory
  python download_dataset.py --output-dir /path/to/your/datasets

  # Download to current directory
  python download_dataset.py --output-dir ./my_datasets
        """
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Download to local datasets directory (./datasets/mailabs) instead of Docker default"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Specify custom output directory for the dataset"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        # User specified custom directory
        output_dir = os.path.abspath(args.output_dir)
        if not output_dir.endswith("mailabs"):
            output_dir = os.path.join(output_dir, "mailabs")
    elif args.local:
        # Local mode: use ./datasets/mailabs
        output_dir = os.path.abspath("./datasets/mailabs")
    else:
        # Default Docker mode
        output_dir = "/opt/datasets/mailabs"
    
    print(f"🎯 Target directory: {output_dir}")
    
    # Create parent directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    except Exception as e:
        print(f"ERROR: Cannot create directory {os.path.dirname(output_dir)}: {e}")
        sys.exit(1)
    
    success = download_mailabs_dataset(output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
