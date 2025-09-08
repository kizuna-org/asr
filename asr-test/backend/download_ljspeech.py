#!/usr/bin/env python3
"""
Script to download the LJSpeech dataset directly from the source
"""

import os
import sys
import subprocess
import time
import requests
import tarfile
from pathlib import Path


def download_ljspeech_dataset():
    """Download the LJSpeech dataset directly from the source."""
    print("Starting LJSpeech dataset download...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")

    # Set data directory
    data_dir = "/app/data/ljspeech"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading to: {data_dir}")

    # Check if dataset already exists
    ljspeech_path = os.path.join(data_dir, "LJSpeech-1.1")
    if os.path.exists(ljspeech_path):
        print(f"LJSpeech dataset already exists at {ljspeech_path}")
        print("Skipping download.")
        return True

    # LJSpeech dataset URL
    dataset_url = "https://data.keithito.com/speech/LJSpeech-1.1.tar.bz2"
    tar_file_path = os.path.join(data_dir, "LJSpeech-1.1.tar.bz2")

    try:
        # Download the dataset
        print(f"Downloading LJSpeech dataset from: {dataset_url}")
        print("This may take several minutes depending on your internet connection...")
        
        # ダウンロードを実行
        response = requests.get(dataset_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(tar_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
        
        print(f"\n✅ Download completed: {tar_file_path}")
        
        # ファイルサイズを確認
        file_size_mb = os.path.getsize(tar_file_path) / (1024 * 1024)
        print(f"Downloaded file size: {file_size_mb:.1f} MB")
        
        # アーカイブを展開
        print("Extracting archive...")
        with tarfile.open(tar_file_path, 'r:bz2') as tar:
            tar.extractall(data_dir)
        
        print("✅ Archive extracted successfully")
        
        # ダウンロード後の確認
        print("Verifying download...")
        if os.path.exists(ljspeech_path):
            print(f"✅ Dataset directory found: {ljspeech_path}")
            
            # ディレクトリの内容を確認
            try:
                result = subprocess.run(["ls", "-la", ljspeech_path], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print("Dataset contents:")
                    print(result.stdout)
                    
                    # ファイル数をカウント
                    wav_files = list(Path(ljspeech_path).glob("wavs/*.wav"))
                    print(f"Number of audio files: {len(wav_files)}")
                    
                else:
                    print(f"Warning: Could not list directory contents: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("Warning: Timeout while listing directory contents")
            except Exception as e:
                print(f"Warning: Error listing directory contents: {e}")
        else:
            print("❌ Dataset directory not found after extraction")
            return False
        
        # 一時ファイルを削除（オプション）
        try:
            os.remove(tar_file_path)
            print(f"✅ Cleaned up temporary file: {tar_file_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download dataset: {e}")
        return False
    except tarfile.TarError as e:
        print(f"ERROR: Failed to extract archive: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        # エラー時の診断情報を出力
        print("\n=== Diagnostic Information ===")
        print(f"Data directory exists: {os.path.exists(data_dir)}")
        print(f"Data directory is writable: {os.access(data_dir, os.W_OK)}")
        print(f"Available disk space:")
        try:
            result = subprocess.run(["df", "-h", data_dir], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(result.stdout)
        except:
            print("Could not check disk space")
        
        return False


if __name__ == "__main__":
    success = download_ljspeech_dataset()
    sys.exit(0 if success else 1)
