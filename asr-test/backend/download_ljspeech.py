#!/usr/bin/env python3
"""
Script to download the LJSpeech dataset directly from the source
"""

SCRIPT_VERSION = "2025-09-24-01"

import os
import sys
import subprocess
import time
import requests
import tarfile
from pathlib import Path


def _safe_extract_tar_bz2(archive_path: str, dest_dir: str) -> None:
    """Safely extract a .tar.bz2 archive to dest_dir, preventing path traversal."""
    with tarfile.open(archive_path, 'r:bz2') as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

        for member in tar.getmembers():
            member_path = os.path.join(dest_dir, member.name)
            if not is_within_directory(dest_dir, member_path):
                raise tarfile.TarError("Blocked path traversal in tar file")
        tar.extractall(dest_dir)


def _download_with_retries(url: str, dest_file: str, attempts: int = 3, timeout: int = 300) -> None:
    """Download URL to dest_file with retries and simple exponential backoff."""
    last_err = None
    for i in range(1, attempts + 1):
        try:
            print(f"Attempt {i}/{attempts}: {url}")
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(dest_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
            print()
            return
        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"Warning: download failed: {e}")
            if i < attempts:
                backoff = min(30, 2 ** i)
                print(f"Retrying in {backoff}s...")
                time.sleep(backoff)
    # If we exit loop without return
    raise last_err if last_err else RuntimeError("Download failed for unknown reason")

def download_ljspeech_dataset():
    """Download the LJSpeech dataset directly from the source."""
    print("Starting LJSpeech dataset download...")
    print(f"download_ljspeech.py version: {SCRIPT_VERSION}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")

    # Set data directory
    data_dir = "/app/data/ljspeech"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading to: {data_dir}")

    # Check if dataset already exists (must include metadata.csv and wavs)
    ljspeech_path = os.path.join(data_dir, "LJSpeech-1.1")
    metadata_path = os.path.join(ljspeech_path, "metadata.csv")
    wavs_dir = os.path.join(ljspeech_path, "wavs")

    def _dataset_complete() -> bool:
        if not (os.path.isdir(ljspeech_path) and os.path.isdir(wavs_dir) and os.path.exists(metadata_path)):
            return False
        # Consider it complete if at least 100 wavs exist
        try:
            import glob
            wav_count = len(glob.glob(os.path.join(wavs_dir, "*.wav")))
            return wav_count >= 100
        except Exception:
            return False

    if _dataset_complete():
        print(f"LJSpeech dataset is complete at {ljspeech_path}")
        print("Skipping download.")
        return True
    else:
        if os.path.isdir(ljspeech_path):
            print(f"Found existing directory but dataset is incomplete at {ljspeech_path}")
            print("Will (re)download and extract to fix missing files (e.g., metadata.csv)")

    # Candidate mirror URLs (first reachable wins). Allow override via env.
    env_url = os.environ.get("LJSPEECH_URL")
    mirror_urls = ([env_url] if env_url else []) + [
        # Legacy keithito paths
        "https://data.keithito.com/speech/LJSpeech-1.1.tar.bz2",
        "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        # OpenSLR mirrors (try http if https fails due to cert or 404)
        "https://www.openslr.org/resources/12/LJSpeech-1.1.tar.bz2",
        "http://www.openslr.org/resources/12/LJSpeech-1.1.tar.bz2",
        "https://openslr.org/resources/12/LJSpeech-1.1.tar.bz2",
        "http://openslr.org/resources/12/LJSpeech-1.1.tar.bz2",
        # Regional mirrors
        "https://us.openslr.org/resources/12/LJSpeech-1.1.tar.bz2",
        "https://openslr.elda.org/resources/12/LJSpeech-1.1.tar.bz2",
    ]
    tar_file_path = os.path.join(data_dir, "LJSpeech-1.1.tar.bz2")

    try:
        # If a tarball already exists, try extraction first
        if os.path.exists(tar_file_path) and os.path.getsize(tar_file_path) > 0:
            print(f"Found existing archive: {tar_file_path}. Trying to extract before re-downloading...")
            try:
                _safe_extract_tar_bz2(tar_file_path, data_dir)
                print("✅ Archive extracted successfully (existing file)")
            except tarfile.TarError as e:
                print(f"Warning: existing archive extraction failed: {e}. Will re-download.")
                try:
                    os.remove(tar_file_path)
                except Exception:
                    pass

        # Download the dataset using mirrors if dataset not yet present
        if not _dataset_complete():
            print("Starting download. This may take several minutes...")
            last_err = None
            for url in mirror_urls:
                print(f"Downloading LJSpeech dataset from: {url}")
                try:
                    _download_with_retries(url, tar_file_path, attempts=3, timeout=300)
                    print(f"✅ Download completed: {tar_file_path}")
                    break
                except Exception as e:
                    last_err = e
                    print(f"Mirror failed: {url} -> {e}")
            else:
                print("\n=== All mirrors failed ===")
                print("Tried URLs in order:")
                for u in mirror_urls:
                    print(f" - {u}")
                print("You can set LJSPEECH_URL to a reachable URL, or place a valid 'LJSpeech-1.1.tar.bz2' under /app/data/ljspeech and rerun.")
                raise requests.exceptions.RequestException(f"All mirrors failed: {last_err}")

            # ファイルサイズを確認
            file_size_mb = os.path.getsize(tar_file_path) / (1024 * 1024)
            print(f"Downloaded file size: {file_size_mb:.1f} MB")

            # アーカイブを展開
            print("Extracting archive...")
            _safe_extract_tar_bz2(tar_file_path, data_dir)
        
        print("✅ Archive extracted successfully")
        
        # ダウンロード後の確認
        print("Verifying download...")
        if os.path.exists(ljspeech_path) and os.path.isdir(ljspeech_path):
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
                    print(f"metadata.csv exists: {os.path.exists(metadata_path)}")
                    
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
            if os.path.exists(tar_file_path):
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
