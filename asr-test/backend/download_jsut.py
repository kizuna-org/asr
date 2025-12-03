#!/usr/bin/env python3
"""
Script to download the JSUT (Japanese Speech Utterance Corpus) dataset
JSUT is a free Japanese speech corpus for end-to-end speech synthesis
"""

SCRIPT_VERSION = "2025-01-27-01"

import os
import sys
import subprocess
import time
import requests
import zipfile
from pathlib import Path


def _safe_extract_zip(archive_path: str, dest_dir: str) -> None:
    """Safely extract a .zip archive to dest_dir, preventing path traversal."""
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        # Check for path traversal
        for member in zip_ref.namelist():
            member_path = os.path.join(dest_dir, member)
            if not os.path.abspath(member_path).startswith(os.path.abspath(dest_dir)):
                raise zipfile.BadZipFile("Blocked path traversal in zip file")
        zip_ref.extractall(dest_dir)


def _download_with_retries(url: str, dest_file: str, attempts: int = 3, timeout: int = 600) -> None:
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


def download_jsut_dataset():
    """Download the JSUT dataset directly from the source."""
    print("Starting JSUT dataset download...")
    print(f"download_jsut.py version: {SCRIPT_VERSION}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")

    # Set data directory
    data_dir = "/app/data/jsut"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading to: {data_dir}")

    # Check if dataset already exists
    jsut_path = os.path.join(data_dir, "jsut_ver1.1")
    basic5000_path = os.path.join(jsut_path, "basic5000")

    def _dataset_complete() -> bool:
        if not (os.path.isdir(jsut_path) and os.path.isdir(basic5000_path)):
            return False
        # Consider it complete if at least 100 wav files exist in basic5000
        try:
            import glob
            wav_count = len(glob.glob(os.path.join(basic5000_path, "wav", "*.wav")))
            return wav_count >= 100
        except Exception:
            return False

    if _dataset_complete():
        print(f"JSUT dataset is complete at {jsut_path}")
        print("Skipping download.")
        return True
    else:
        if os.path.isdir(jsut_path):
            print(f"Found existing directory but dataset is incomplete at {jsut_path}")
            print("Will (re)download and extract to fix missing files")

    # JSUT download URLs (multiple mirrors)
    env_url = os.environ.get("JSUT_URL")
    mirror_urls = ([env_url] if env_url else []) + [
        # Primary download URL (GitHub releases or direct download)
        "https://github.com/sarulab-speech/jsut-label/archive/refs/heads/master.zip",
        # Alternative: direct download from research site
        # Note: JSUT is typically distributed via GitHub or research sites
        # If direct download URL is not available, users may need to download manually
    ]
    zip_file_path = os.path.join(data_dir, "jsut.zip")

    try:
        # If a zip file already exists, try extraction first
        if os.path.exists(zip_file_path) and os.path.getsize(zip_file_path) > 0:
            print(f"Found existing archive: {zip_file_path}. Trying to extract before re-downloading...")
            try:
                _safe_extract_zip(zip_file_path, data_dir)
                print("✅ Archive extracted successfully (existing file)")
            except zipfile.BadZipFile as e:
                print(f"Warning: existing archive extraction failed: {e}. Will re-download.")
                try:
                    os.remove(zip_file_path)
                except Exception:
                    pass

        # Download the dataset using mirrors if dataset not yet present
        if not _dataset_complete():
            print("Starting download. This may take several minutes...")
            print("Note: JSUT dataset may need to be downloaded manually from:")
            print("  https://sites.google.com/site/shinnosuketakamichi/research-topics/jsut_corpus")
            print("  or")
            print("  https://github.com/sarulab-speech/jsut-label")
            print()
            print("Attempting automatic download from available sources...")
            
            last_err = None
            download_success = False
            for url in mirror_urls:
                print(f"Downloading JSUT dataset from: {url}")
                try:
                    _download_with_retries(url, zip_file_path, attempts=3, timeout=600)
                    print(f"✅ Download completed: {zip_file_path}")
                    download_success = True
                    break
                except Exception as e:
                    last_err = e
                    print(f"Mirror failed: {url} -> {e}")
            
            if not download_success:
                print("\n=== Automatic download failed ===")
                print("JSUT dataset needs to be downloaded manually.")
                print("Please download jsut_ver1.1.zip from one of the following sources:")
                print("  1. https://sites.google.com/site/shinnosuketakamichi/research-topics/jsut_corpus")
                print("  2. https://github.com/sarulab-speech/jsut-label")
                print(f"Then place it as: {zip_file_path}")
                print("Or set JSUT_URL environment variable to a direct download URL.")
                print("\nTried URLs:")
                for u in mirror_urls:
                    print(f" - {u}")
                return False

            # ファイルサイズを確認
            file_size_mb = os.path.getsize(zip_file_path) / (1024 * 1024)
            print(f"Downloaded file size: {file_size_mb:.1f} MB")

            # アーカイブを展開
            print("Extracting archive...")
            _safe_extract_zip(zip_file_path, data_dir)
        
        print("✅ Archive extracted successfully")
        
        # Handle different extraction structures
        # JSUT may extract as jsut-label-master or jsut_ver1.1
        extracted_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        jsut_extracted = None
        for d in extracted_dirs:
            if 'jsut' in d.lower():
                jsut_extracted = os.path.join(data_dir, d)
                break
        
        if jsut_extracted and jsut_extracted != jsut_path:
            # Rename or move to expected structure
            if os.path.exists(jsut_path):
                import shutil
                shutil.rmtree(jsut_path)
            os.rename(jsut_extracted, jsut_path)
            print(f"Renamed extracted directory to: {jsut_path}")
        
        # ダウンロード後の確認
        print("Verifying download...")
        if os.path.exists(jsut_path) and os.path.isdir(jsut_path):
            print(f"✅ Dataset directory found: {jsut_path}")
            
            # ディレクトリの内容を確認
            try:
                result = subprocess.run(["ls", "-la", jsut_path], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print("Dataset contents:")
                    print(result.stdout)
                    
                    # ファイル数をカウント
                    wav_files = list(Path(jsut_path).glob("**/wav/*.wav"))
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
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
                print(f"✅ Cleaned up temporary file: {zip_file_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download dataset: {e}")
        return False
    except zipfile.BadZipFile as e:
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
    success = download_jsut_dataset()
    sys.exit(0 if success else 1)

