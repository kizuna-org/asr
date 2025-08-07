#!/usr/bin/env bash

set -euo pipefail

# OpenSLR Download Script
# Downloads all files from openslr-urls.txt to /Volumes/SSD-PHPU3A/openslr/<resource_number>

# Base directory
BASE_DIR="/Volumes/SSD-PHPU3A/datasets/openslr"

# Rate limiting settings
DELAY_BETWEEN_DOWNLOADS=2  # seconds between downloads
MAX_RETRIES=3              # maximum retry attempts
RETRY_DELAY=5              # seconds to wait before retry

# Parallel download settings
PARALLEL_DOWNLOADS=4       # number of concurrent downloads
PARALLEL_MODE=true         # set to false to disable parallel downloads

# External tool mode
USE_ARIA2C=true            # set to true to use aria2c for downloads

# Progress tracking file
PROGRESS_FILE="download_progress.txt"

# Create base directory if it doesn't exist
mkdir -p "$BASE_DIR"

# Check if openslr-urls.txt exists
if [ ! -f "openslr-urls.txt" ]; then
    echo "Error: openslr-urls.txt not found in current directory"
    exit 1
fi

# Function to extract resource number from URL
extract_resource_number() {
    local url="$1"
    # Extract number between /resources/ and / or end of URL
    echo "$url" | sed -n 's|.*/resources/\([0-9]*\)/.*|\1|p'
}

# Function to get filename from URL
get_filename() {
    local url="$1"
    echo "$url" | sed 's|.*/||'
}

# Function to check if file already exists and is valid
file_exists_and_valid() {
    local file_path="$1"
    local url="$2"

    if [ -f "$file_path" ]; then
        # Check if file has content (not empty)
        if [ -s "$file_path" ]; then
            # Get expected file size from server
            local expected_size=$(curl -sI "$url" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')
            local actual_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo "0")

            # If we can get the expected size, validate it
            if [ -n "$expected_size" ] && [ "$expected_size" -gt 0 ]; then
                if [ "$actual_size" -eq "$expected_size" ]; then
                    return 0  # File exists and has correct size
                else
                    echo "  ⚠ File size mismatch: expected ${expected_size} bytes, got ${actual_size} bytes"
                    return 1  # File size doesn't match
                fi
            else
                # If we can't get expected size, check if file is reasonably sized (>1KB)
                if [ "$actual_size" -gt 1024 ]; then
                    return 0  # File exists and has reasonable size
                else
                    echo "  ⚠ File too small (${actual_size} bytes), likely incomplete"
                    return 1  # File too small
                fi
            fi
        else
            echo "  ⚠ File exists but is empty"
            return 1  # File is empty
        fi
    fi
    return 1  # File doesn't exist
}

# --- ARIA2C MODE ---
if [ "$USE_ARIA2C" = true ]; then
    if ! command -v aria2c >/dev/null 2>&1; then
        echo "aria2c is not installed. Please install it (e.g., brew install aria2) and rerun the script."
        exit 1
    fi

    echo "Using aria2c for parallel downloads..."
    echo "Preparing input file for aria2c..."
    ARIA2_INPUT="aria2c_input.txt"
    rm -f "$ARIA2_INPUT"

    # Count total files for progress display
    total_files=$(wc -l < openslr-urls.txt)
    current_file=0

    while IFS= read -r url; do
        [ -z "$url" ] && continue
        current_file=$((current_file + 1))

        resource_num=$(echo "$url" | sed -n 's|.*/resources/\([0-9]*\)/.*|\1|p')
        filename=$(echo "$url" | sed 's|.*/||')
        dir="$BASE_DIR/$resource_num"
        file_path="$dir/$filename"

        mkdir -p "$dir"

        # Check if file already exists and is valid
        if file_exists_and_valid "$file_path" "$url"; then
            file_size=$(du -h "$file_path" 2>/dev/null | cut -f1 || echo "unknown")
            echo "[$current_file/$total_files] Skipping: $filename (Resource $resource_num) - Already exists ($file_size)"
        else
            # Add to aria2c input file for download
            echo "$url" >> "$ARIA2_INPUT"
            echo "  dir=$dir" >> "$ARIA2_INPUT"
            echo "  out=$filename" >> "$ARIA2_INPUT"
            echo "[$current_file/$total_files] Queuing: $filename (Resource $resource_num)"
        fi
    done < openslr-urls.txt

    echo "Starting aria2c..."
    aria2c -i "$ARIA2_INPUT" -j "$PARALLEL_DOWNLOADS" --continue=true --max-connection-per-server=4 --summary-interval=5
    echo "Download completed!"
    echo "Files downloaded to: $BASE_DIR"
    echo ""
    echo "Directory structure:"
    echo "$BASE_DIR/"
    for dir in "$BASE_DIR"/*/; do
        if [ -d "$dir" ]; then
            resource_num=$(basename "$dir")
            file_count=$(ls -1 "$dir" 2>/dev/null | wc -l)
            echo "  $resource_num/ ($file_count files)"
        fi
    done
    exit 0
fi

# Function to download with retry logic
download_with_retry() {
    local url="$1"
    local file_path="$2"
    local filename="$3"
    local retry_count=0

    while [ $retry_count -lt $MAX_RETRIES ]; do
        if curl -L -o "$file_path" "$url" --progress-bar 2>&1; then
            return 0  # Success
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                echo "  ⚠ Retry $retry_count/$MAX_RETRIES in ${RETRY_DELAY}s..."
                sleep $RETRY_DELAY
            fi
        fi
    done

    return 1  # Failed after all retries
}

# Function to download a single file (for parallel processing)
download_single_file() {
    local url="$1"
    local current_file="$2"
    local total_files="$3"

    # Extract resource number and filename
    local resource_num=$(extract_resource_number "$url")
    local filename=$(get_filename "$url")

    # Create directory for this resource
    local resource_dir="$BASE_DIR/$resource_num"
    mkdir -p "$resource_dir"

    # Full path for the file
    local file_path="$resource_dir/$filename"

    # Check if file already exists and is valid
    if file_exists_and_valid "$file_path" "$url"; then
        local file_size=$(du -h "$file_path" 2>/dev/null | cut -f1 || echo "unknown")
        echo "[$current_file/$total_files] Skipping: $filename (Resource $resource_num) - Already exists ($file_size)"
        return 0
    fi

    echo "[$current_file/$total_files] Downloading: $filename (Resource $resource_num)"

    # Download with retry logic
    if download_with_retry "$url" "$file_path" "$filename"; then
        # Get file size for display
        if [ -f "$file_path" ]; then
            local file_size=$(du -h "$file_path" | cut -f1)
            echo "[$current_file/$total_files] ✓ Successfully downloaded: $filename ($file_size)"
        else
            echo "[$current_file/$total_files] ✓ Successfully downloaded: $filename"
        fi
        return 0
    else
        echo "[$current_file/$total_files] ✗ Failed to download: $filename after $MAX_RETRIES retries"
        # Remove the file if download failed
        [ -f "$file_path" ] && rm "$file_path"
        return 1
    fi
}

# Function to save progress
save_progress() {
    local current_file="$1"
    echo "$current_file" > "$PROGRESS_FILE"
}

# Function to load progress
load_progress() {
    if [ -f "$PROGRESS_FILE" ]; then
        cat "$PROGRESS_FILE"
    else
        echo "0"
    fi
}

# Function to wait for background jobs and manage concurrency
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $PARALLEL_DOWNLOADS ]; do
        sleep 1
    done
}

# Function to wait for all background jobs to complete
wait_for_all_jobs() {
    wait
}

# Handle script interruption
cleanup() {
    echo ""
    echo "Script interrupted. Progress saved. You can resume by running the script again."
    # Kill all background jobs
    jobs -p | xargs -r kill
    exit 0
}

# Set up signal handlers
trap cleanup INT TERM

# Load previous progress
last_downloaded=$(load_progress)
if [ "$last_downloaded" -gt 0 ]; then
    echo "Resuming from file $last_downloaded..."
    echo ""
fi

# Counter for progress tracking
total_files=$(wc -l < openslr-urls.txt)
current_file=$last_downloaded

echo "Starting download of $total_files files from OpenSLR..."
echo "Base directory: $BASE_DIR"
if [ "$PARALLEL_MODE" = true ]; then
    echo "Parallel mode: Enabled with $PARALLEL_DOWNLOADS concurrent downloads"
else
    echo "Parallel mode: Disabled (sequential downloads)"
    echo "Rate limiting: ${DELAY_BETWEEN_DOWNLOADS}s delay between downloads"
fi
echo "Retry settings: ${MAX_RETRIES} retries, ${RETRY_DELAY}s delay"
echo "Resume enabled: Will skip already downloaded files"
echo ""

# Read each URL from the file
while IFS= read -r url; do
    # Skip empty lines
    [ -z "$url" ] && continue

    current_file=$((current_file + 1))

    if [ "$PARALLEL_MODE" = true ]; then
        # Parallel mode: start download in background
        wait_for_slot
        download_single_file "$url" "$current_file" "$total_files" &

        # Save progress periodically (every 10 downloads)
        if [ $((current_file % 10)) -eq 0 ]; then
            save_progress "$current_file"
        fi
    else
        # Sequential mode: download one at a time
        download_single_file "$url" "$current_file" "$total_files"

        # Save progress after each download
        save_progress "$current_file"

        # Rate limiting: wait between downloads
        if [ $current_file -lt $total_files ]; then
            echo "  ⏳ Waiting ${DELAY_BETWEEN_DOWNLOADS}s before next download..."
            sleep $DELAY_BETWEEN_DOWNLOADS
        fi
    fi

    echo ""

done < openslr-urls.txt

# Wait for all background jobs to complete if in parallel mode
if [ "$PARALLEL_MODE" = true ]; then
    echo "Waiting for all downloads to complete..."
    wait_for_all_jobs
    echo "All parallel downloads completed!"
fi

# Save final progress
save_progress "$current_file"

# Clean up progress file when complete
rm -f "$PROGRESS_FILE"

echo "Download completed!"
echo "Files downloaded to: $BASE_DIR"
echo ""
echo "Directory structure:"
echo "$BASE_DIR/"
for dir in "$BASE_DIR"/*/; do
    if [ -d "$dir" ]; then
        resource_num=$(basename "$dir")
        file_count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "  $resource_num/ ($file_count files)"
    fi
done
