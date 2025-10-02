# Realtime Inference Fix

## Issues Identified

1. **No logs appearing** - The logging was too verbose in some places and
   missing in others
2. **CTC decoding threshold too strict** - The default threshold of -10.0 was
   preventing character detection
3. **Test scripts not available in container** - Test files were not copied to
   the Docker image
4. **Silent failures** - Errors were not being properly logged or surfaced

## Changes Made

### 1. Fixed CTC Decoder (`backend/app/models/realtime.py`)

**Changes:**

- Changed default threshold from `-10.0` to `-5.0` (less strict)
- Added statistical logging (avg prob, max prob, detection summary)
- Removed excessive per-step logging
- Added fallback logic: if no characters detected with default threshold, retry
  with `-10.0`

**Why:** The model was generating probabilities but the threshold was too
strict, causing all characters to be filtered out.

### 2. Improved Inference Logging (`backend/app/models/realtime.py`)

**Changes:**

- Changed most logs from INFO to DEBUG level
- Added summary-level INFO logs for key events
- Added character count and truncated output for long results
- Added explicit logging when no text is recognized

**Why:** Too many logs made it hard to see what was happening. Now we have
clean, informative logging.

### 3. Updated Dockerfile (`backend/Dockerfile`)

**Changes:**

- Added test scripts to Docker image:
  - `test_realtime_model.py`
  - `demo_realtime.py`
  - `test_training_fix.py`
  - `migrate_realtime_checkpoints.py`

**Why:** Test scripts were not available inside the container, making debugging
impossible.

### 4. Created Test Script (`test_realtime_inference.sh`)

**Purpose:** Easy-to-run script that tests realtime inference on the GPU server
via SSH

**Usage:**

```bash
cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
./test_realtime_inference.sh
```

**What it does:**

1. Runs unit tests for the realtime model
2. Runs the realtime demo
3. Shows recent logs from the container

### 5. Adjusted Logging Levels (`backend/app/main.py`)

**Changes:**

- Changed `model` logger from DEBUG to INFO
- Changed `asr-api` logger from DEBUG to INFO

**Why:** Reduced log noise while keeping important information visible.

## How to Deploy and Test

1. **Deploy the changes to GPU server:**
   ```bash
   cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
   ./run.sh
   ```

2. **Run the tests:**
   ```bash
   ./test_realtime_inference.sh
   ```

3. **Check container logs:**
   ```bash
   ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f asr-api"
   ```

4. **Test via WebSocket:**
   - Access the frontend at `http://localhost:58080`
   - Use the "リアルタイム推論" feature
   - Upload an audio file or use microphone input
   - Check if transcription appears

## Expected Behavior After Fix

### Before:

- No logs appearing during inference
- Empty transcription results
- Silent failures with no indication of what went wrong

### After:

- Clear, informative logs showing:
  - Feature extraction shape
  - Encoder output shape
  - CTC probability statistics
  - Character detection summary
  - Final transcription result
- Characters are detected even from untrained models (may be gibberish but
  proves the pipeline works)
- Fallback mechanism tries different thresholds if first attempt fails

## Troubleshooting

### If still no output:

1. **Check model initialization:**
   ```bash
   ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose exec -T asr-api python -c 'from app.models.realtime import RealtimeASRModel; from app import config_loader; config = config_loader.load_config(); model = RealtimeASRModel(config[\"models\"][\"realtime\"]); print(\"Model initialized successfully\")'"
   ```

2. **Check audio processing:**
   - Verify audio sample rate matches config (16kHz)
   - Verify audio is mono (not stereo)
   - Verify audio format is supported

3. **Check GPU availability:**
   ```bash
   ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose exec -T asr-api python gpu_check.py"
   ```

4. **Check logs for errors:**
   ```bash
   ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose logs --tail=100 asr-api | grep -i error"
   ```

## Technical Details

### CTC Decoding Threshold

The threshold is a log probability value (negative):

- `-5.0`: More permissive (detects more characters, may include noise)
- `-10.0`: Stricter (only high-confidence characters)
- Lower values (more negative) = stricter

The model now:

1. First tries with `-5.0` (reasonable threshold)
2. If no characters detected, retries with `-10.0` (strict threshold)
3. Returns whatever it finds, or empty string if nothing

### Why Untrained Models Still Produce Output

Even an untrained model will:

1. Extract mel-spectrogram features (deterministic)
2. Pass through encoder (random weights → some output)
3. Pass through CTC decoder (probabilities sum to 1)
4. Greedy decode to characters (picks max probability)

The output will be gibberish, but it proves the pipeline works. After training,
it should produce meaningful text.

## Next Steps

1. **Train the model** with actual data to get meaningful transcriptions
2. **Fine-tune threshold** based on validation set performance
3. **Add confidence scores** to help users understand output quality
4. **Implement beam search** for better accuracy (currently using greedy
   decoding)
5. **Add language model** for better word-level accuracy
