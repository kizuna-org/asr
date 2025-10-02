# Enhanced WebSocket Debugging Logs

## Overview

Added comprehensive debugging logs to track the complete WebSocket audio
streaming lifecycle, from connection to inference results.

## Changes Made

### File: `backend/app/websocket.py`

All changes add print statements for immediate visibility alongside existing
structured logging.

### 1. Model Setup Logging

**Added after model loading (~line 156):**

```python
print(f"[WS] Model loaded, setting up resampler...")
```

**Added for resampler initialization:**

```python
print(f"[WS] Resampler initialized: {input_sample_rate} → 16000 Hz")
# or
print(f"[WS] No resampler needed (already 16000 Hz)")
```

**Added before sending ready status:**

```python
print(f"[WS] Sending ready status to client...")
await websocket.send_text(json.dumps({"type": "status", "payload": {"status": "ready"}}))
print(f"[WS] ✅ Setup complete, ready to receive audio frames")
```

### 2. Audio Frame Reception Logging

**Added when first frame received (~line 210):**

```python
if not hasattr(websocket_endpoint, '_first_frame_logged'):
    print(f"[WS] 🎤 First audio frame received: {len(raw)} bytes")
    websocket_endpoint._first_frame_logged = True
```

**Added when frame ignored (before start):**

```python
print(f"[WS] ⚠️ Audio frame ignored - streaming not started")
```

### 3. Inference Logging

**Added before inference:**

```python
print(f"[WS] 🤖 Running inference on {waveform.numel()} samples ({waveform.numel()/16000:.2f}s)...")
```

**Added after inference:**

```python
print(f"[WS] ✅ Inference completed in {inference_time:.3f}s")
```

**Added for results:**

```python
# Non-empty result
print(f"[WS] 📝 Partial result: {transcription[:100]}...")

# Empty result
print(f"[WS] ⚠️ Empty transcription result")
```

**Added for errors:**

```python
print(f"[WS] ❌ Inference error: {e}")
```

### 4. Disconnection Logging

**Added for normal disconnect:**

```python
print(f"[WS] 🔌 Client disconnected")
```

**Added for errors:**

```python
print(f"[WS] ❌ WebSocket error: {e}")
```

**Added for cleanup:**

```python
print(f"[WS] 🧹 Cleaning up WebSocket connection")
```

## Complete Log Flow

### Successful Streaming Session

```
[WS] Received text message: type=start
[WS] 🚀 Starting audio streaming: model=conformer, rate=48000, format=f32
[WS] Loading model class: ConformerASRModel
[WS] No checkpoint found for model: conformer
[WS] Model conformer loaded and cached successfully
[WS] Model loaded, setting up resampler...
[WS] Resampler initialized: 48000 → 16000 Hz
[WS] Sending ready status to client...
[WS] ✅ Setup complete, ready to receive audio frames

[User starts speaking]
[WS] 🎤 First audio frame received: 4096 bytes

[After ~1 second of audio]
[WS] 🤖 Running inference on 16000 samples (1.00s)...
[WS] ✅ Inference completed in 0.123s
[WS] 📝 Partial result: hello world...

[More audio, more inference cycles...]

[User clicks stop]
[WS] 🔌 Client disconnected
[WS] 🧹 Cleaning up WebSocket connection
```

### Connection Issues

```
[WS] Received text message: type=start
[WS] 🚀 Starting audio streaming: model=conformer, rate=48000, format=f32
[WS] Loading model class: ConformerASRModel
[WS] Model conformer loaded and cached successfully
[WS] Model loaded, setting up resampler...
❌ (No further logs = connection lost during setup)
```

### Audio Not Flowing

```
[WS] ✅ Setup complete, ready to receive audio frames
(No "First audio frame received" = frontend not sending audio)
```

### Inference Problems

```
[WS] 🎤 First audio frame received: 4096 bytes
[WS] 🤖 Running inference on 16000 samples (1.00s)...
[WS] ❌ Inference error: CUDA out of memory
```

## Troubleshooting Guide

### Scenario 1: Logs Stop After Model Loading

**Symptoms:**

```
[WS] Model conformer loaded and cached successfully
(nothing else)
```

**Problem:** Connection lost during model loading (model loading takes too long,
client timeout)

**Solutions:**

- Check model loading time
- Increase client timeout
- Add connection keep-alive

### Scenario 2: Setup Complete But No Audio Frames

**Symptoms:**

```
[WS] ✅ Setup complete, ready to receive audio frames
(no "First audio frame received")
```

**Problem:** Frontend not sending audio frames

**Check:**

- WebRTC audio capture working?
- Audio puller thread running?
- Check frontend logs for errors
- Verify microphone permissions

### Scenario 3: Audio Frames But No Inference

**Symptoms:**

```
[WS] 🎤 First audio frame received: 4096 bytes
(no inference logs after 1+ second)
```

**Problem:** Buffer not accumulating or inference timing issue

**Check:**

- Is buffer being populated?
- Is 1-second timer firing?
- Check async event loop

### Scenario 4: Inference But Empty Results

**Symptoms:**

```
[WS] ✅ Inference completed in 0.123s
[WS] ⚠️ Empty transcription result
```

**Problem:** Model producing empty output (expected for untrained model)

**Expected:** This is normal if model is not trained yet

### Scenario 5: Inference Errors

**Symptoms:**

```
[WS] ❌ Inference error: CUDA out of memory
```

**Problems & Solutions:**

- **CUDA OOM:** Reduce buffer size, use smaller model
- **Shape mismatch:** Check audio format/resampling
- **Model error:** Check model compatibility

## Log Symbols Legend

| Symbol | Meaning                 |
| ------ | ----------------------- |
| 🚀     | Starting/Initialization |
| ✅     | Success/Completion      |
| 🎤     | Audio data received     |
| 🤖     | AI inference            |
| 📝     | Results/Output          |
| ⚠️     | Warning                 |
| ❌     | Error                   |
| 🔌     | Connection event        |
| 🧹     | Cleanup                 |

## Benefits

### Immediate Visibility

**Before:**

- Only structured JSON logs (hard to read quickly)
- Missing key lifecycle events
- Difficult to spot issues

**After:**

- Simple print statements show up immediately
- Clear symbols for quick identification
- Complete lifecycle visibility

### Easier Debugging

**Can now answer:**

1. ✅ Is WebSocket connection successful?
2. ✅ Is model loading complete?
3. ✅ Is resampler configured correctly?
4. ✅ Are audio frames being received?
5. ✅ Is inference running?
6. ✅ Are results being produced?
7. ✅ Where exactly does it fail?

### Performance Insights

**Timing information:**

- Model loading time (implied from log timing)
- Inference time (explicit in logs)
- Audio buffering rate (frame count)

## Testing

### Deploy and Test

```bash
cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
./run.sh
```

### Test Scenarios

#### Test 1: Normal Streaming

1. Click "START" → Enable microphone
2. Click "リアルタイム開始"
3. **Look for:** Setup complete message
4. Speak into microphone
5. **Look for:** Audio frame and inference logs

**Expected:**

```
[WS] ✅ Setup complete, ready to receive audio frames
[WS] 🎤 First audio frame received: ...
[WS] 🤖 Running inference on ...
[WS] 📝 Partial result: ...
```

#### Test 2: Connection Timing

1. Start streaming
2. **Measure:** Time from "Starting audio streaming" to "Setup complete"
3. **Should be:** < 15 seconds (model loading time)

**If longer:**

- Model loading too slow
- Consider model caching or preloading

#### Test 3: Audio Flow

1. Start streaming
2. Speak continuously
3. **Count:** "Running inference" logs
4. **Should see:** One every ~1 second

**If not:**

- Audio not flowing
- Check frontend audio puller

#### Test 4: Stop Behavior

1. Start streaming
2. Click "停止"
3. **Look for:** "Client disconnected" and "Cleaning up"

**Expected:**

```
[WS] 🔌 Client disconnected
[WS] 🧹 Cleaning up WebSocket connection
```

## Next Steps

### If Setup Completes But No Audio

**Check frontend:**

- Is audio puller thread running?
- Is WebRTC audio_receiver active?
- Check browser console for WebRTC errors

### If Audio Frames Received But No Inference

**Check backend:**

- Is async event loop running?
- Is buffer being populated?
- Add buffer size logging

### If Inference Produces Empty Results

**Expected behavior:**

- Untrained model produces empty/gibberish output
- This means the pipeline is working!
- Train the model for real transcriptions

## Summary

**Added comprehensive debug logging to track WebSocket audio streaming from
connection to inference results.**

**Key additions:**

- 🚀 Setup progress tracking
- 🎤 Audio reception confirmation
- 🤖 Inference execution logging
- 📝 Results visibility
- ❌ Error identification
- 🔌 Connection lifecycle tracking

**Result:** Complete visibility into audio streaming pipeline for easier
debugging and verification.
