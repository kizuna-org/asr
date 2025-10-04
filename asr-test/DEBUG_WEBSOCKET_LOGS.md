# Added Debugging Logs to Backend WebSocket Handler

## Changes Made

Added print statements to the backend WebSocket handler to better debug the
realtime streaming connection flow.

## Modified File

`backend/app/websocket.py`

## Changes

### 1. Log When Text Message Received

**Location:** Line ~136

**Added:**

```python
print(f"[WS] Received text message: type={data.get('type')}")
```

**Purpose:**

- See exactly when the backend receives messages from the frontend
- Identify if the start message is being sent/received

### 2. Log When Starting Audio Streaming

**Location:** Line ~147

**Added:**

```python
print(f"[WS] 🚀 Starting audio streaming: model={model_name}, rate={input_sample_rate}, format={input_dtype}")
```

**Purpose:**

- Confirm the start message was processed
- Show the streaming parameters being used
- More visible than structured JSON logs

## Expected Log Flow

### Successful Connection

```
[WS] Loading model class: ConformerASRModel
[WS] No checkpoint found for model: conformer
[WS] Model conformer loaded and cached successfully
[WS] Received text message: type=start  ← NEW!
[WS] 🚀 Starting audio streaming: model=conformer, rate=48000, format=f32  ← NEW!
```

### If Start Message Not Received

```
[WS] Loading model class: ConformerASRModel
[WS] No checkpoint found for model: conformer
[WS] Model conformer loaded and cached successfully
(no more logs = connection closed before start message sent)
```

## What This Tells Us

### Scenario 1: See Both New Logs

✅ **WebSocket working correctly:**

- Connection established
- Start message sent and received
- Streaming initialized
- **Issue must be elsewhere** (WebRTC, audio capture, etc.)

### Scenario 2: See "Received text message" But Not "Starting audio streaming"

⚠️ **Message received but not processed:**

- Start message arrived
- But condition `if data.get("type") == "start"` not matching
- Possible JSON parsing issue
- Need to check message format

### Scenario 3: Don't See Either New Log

❌ **Start message never sent/received:**

- WebSocket connection closes too quickly
- Frontend not sending start message
- Network/timing issue
- Need to check frontend WebSocket code

## Why We Need This

### Problem

The structured JSON logs from the logger are sometimes hard to read in the
output. The print statements are:

- **Immediate:** Show up right away
- **Simple:** Easy to spot in logs
- **Clear:** Plain text format

### Current Situation

You showed logs like:

```
asr-api-1  | [WS] Loading model class: ConformerASRModel
asr-api-1  | [WS] No checkpoint found for model: conformer
asr-api-1  | [WS] Model conformer loaded and cached successfully
```

But we don't see if the start message was received. The new logs will fill that
gap.

## Testing

### Run the Application

```bash
cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
./run.sh
```

### Click "リアルタイム開始"

Watch for the new log lines in the terminal.

### What to Look For

**Good:** See all logs in sequence

```
[WS] Model conformer loaded and cached successfully
[WS] Received text message: type=start  ← You should see this!
[WS] 🚀 Starting audio streaming: model=conformer, rate=48000, format=f32  ← And this!
```

**Bad:** Missing logs

```
[WS] Model conformer loaded and cached successfully
(nothing else = problem!)
```

## Debugging Next Steps

### If You See "Received text message: type=start"

The WebSocket is working! The issue is likely:

- WebRTC audio capture
- Audio frame transmission
- Queue/threading issues

### If You Don't See It

The WebSocket connection is closing too quickly. Need to:

- Check frontend WebSocket connection code
- Add retry logic
- Increase connection timeout
- Check network connectivity

## Related Files

- `frontend/app.py` - WebSocket client code
- `backend/app/websocket.py` - WebSocket server code (this file)
- `run.sh` - Deployment script that shows logs

## Summary

**Added simple print statements to backend WebSocket handler to debug message
flow and identify where the connection is failing.**

**Key logs to watch for:**

1. `[WS] Received text message: type=start` - Message arrived
2. `[WS] 🚀 Starting audio streaming:...` - Streaming initialized

If you see both, WebSocket is working and we can focus on other issues. If not,
we know the WebSocket connection is the problem.
