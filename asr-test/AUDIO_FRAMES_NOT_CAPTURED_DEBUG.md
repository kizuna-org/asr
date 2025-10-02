# Audio Frames Not Being Captured - Debugging Session

## Current Status

**Problem:** Backend WebSocket is ready and waiting for audio frames, but NO
frames are arriving from the frontend.

### What's Working ✅

- Backend WebSocket connection: ✅ Connected
- Model loading: ✅ Working (cached, instant on 2nd attempt)
- Resampler setup: ✅ Ready (48kHz → 16kHz)
- threading.Event fix: ✅ WebSocket stays open (no premature stop)
- Frontend thread initialization: ✅ ALL threads start successfully

### What's NOT Working ❌

- Audio frames NOT reaching backend
- Backend never sees: `[WS] 🎤 First audio frame received`

## Debug Progress

### Phase 1: Threading.Event Fix ✅

**Fixed:** WebSocket was closing immediately after model load because threads
couldn't access `st.session_state`.

**Solution:** Replaced session state checks with `threading.Event` for
cross-thread communication.

**Result:** WebSocket now stays open indefinitely. Backend shows:

```
[WS] ✅ Setup complete, ready to receive audio frames
(... waits forever, no frames arrive)
```

### Phase 2: Thread Initialization Verification ✅

**Added debug messages to confirm threads start:**

User sees in UI:

```
🔧 DEBUG: Starting thread initialization...
🚀 Starting realtime streaming with audio_receiver
🔧 DEBUG: WebSocket thread started
🔧 DEBUG: About to start audio puller thread...
🔧 DEBUG: Audio puller thread started!
✅ リアルタイムストリーミングを開始しました！
```

**Conclusion:** Both threads (WebSocket sender + audio puller) ARE starting
successfully.

### Phase 3: Audio Capture Investigation 🔍 (Current)

**The mystery:**

- WebRTC shows: `rtc_ctx.state.playing: True` ✅
- WebRTC shows: `rtc_ctx.audio_receiver: True` ✅
- Threads are running ✅
- But NO audio frames reaching backend ❌

**Hypothesis:** The audio puller thread is running, but:

1. Either `get_frames()` is returning empty
2. Or there's an error in the frame processing
3. Or the frames aren't being put in the queue
4. Or the WebSocket sender isn't pulling from the queue

**Latest debugging added:**

```python
# In audio puller loop:
loop_count += 1
if loop_count % 100 == 0:
    sys.stderr.write(f"[FRONTEND] Audio puller loop iteration {loop_count}\n")

frames = rtc_ctx.audio_receiver.get_frames()
if frames:
    sys.stderr.write(f"[FRONTEND] 🎤 Got {len(frames)} frames!\n")
elif loop_count % 500 == 0:
    sys.stderr.write(f"[FRONTEND] ⚠️ get_frames() returned empty\n")
```

**What we'll learn:**

- Is the while loop actually running?
- Is `get_frames()` being called?
- Is it returning empty arrays?
- Is it throwing errors?

## Next Steps

1. Deploy with new logging
2. User clicks "リアルタイム開始"
3. **User SPEAKS into microphone** (critical!)
4. Check stderr output for `[FRONTEND]` messages

### Expected Outcomes

**Scenario A: Loop NOT running**

- No "[FRONTEND] Audio puller loop iteration" messages
- → Thread is exiting immediately
- → Check `running_event.is_set()`

**Scenario B: Loop running, but get_frames() empty**

- See loop iteration messages
- See "get_frames() returned empty" warnings
- → WebRTC audio_receiver not getting microphone data
- → Check browser microphone permissions
- → Check WebRTC audio pipeline

**Scenario C: Loop running, get_frames() has data**

- See "🎤 Got X frames!" messages
- → Problem is downstream (frame processing or queue)

**Scenario D: Loop running, but errors**

- Check exception logs
- → Fix the error

## Technical Context

### WebRTC Audio Flow

```
Browser Microphone
    ↓
streamlit-webrtc (WebRTC component)
    ↓
rtc_ctx.audio_receiver.get_frames()
    ↓
Audio Puller Thread (pull_audio_frames)
    ↓
Queue (send_queue)
    ↓
WebSocket Sender Thread (stream_audio_to_ws)
    ↓
Backend WebSocket Endpoint
    ↓
Model Inference
```

**Current status:** Stuck between `rtc_ctx.audio_receiver.get_frames()` and
`Backend WebSocket`

### Key Code Locations

**Frontend:**

- Thread initialization: `frontend/app.py` ~line 1310
- Audio puller thread: `frontend/app.py` ~line 1330
- WebSocket sender: `frontend/app.py` ~line 1056

**Backend:**

- WebSocket handler: `backend/app/websocket.py` ~line 130

## Files Modified

- `frontend/app.py` - Added extensive debugging logs
- `FIX_THREADING_EVENT_FOR_REALTIME_STREAMING.md` - Documents the Event fix

## Commands for Debugging

**Check frontend logs:**

```bash
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose logs -f frontend" | grep "\[FRONTEND\]"
```

**Check backend logs:**

```bash
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose logs -f asr-api" | grep "\[WS\]"
```

## Summary

We've progressively narrowed down the problem:

1. ✅ Fixed: WebSocket premature stop (threading.Event)
2. ✅ Verified: Threads are starting
3. 🔍 **Current:** Investigating why audio frames aren't being captured from
   WebRTC

The next debug deployment will tell us exactly where in the audio capture
pipeline the flow is breaking.
