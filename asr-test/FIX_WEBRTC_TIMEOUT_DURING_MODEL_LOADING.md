# Fix: WebRTC Timeout During Model Loading

## Problem

**WebRTC connection disconnects before audio streaming can start**, causing
realtime inference to fail.

### Root Cause

Timeline of the issue:

```
T=0s:    User clicks START → WebRTC establishes
T=1s:    User clicks "リアルタイム開始" → Backend starts model loading
T=1-14s: Model loading in progress (~13 seconds)
         ↓
         WebRTC stays idle (no keep-alive)
         ↓
T=~10s:  ❌ WebRTC times out and disconnects
         ↓
T=14s:   Backend finishes model loading
         Backend sends "ready" status
         ❌ But WebRTC is already disconnected!
         ❌ Audio frames cannot be sent
```

**The problem:** Model loading takes ~13-15 seconds, but WebRTC disconnects
after ~10 seconds of inactivity.

## Solutions Implemented

### 1. User Warnings and Progress Indication

**Added clear warnings before starting:**

```python
st.success("✅ マイクが有効です。'リアルタイム開始'ボタンをクリックしてください。")
st.info("💡 **重要:** モデルの読み込みに約15秒かかります。その間マイクをONのままにしてください。")
```

**Purpose:**

- Inform users about the waiting time
- Warn them not to close/refresh the page
- Set expectations

### 2. Model Loading Progress Bar

**Added visual progress indicator:**

```python
elif st.session_state.get("_should_start_realtime", False):
    # Model loading phase - show progress
    if "_model_loading_start_time" in st.session_state:
        elapsed = time.time() - st.session_state["_model_loading_start_time"]
        progress = min(elapsed / 15.0, 1.0)

        st.progress(progress)
        st.info(f"⏳ モデル読み込み中... {elapsed:.1f}秒 / ~15秒 ({progress*100:.0f}%)")
        st.warning("💡 **重要:** マイクをONのまま待ってください！")
```

**Features:**

- Real-time progress bar (0-100%)
- Elapsed time display
- Reminder to keep microphone ON
- Updates automatically

### 3. WebRTC Disconnection Detection During Loading

**Added monitoring:**

```python
# Check if WebRTC is still alive during model loading
if not rtc_ctx.state.playing:
    st.error(f"❌ マイクが {elapsed:.1f}秒後に切断されました！")
    st.info("💡 'START'ボタンをもう一度クリックして、マイクを再度ONにしてください。")
```

**Benefits:**

- Immediate feedback if disconnection occurs
- Shows exactly when it disconnected
- Clear recovery instructions

### 4. Enhanced Error Messages

**Added timing information:**

```python
if not rtc_ctx or not rtc_ctx.audio_receiver or not rtc_ctx.state.playing:
    st.session_state["_should_start_realtime"] = False

    if "_model_loading_start_time" in st.session_state:
        elapsed = time.time() - st.session_state["_model_loading_start_time"]
        st.error(f"❌ WebRTC接続が切断されました（{elapsed:.1f}秒後）。")

    st.info("💡 **解決方法:** 'START'ボタンをクリック → マイクON → 'リアルタイム開始'をクリック → **15秒待つ間マイクをONのまま**にしてください。")
    st.info("💡 ヒント: モデルが一度読み込まれると、次回は瞬時に開始できます。")
```

**Information provided:**

- Exact timing of failure
- Step-by-step recovery instructions
- Optimization hint (model caching)

## User Experience Improvements

### Before Fixes

❌ **Confusing failure:**

```
User: *clicks button*
System: (silently fails)
User: "Why isn't it working?"
```

- No indication of what's happening
- Silent failure during model loading
- No recovery guidance
- User doesn't know to wait

### After Fixes

✅ **Clear communication:**

```
User: *clicks button*
System: "⏳ モデル読み込み中... 3.2秒 / ~15秒 (21%)"
        "💡 重要: マイクをONのまま待ってください！"
        [Progress bar: ████░░░░░░░░░░]

User: *waits patiently*
System: "✅ リアルタイムストリーミングを開始しました！"
```

- Clear progress indication
- Time estimates
- Warning reminders
- Success confirmation

## Technical Details

### Progress Calculation

```python
elapsed = time.time() - start_time
progress = min(elapsed / 15.0, 1.0)  # Cap at 100%
```

- Linear progress based on expected 15-second load time
- Capped at 100% if it takes longer
- Real-time updates

### State Management

**New session state variables:**

```python
st.session_state["_model_loading_start_time"]  # Timestamp when loading started
st.session_state["_should_start_realtime"]      # Flag for initialization
```

**Lifecycle:**

```
1. Button clicked:
   _should_start_realtime = True
   _model_loading_start_time = now()

2. During loading:
   Show progress based on elapsed time
   Monitor WebRTC state

3. After loading completes:
   _should_start_realtime = False
   realtime_running = True
   Clear _model_loading_start_time

4. If WebRTC disconnects:
   _should_start_realtime = False
   Show error with elapsed time
```

## Why WebRTC Disconnects

### Browser Behavior

Most browsers have WebRTC connection timeouts:

- **Chrome/Edge:** ~10-15 seconds without data flow
- **Firefox:** ~10 seconds without activity
- **Safari:** Variable, usually ~10 seconds

### ICE Connection States

```
new → checking → connected → completed →
  ↓ (no activity)
disconnected → failed
```

Without keep-alive or data flow, connection moves to "disconnected".

## Permanent Solutions (Future)

### Option 1: Model Preloading

**Load model when server starts:**

```python
# In backend startup
@app.on_event("startup")
async def startup_event():
    # Preload all models
    for model_name in ["conformer", "realtime"]:
        get_model_for_inference(model_name)
```

**Benefits:**

- Instant streaming start
- No waiting time for users
- Better UX

**Tradeoffs:**

- Longer server startup
- More memory usage
- Models always loaded even if not used

### Option 2: Keep-Alive Mechanism

**Send dummy audio frames during loading:**

```python
# In frontend, during model loading
while model_loading:
    # Send silent audio frames to keep connection alive
    silent_frame = np.zeros(480, dtype=np.float32)
    send_queue.put(silent_frame.tobytes())
    await asyncio.sleep(0.02)  # 50Hz
```

**Benefits:**

- Keeps WebRTC alive
- No timeout issues
- Works with any model loading time

**Tradeoffs:**

- More complex implementation
- Network bandwidth during loading
- Backend must handle pre-start frames

### Option 3: Lazy Model Loading

**Load smaller base model, then full model:**

```python
# Quick load: lightweight model (~1 second)
quick_model = load_quick_model()

# Start streaming immediately with quick model
start_streaming(quick_model)

# Background: load full model while streaming
full_model = load_full_model_async()

# Swap models when ready
swap_model(full_model)
```

**Benefits:**

- Instant start
- No timeout risk
- Graceful upgrade

**Tradeoffs:**

- Complex model management
- Potential quality difference
- More code complexity

## Current Workaround

**For Users:**

1. Click "START" → Enable microphone
2. Click "リアルタイム開始"
3. **Wait patiently for ~15 seconds**
4. Keep browser tab active
5. Don't touch anything during loading
6. Once started, subsequent uses are faster (model cached)

**Visual Indicators:**

- Progress bar shows loading progress
- Timer shows elapsed time
- Warnings remind to keep microphone ON
- Error messages if disconnection occurs

## Testing

### Test Scenario 1: Patient User

```
1. Click START → Microphone ON
2. Click "リアルタイム開始"
3. See progress bar: 0% → 10% → 20% → ... → 100%
4. Wait full 15 seconds
5. Expected: "✅ リアルタイムストリーミングを開始しました！"
6. Speak → See transcription
```

### Test Scenario 2: Impatient User

```
1. Click START → Microphone ON
2. Click "リアルタイム開始"
3. See progress: "⏳ 3.2秒 / ~15秒 (21%)"
4. User closes tab or refreshes page
5. Expected: Connection lost, need to restart
6. Clear error message shown
```

### Test Scenario 3: Second Attempt (Model Cached)

```
1. First attempt: Wait 15 seconds, success
2. Click "停止"
3. Click "リアルタイム開始" again
4. Expected: Instant start (model already loaded)
5. Progress shows 100% immediately
```

## Metrics

### Model Loading Time

**Measured from logs:**

```
T=04:50:27.278 - Start message received
T=04:50:40.950 - Model loaded
Duration: 13.67 seconds
```

### Progress Accuracy

- Expected: 15 seconds
- Actual: 13-14 seconds
- Progress bar slightly underestimates (better than overestimate)

## Summary

**Added user warnings, progress indication, and error handling for WebRTC
timeout during model loading.**

**Key improvements:**

- ⏳ Real-time progress bar (0-100%)
- ⏱️ Elapsed time display
- ⚠️ Warning reminders to keep microphone ON
- ❌ Clear error messages if disconnection occurs
- 💡 Recovery instructions
- 🎯 Hints about model caching

**Result:** Users are informed and guided through the waiting period, reducing
confusion and failed attempts.

**Future:** Consider model preloading or keep-alive mechanisms for instant
start.
