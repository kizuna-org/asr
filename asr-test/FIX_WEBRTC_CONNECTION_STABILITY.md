# Fix: WebRTC Connection Stability Improvements

## Problem

The WebRTC connection was stopping/disconnecting after a short period of time
during realtime streaming, causing:

- Audio streaming to stop unexpectedly
- Silent failures without user notification
- Need to manually restart streaming frequently

## Root Causes

### 1. Missing ICE Configuration

**Problem:** No STUN/TURN servers configured for NAT traversal.

**Impact:**

- Connection can fail in certain network conditions
- ICE candidates may not be gathered properly
- Connection timeouts more likely

### 2. No Connection State Monitoring

**Problem:** Application didn't detect when WebRTC connection was lost.

**Impact:**

- Threads kept running even after connection died
- No user feedback about connection status
- Resources wasted on dead connections

### 3. No Error Recovery

**Problem:** No handling for connection degradation or loss.

**Impact:**

- Silent failures
- Audio frames dropped without notification
- No automatic cleanup

## Solutions Implemented

### 1. Added ICE Server Configuration

**File:** `frontend/app.py`

**Location:** Line ~1170

**Added:**

```python
# WebRTC設定 - ICE接続の安定性を向上
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN server
    ],
    "iceTransportPolicy": "all",  # すべてのICE候補を使用
}

rtc_ctx = webrtc_streamer(
    key="asr-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=2048,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    rtc_configuration=rtc_configuration,  # ICE設定を追加
)
```

**Benefits:**

- ✅ Better NAT traversal
- ✅ More reliable connection establishment
- ✅ Improved connection stability
- ✅ Works across different network configurations

**How it works:**

- STUN server helps discover public IP address
- ICE candidates gathered from multiple sources
- Better connectivity in restricted networks

### 2. Added WebRTC Connection State Monitoring

**File:** `frontend/app.py`

**Location:** After streaming status display (~line 1450)

**Added:**

```python
# WebRTC接続状態を監視
if rtc_ctx and rtc_ctx.state:
    if not rtc_ctx.state.playing:
        st.warning("⚠️ WebRTC接続が切断されました。STARTボタンを押して再接続してください。")
        # 自動的にストリーミングを停止
        if st.session_state.get("realtime_running"):
            logger.warning("WebRTC disconnected, stopping streaming")
            st.session_state["realtime_running"] = False
            loop = st.session_state.get("realtime_loop")
            if loop and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
    elif rtc_ctx.audio_receiver is None:
        st.warning("⚠️ オーディオレシーバーが利用できません。")
```

**Benefits:**

- ✅ User notified when connection lost
- ✅ Automatic cleanup of streaming threads
- ✅ Clear instructions for recovery
- ✅ Prevents resource waste

**How it works:**

- Checks `rtc_ctx.state.playing` on every render
- If `False`, connection is lost
- Automatically stops streaming threads
- Shows warning to user

### 3. Added Connection Monitoring in Audio Puller Thread

**File:** `frontend/app.py`

**Location:** Inside `pull_audio_frames()` function (~line 1310)

**Added:**

```python
consecutive_errors = 0
max_consecutive_errors = 50  # 50回連続エラーで停止

while running_flag:
    # WebRTC接続状態をチェック
    if not rtc_ctx.state.playing:
        logger.warning("WebRTC connection lost (not playing), stopping puller")
        if msg_queue:
            try:
                msg_queue.put({"type": "error", "payload": {"message": "WebRTC接続が切断されました"}})
            except Exception:
                pass
        break

    if rtc_ctx.audio_receiver:
        # ... audio processing ...
```

**Benefits:**

- ✅ Thread stops immediately when connection lost
- ✅ No wasted CPU cycles on dead connection
- ✅ Error message sent to UI
- ✅ Clean thread termination

### 4. Added Error Counting and Threshold Detection

**Location:** Inside audio frame processing loop

**Added:**

```python
try:
    send_queue.put(pcm_f32.tobytes(), timeout=0.05)
    frames_sent += 1
    consecutive_errors = 0  # 成功したらエラーカウントリセット
except queue.Full:
    consecutive_errors += 1
    # ...

# 連続エラーが多すぎる場合は停止
if consecutive_errors >= max_consecutive_errors:
    logger.error("Too many consecutive errors, stopping puller")
    if msg_queue:
        msg_queue.put({"type": "error", "payload": {"message": "音声処理エラーが多発しています"}})
    break
```

**Benefits:**

- ✅ Detects connection degradation
- ✅ Prevents infinite error loops
- ✅ Automatic shutdown on persistent failure
- ✅ User notification of problem

**How it works:**

- Counts consecutive errors
- Resets to 0 on successful frame processing
- If errors exceed threshold (50), stop thread
- Sends error message to UI

## Impact

### Before Fixes

❌ **Unstable connection:**

- Disconnects after short time
- No NAT traversal support
- Silent failures
- Threads keep running after connection lost
- No user feedback

### After Fixes

✅ **Stable connection:**

- STUN server for NAT traversal
- Connection state monitored
- Automatic cleanup on disconnect
- User notified of issues
- Error threshold detection

## Connection Flow

### Normal Operation

```
1. User clicks START
   → WebRTC establishes with STUN
   → ICE candidates gathered
   → Connection established

2. User clicks "リアルタイム開始"
   → Threads start
   → Connection state monitored
   → Audio frames flowing

3. Streaming active
   → WebRTC state checked every render
   → Puller thread checks state in loop
   → Errors counted and tracked
   → Everything working normally

4. User clicks "停止"
   → Clean shutdown
   → Resources released
```

### Connection Lost Scenario

```
1. Streaming active
   → Suddenly network issue
   → WebRTC state.playing = False

2. Next render:
   → State monitoring detects issue
   → Warning shown to user
   → Streaming threads stopped
   → realtime_running = False

3. Puller thread:
   → Detects state.playing = False
   → Sends error to message queue
   → Exits loop
   → Thread terminates

4. User sees:
   → "⚠️ WebRTC接続が切断されました"
   → "STARTボタンを押して再接続してください"
   → Clear recovery path
```

### Error Threshold Scenario

```
1. Streaming active
   → Connection degrades
   → Frames start failing

2. Error counting:
   → consecutive_errors: 1, 2, 3, ...
   → Keeps trying to process
   → Errors continue

3. Threshold reached (50):
   → Logger.error("Too many consecutive errors")
   → Error message to UI
   → Thread stops
   → Clean exit

4. User sees:
   → "音声処理エラーが多発しています"
   → Can restart streaming
```

## Testing

### Test 1: Normal Streaming

```bash
1. Start application
2. Click START → Enable microphone
3. Click "リアルタイム開始"
4. Speak for 2-3 minutes
5. Expected: Continuous streaming without disconnection
```

### Test 2: Connection Loss Detection

```bash
1. Start streaming
2. Disable network (or close browser tab)
3. Expected: Warning shown, threads stopped
```

### Test 3: Error Recovery

```bash
1. Start streaming
2. Wait for any errors in console
3. Expected: Automatic recovery or clean shutdown after threshold
```

## Configuration

### STUN Server

**Currently using:** `stun:stun.l.google.com:19302`

**To change:**

```python
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:YOUR_STUN_SERVER:PORT"]},
    ],
}
```

### Error Threshold

**Currently:** 50 consecutive errors

**To change:**

```python
max_consecutive_errors = 100  # Your desired threshold
```

### Connection Check Frequency

**Currently:** Every render cycle + every audio frame

**To change monitoring frequency:**

- Adjust Streamlit rerun frequency
- Add explicit timer in puller thread

## Troubleshooting

### Still Disconnecting Quickly

**Check:**

1. Firewall blocking STUN traffic
2. Network restrictions (corporate proxy)
3. Browser WebRTC settings
4. Console errors in browser

**Solutions:**

- Add TURN server for relay
- Check browser console for ICE errors
- Test on different network

### Too Many Error Messages

**If threshold too low:**

```python
max_consecutive_errors = 100  # Increase
```

**If errors legitimate:**

- Check audio device
- Check microphone permissions
- Check sample rate compatibility

### Memory/CPU Usage High

**If connection monitoring too frequent:**

- Errors are already at DEBUG level
- State checking is lightweight
- Should not impact performance

## Advanced Configuration

### Add TURN Server (For Restricted Networks)

```python
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:YOUR_TURN_SERVER:PORT"],
            "username": "user",
            "credential": "pass"
        }
    ],
    "iceTransportPolicy": "all",
}
```

### Adjust ICE Transport Policy

```python
# Only use relay (TURN) servers
"iceTransportPolicy": "relay"

# Use all candidates (STUN + TURN + host)
"iceTransportPolicy": "all"  # Default, recommended
```

## Summary

**Improved WebRTC connection stability through:**

1. ✅ **ICE configuration** - STUN server for NAT traversal
2. ✅ **State monitoring** - Detect connection loss immediately
3. ✅ **Automatic cleanup** - Stop threads when connection dies
4. ✅ **Error threshold** - Detect degraded connections
5. ✅ **User feedback** - Clear warnings and recovery instructions

**Result:** More reliable realtime streaming with graceful error handling and
user notifications.
