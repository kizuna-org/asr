# Fix: Realtime Streaming Not Starting After Button Click

## Problem

After clicking "リアルタイム開始", the debug info showed:

```
start_btn: True
rtc_ctx: True
rtc_ctx.state.playing: True
rtc_ctx.audio_receiver: True
realtime_running: False  ← Should be True!
```

WebRTC was working correctly, but streaming never started.

## Root Cause

**Streamlit's execution model caused a race condition:**

1. User clicks "リアルタイム開始" → `start_btn = True`
2. Code sets `st.session_state["realtime_running"] = True`
3. **Streamlit triggers automatic rerun** (because button was clicked)
4. On next render: `start_btn = False` (button state doesn't persist)
5. Condition `if start_btn and rtc_ctx and ...` is now False
6. Thread initialization code never runs!
7. Result: `realtime_running = False` (or threads never started)

## The Fix

**Use a two-phase approach with explicit rerun:**

### Phase 1: Button Click Handler

```python
if start_btn:
    # Validate conditions
    if not rtc_ctx:
        st.error("❌ WebRTCコンテキストが初期化されていません")
    elif not rtc_ctx.state.playing:
        st.error("❌ マイクが有効になっていません...")
    elif not rtc_ctx.audio_receiver:
        st.error("❌ オーディオレシーバーが初期化されていません...")
    else:
        st.success("✅ リアルタイム推論を開始します...")
        # Set flag to start threads on NEXT render
        st.session_state["_should_start_realtime"] = True
        st.rerun()  # Explicit rerun
```

### Phase 2: Thread Initialization (Next Render)

```python
if (st.session_state.get("_should_start_realtime", False) and
    not st.session_state.get("realtime_running", False) and
    rtc_ctx and rtc_ctx.audio_receiver and rtc_ctx.state.playing):

    # Clear the flag
    st.session_state["_should_start_realtime"] = False

    # Start threads
    st.session_state["realtime_running"] = True
    # ... initialize threads ...
    st.rerun()  # Rerun to show updated status
```

## Why This Works

**Separation of concerns:**

- **Button click** → Set flag, trigger rerun
- **Next render** → Check flag, start threads if conditions met
- **Subsequent renders** → Show streaming status

**Prevents race conditions:**

- Flag persists across reruns (session state)
- Thread initialization happens when all states are stable
- Explicit reruns ensure UI updates

**Idempotent:**

- Flag is cleared after use
- Check `realtime_running` to prevent double-start
- Clean state management

## Flow Diagram

```
User Action: Click "リアルタイム開始"
    ↓
[Render 1: Button Click]
    start_btn = True
    Set: _should_start_realtime = True
    Call: st.rerun()
    ↓
[Render 2: Thread Initialization]
    start_btn = False (button state lost)
    _should_start_realtime = True (persisted!)
    Check: All conditions met?
        Yes → Start threads
        Set: realtime_running = True
        Clear: _should_start_realtime = False
        Call: st.rerun()
    ↓
[Render 3+: Streaming Active]
    _should_start_realtime = False
    realtime_running = True
    Show: "🎙️ リアルタイムストリーミング実行中..."
```

## Additional Improvements

### 1. Status Indicators

```python
if st.session_state.get("realtime_running", False):
    st.info("🎙️ リアルタイムストリーミング実行中... 話してください！")
```

Users now see clear feedback that streaming is active.

### 2. Success Messages

```python
st.success("✅ リアルタイムストリーミングを開始しました！")
st.rerun()
```

Immediate feedback after thread initialization.

## Testing

### Expected Behavior:

**Initial State:**

```
start_btn: False
realtime_running: False
_should_start_realtime: False
```

**After Click:**

```
[Render 1]
start_btn: True
realtime_running: False
_should_start_realtime: True
Message: "✅ リアルタイム推論を開始します..."
```

**After Auto-Rerun:**

```
[Render 2]
start_btn: False
realtime_running: True
_should_start_realtime: False
Threads: Started!
Message: "✅ リアルタイムストリーミングを開始しました！"
```

**Subsequent Renders:**

```
[Render 3+]
start_btn: False
realtime_running: True
_should_start_realtime: False
Status: "🎙️ リアルタイムストリーミング実行中..."
```

## Code Changes Summary

**File:** `frontend/app.py`

**Changes:**

1. Added intermediate flag: `_should_start_realtime`
2. Button click handler: Set flag + explicit rerun
3. Separate condition: Check flag to start threads
4. Added status indicator during streaming
5. Added success message after thread start

## Deploy and Test

```bash
cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
./run.sh
```

**Test Steps:**

1. Open http://localhost:58080
2. Click "START" to enable microphone
3. Wait for "✅ マイクが有効です..."
4. Click "リアルタイム開始"
5. Should see: "✅ リアルタイム推論を開始します..."
6. Page auto-reloads
7. Should see: "✅ リアルタイムストリーミングを開始しました！"
8. Then see: "🎙️ リアルタイムストリーミング実行中... 話してください！"
9. Speak into microphone
10. Results should appear in real-time!

## Debug Tips

Check the debug panel:

- `start_btn` will be `True` only for one render
- `realtime_running` should become `True` after threads start
- `_should_start_realtime` flag should be `True` briefly, then `False`

If streaming doesn't start:

- Check if `_should_start_realtime` flag was set
- Check if all conditions (rtc_ctx, audio_receiver, playing) are met
- Look for error messages in the UI
- Check backend logs for WebSocket connection issues

## Key Takeaway

**In Streamlit, button states don't persist across reruns. Use session state
flags for multi-step operations that span multiple renders.**

This pattern is useful for any operation that:

- Requires state validation
- Needs to start background processes
- Must survive automatic reruns
- Requires clean state management
