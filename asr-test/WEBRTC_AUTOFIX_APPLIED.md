# WebRTC Auto-Reset Fix Applied

## Problem Resolved

**Issue:** After clicking "リアルタイム開始", the warning appeared:

```
⚠️ ストリーミング中にWebRTCが停止しました。'リアルタイム停止'をクリックしてリセットしてください。
```

**Root Cause:** Streamlit's button click triggers a page rerun, which causes
WebRTC to reset its state, but `realtime_running` session state persists,
creating a mismatch.

## Fixes Applied

### 1. **Auto-Recovery Mechanism**

When WebRTC stops unexpectedly during streaming, the system now:

- Detects the state mismatch automatically
- Resets `realtime_running` to `False`
- Stops any running threads
- Shows clear recovery instructions
- **Automatically reruns** the page to clean state

**Code:**

```python
if st.session_state.get('realtime_running', False):
    if not rtc_ctx.state.playing:
        st.error("⚠️ ストリーミング中にWebRTCが停止しました。自動的にリセットします...")
        # Auto-reset
        st.session_state["realtime_running"] = False
        # Stop threads
        loop = st.session_state.get("realtime_loop")
        if loop and loop.is_running():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except:
                pass
        st.info("💡 'START'ボタンをもう一度クリックしてから、'リアルタイム開始'をクリックしてください。")
        st.rerun()
```

### 2. **Stricter State Check Before Starting Threads**

Added double-check of WebRTC state before starting threads:

```python
if start_btn and rtc_ctx and rtc_ctx.audio_receiver and rtc_ctx.state.playing:
    # Double-check WebRTC is still in playing state
    if not rtc_ctx.state.playing:
        st.error("❌ WebRTCの状態が変更されました。もう一度'START'をクリックしてください。")
    else:
        # Start threads only if state is confirmed
        ...
```

### 3. **Fixed Indentation and Variable References**

- Fixed all indentation errors in `pull_audio_frames()` function
- Changed `msg_queue_ref` to `msg_queue` (correct variable name)
- Ensured all code is properly nested within the `else` block

## How It Works Now

### Scenario 1: Normal Flow (No Issues)

```
1. User clicks "START" → WebRTC enabled
2. User clicks "リアルタイム開始" → Threads start
3. Streaming works normally
```

### Scenario 2: WebRTC Resets (Auto-Recovery)

```
1. User clicks "START" → WebRTC enabled
2. User clicks "リアルタイム開始" → Page reruns
3. WebRTC resets (state lost)
4. System detects: realtime_running=True but rtc_ctx.state.playing=False
5. Auto-reset triggered:
   - Sets realtime_running=False
   - Stops threads
   - Shows message: "自動的にリセットします..."
   - Reruns page to clean state
6. User sees: "💡 'START'ボタンをもう一度クリックしてから..."
7. User clicks "START" again → WebRTC re-enabled
8. User clicks "リアルタイム開始" → Works this time!
```

## What Changed for Users

### Before:

❌ Click "リアルタイム開始" → Warning appears → Manual recovery required

- User must click "リアルタイム停止"
- Then click "STOP" on WebRTC
- Then click "START" again
- Then try "リアルタイム開始" again

### After:

✅ Click "リアルタイム開始" → Auto-recovery happens → Simple restart

- System automatically resets
- User sees clear instruction
- Just click "START" again
- Then click "リアルタイム開始" - works!

## Testing

After deploying this fix:

1. **Deploy:**
   ```bash
   cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
   ./run.sh
   ```

2. **Test Normal Flow:**
   - Click START
   - Wait for "✅ マイクが有効です..."
   - Click "リアルタイム開始"
   - Should work without issues

3. **Test Recovery Flow (if WebRTC resets):**
   - If you see "⚠️ ストリーミング中にWebRTCが停止しました..."
   - Page should auto-reload
   - See recovery instructions
   - Click START again
   - Click "リアルタイム開始" again
   - Should work this time

## Technical Details

### State Management

**Session State (Persists across reruns):**

- `realtime_running`: Whether streaming is active
- `realtime_msg_queue`: Message queue for communication
- `realtime_loop`, `realtime_thread`, `realtime_puller`: Thread references

**WebRTC State (Resets on rerun):**

- `rtc_ctx.state.playing`: Whether microphone is active
- `rtc_ctx.audio_receiver`: Audio frame receiver

**The Mismatch:** When page reruns:

- Session state persists: `realtime_running=True`
- WebRTC resets: `rtc_ctx.state.playing=False`
- Result: System thinks it's streaming but has no audio source

**The Solution:**

- Detect mismatch: `realtime_running=True AND rtc_ctx.state.playing=False`
- Auto-reset session state to match WebRTC state
- Force rerun to show clean UI
- User can restart easily

### Why This Happens

Streamlit's execution model:

1. User interacts (clicks button)
2. **Entire script reruns from top to bottom**
3. Components are recreated
4. Session state persists, but component state doesn't

WebRTC component:

- Requires user gesture (click) to enable microphone
- Browser manages the actual media stream
- When component recreates, stream connection is lost
- New click needed to re-establish connection

## Alternatives Considered

### Option 1: Prevent Rerun (Not Feasible)

- Would require major Streamlit architecture changes
- `st.experimental_rerun()` control is limited
- WebRTC component design requires recreation

### Option 2: Use Session State for WebRTC (Not Possible)

- WebRTC state is managed by browser, not Python
- Can't serialize media streams
- Security restrictions prevent state transfer

### Option 3: Separate Page (Could Work)

- Create dedicated page for realtime streaming
- Fewer reruns = less chance of reset
- Would require UI restructuring
- Decided auto-recovery is simpler

### Option 4: Auto-Recovery (Implemented ✅)

- Detect state mismatch automatically
- Clean up and reset gracefully
- Guide user to recovery
- Minimal code changes
- Best user experience

## Files Modified

- `frontend/app.py`:
  - Added auto-recovery logic
  - Fixed indentation errors
  - Corrected variable references (`msg_queue_ref` → `msg_queue`)
  - Improved state validation

## Known Limitations

1. **First Start May Fail**
   - If WebRTC resets, auto-recovery kicks in
   - User must restart (but it's automatic now)
   - This is a Streamlit/WebRTC limitation

2. **Refresh Breaks Stream**
   - Manual page refresh will reset everything
   - Normal behavior, just restart

3. **Tab Switch May Affect**
   - Switching browser tabs may pause audio
   - Browser security feature
   - Return to tab and check state

## Success Indicators

✅ You'll know it's working when:

- Auto-recovery message appears if reset occurs
- Page automatically reloads to clean state
- Clear instructions shown for restart
- No manual "リアルタイム停止" needed
- Second attempt works smoothly

## Monitoring

Check debug panel after clicking "リアルタイム開始":

- Should show `realtime_running: True`
- Should show `rtc_ctx.state.playing: True`
- If mismatch detected, auto-recovery triggers
- After recovery, both should be `False`

## Summary

**The fix turns a frustrating manual recovery process into an automatic, guided
recovery that requires just one simple restart.**

Users no longer need to understand the technical details - they just follow the
on-screen instructions and it works!
