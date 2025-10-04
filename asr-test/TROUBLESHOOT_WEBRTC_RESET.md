# Troubleshooting: WebRTC Reset Issue

## Problem Description

After clicking the "リアルタイム開始" (Start Realtime) button, the page
refreshes and the WebRTC state resets:

```
start_btn: False
rtc_ctx: True
rtc_ctx.state.playing: False  ← Should be True
rtc_ctx.audio_receiver: False ← Should be True
realtime_running: True        ← Session state persists but WebRTC resets
```

## Root Cause

This is a **Streamlit behavior** where clicking any button triggers a page
rerun, which causes the WebRTC component to reinitialize and lose its state.

## Solutions Implemented

### 1. Stricter Button Conditions

The "リアルタイム開始" button now requires ALL conditions to be met:

- ✅ WebRTC is playing (`rtc_ctx.state.playing == True`)
- ✅ Audio receiver is ready (`rtc_ctx.audio_receiver != None`)
- ✅ Model is selected
- ✅ Not already running

### 2. Better User Feedback

The UI now shows clear status messages:

**Before starting:**

- ⚠️ Warning if microphone is not enabled
- ✅ Success message when ready to start

**After starting:**

- ✅ Confirmation message
- ⚠️ Error if WebRTC stops unexpectedly

### 3. Debug Information (Collapsible)

The debug panel is now in an expander to reduce clutter:

- Click "🔍 デバッグ情報" to expand
- Shows all relevant state information

## Workaround Steps

If you still experience the WebRTC reset issue:

### Option A: Don't Navigate Away (Recommended)

1. **Click START** (enable microphone)
2. **Keep the START button in "STOP" state** - DO NOT CLICK ANYTHING ELSE
3. **Select model** from dropdown (this won't trigger a full rerun)
4. **Click "リアルタイム開始"**
5. The WebRTC should remain active

### Option B: Manual Recovery

If the WebRTC resets after starting:

1. **Click "リアルタイム停止"** to clean up
2. **Click the WebRTC "STOP" button**
3. **Click the WebRTC "START" button again**
4. Wait for microphone to become active
5. **Click "リアルタイム開始"** again

### Option C: Use Browser Console (Advanced)

Monitor the WebRTC state in browser console:

1. Press **F12** to open Developer Tools
2. Go to **Console** tab
3. Look for WebRTC-related messages
4. Check for errors like:
   - `MediaStreamTrack.stop()`
   - `WebRTC connection closed`
   - `Audio context suspended`

## Prevention Tips

### ✅ DO:

- Enable microphone FIRST (click START)
- Wait for confirmation that mic is active
- Then click "リアルタイム開始"
- Keep the page focused during streaming

### ❌ DON'T:

- Click multiple buttons rapidly
- Switch browser tabs during setup
- Refresh the page while streaming
- Click other UI elements during initialization

## Understanding the State

### Normal Startup Flow:

```
1. User clicks WebRTC "START"
   → rtc_ctx.state.playing = True
   → rtc_ctx.audio_receiver = initialized

2. User selects model
   → selected_realtime_model = "realtime"

3. User clicks "リアルタイム開始"
   → Streamlit reruns (normal behavior)
   → WebRTC state SHOULD persist if properly initialized

4. Audio streaming begins
   → realtime_running = True
   → Audio frames sent to backend
```

### What Goes Wrong:

```
1. User clicks "リアルタイム開始"
   → Streamlit reruns
   → WebRTC component reinitializes (BUG)
   → State is lost:
      - rtc_ctx.state.playing = False
      - rtc_ctx.audio_receiver = None
   → But session state persists:
      - realtime_running = True

2. Result: Mismatched state
   → Streaming thread is running
   → But no audio receiver to get frames from
   → Silent failure
```

## Technical Explanation

### Why This Happens

**Streamlit's Execution Model:**

- Every button click triggers a full script rerun
- Components are recreated from scratch
- Session state persists, but component state doesn't

**WebRTC Component:**

- Needs user gesture to enable microphone
- State managed by browser, not Python
- Gets reset when component is recreated

### The Fix

The updated code now:

1. **Checks audio_receiver** before allowing start
2. **Shows clear warnings** if prerequisites aren't met
3. **Detects state mismatch** (running but no receiver)
4. **Provides recovery instructions**

## Monitoring

### Check These Indicators:

**WebRTC Panel:**

- Button should say "STOP" (not "START")
- Microphone icon should be visible
- Audio level indicator should show activity

**Status Messages:**

- Look for "✅ マイクが有効です" (Microphone is enabled)
- Avoid "⚠️" warnings

**Debug Info (Expand):**

```
✅ Good state:
- rtc_ctx.state.playing: True
- rtc_ctx.audio_receiver: True
- realtime_running: False (before start) or True (after start)

❌ Bad state:
- rtc_ctx.state.playing: False
- rtc_ctx.audio_receiver: False
- realtime_running: True
^ This means WebRTC reset!
```

## Alternative Solutions (Future)

### Potential Improvements:

1. **Use st.experimental_rerun() sparingly**
   - Avoid automatic reruns after button clicks
   - Let user manually trigger updates

2. **WebSocket-only approach**
   - Skip WebRTC entirely
   - Use direct WebSocket audio streaming
   - More complex but more reliable

3. **Separate page for realtime**
   - Dedicated page with minimal reruns
   - Better state management
   - Less interference from other UI elements

4. **Session-based recovery**
   - Store WebRTC state in session
   - Auto-restart if detected as broken
   - Requires more complex state management

## Verification

After applying the fixes, you should see:

### On Fresh Page Load:

```
rtc_ctx.state.playing: False
rtc_ctx.audio_receiver: False
realtime_running: False
```

### After Clicking START:

```
rtc_ctx.state.playing: True
rtc_ctx.audio_receiver: True  ← Now initialized!
realtime_running: False
```

### After Clicking "リアルタイム開始":

```
rtc_ctx.state.playing: True   ← Should STAY True
rtc_ctx.audio_receiver: True  ← Should STAY True
realtime_running: True
```

### During Streaming:

```
- Partial results update in real-time
- Audio chunks being sent (check metrics)
- No error messages
```

## Next Steps

1. **Deploy the updated code:**
   ```bash
   cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
   ./run.sh
   ```

2. **Test the new flow:**
   - Click START
   - Wait for confirmation
   - Click "リアルタイム開始"
   - Verify streaming works

3. **Monitor the debug panel:**
   - Expand "🔍 デバッグ情報"
   - Check all values are correct
   - Watch for state changes

4. **Report if issue persists:**
   - Note exact steps taken
   - Screenshot of debug panel
   - Browser console errors

## Success Indicators

✅ You'll know it's working when:

- Button stays enabled after START
- No warnings appear
- Streaming starts immediately
- Results appear within ~1 second
- No need to restart WebRTC

## Need More Help?

If the issue persists after these fixes:

1. Check browser compatibility (use Chrome/Firefox)
2. Clear browser cache and reload
3. Try incognito mode
4. Check backend logs for errors
5. Test with different microphone

---

**Summary:** The WebRTC reset is a known Streamlit issue. The fixes add better
state checking and user feedback to work around this limitation.
