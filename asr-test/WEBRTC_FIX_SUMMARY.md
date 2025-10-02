# WebRTC Reset Fix - Quick Summary

## The Problem

You click "リアルタイム開始" → Page refreshes → WebRTC resets:

- `rtc_ctx.state.playing: False` ❌ (should be True)
- `rtc_ctx.audio_receiver: False` ❌ (should be True)
- `realtime_running: True` ✓ (but useless without audio receiver)

## Why It Happens

**Streamlit reruns the entire script** when you click a button, causing the
WebRTC component to reinitialize and lose its state.

## The Fix

I've updated the code to:

1. **Check audio_receiver before allowing start**
2. **Show clear status messages:**
   - ⚠️ "先に上の'START'ボタンをクリックしてマイクを有効にしてください" (if mic
     not ready)
   - ✅ "マイクが有効です。'リアルタイム開始'ボタンをクリックしてください" (when
     ready)
3. **Stricter button conditions** - won't enable until ALL conditions met
4. **Better error detection** - warns if WebRTC stops during streaming

## How to Use Now

### Correct Order:

```
1. Click WebRTC "START" button
   ↓
2. See: ✅ "マイクが有効です..."
   ↓
3. Select model from dropdown
   ↓
4. Click "リアルタイム開始"
   ↓
5. Speak and see results!
```

### What Changed:

**Before (broken):**

- Button was enabled even without audio_receiver
- No clear feedback about what's wrong
- Silent failures

**After (fixed):**

- Button only enables when EVERYTHING is ready
- Clear status messages at each step
- Warnings if something goes wrong

## Deploy the Fix

```bash
cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
./run.sh
```

Wait for deployment, then test at: http://localhost:58080

## Verify It's Working

Check the debug panel (click "🔍 デバッグ情報"):

**Should show this when ready:**

```
rtc_ctx.state.playing: True       ✅
rtc_ctx.audio_receiver: True      ✅
realtime_running: False           ✅
```

**After clicking "リアルタイム開始":**

```
rtc_ctx.state.playing: True       ✅ (stays True!)
rtc_ctx.audio_receiver: True      ✅ (stays True!)
realtime_running: True            ✅
```

## If Still Not Working

Try this manual workaround:

1. If WebRTC resets, click "リアルタイム停止"
2. Click WebRTC "STOP" then "START" again
3. Wait for green checkmark
4. Try "リアルタイム開始" again

## Files Modified

- `frontend/app.py` - Added better state checking and user feedback
- `backend/app/models/realtime.py` - Fixed inference logging and thresholds
- `backend/Dockerfile` - Added test scripts to container

## More Details

See comprehensive guides:

- `HOW_TO_USE_REALTIME_WEB_INTERFACE.md` - Complete usage guide
- `TROUBLESHOOT_WEBRTC_RESET.md` - Detailed troubleshooting
- `QUICK_START_REALTIME.md` - Quick visual guide

---

**TL;DR:** Make sure to click START and wait for the green checkmark before
clicking "リアルタイム開始". The button will stay disabled until everything is
ready.
