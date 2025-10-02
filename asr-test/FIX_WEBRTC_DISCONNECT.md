# Fix: WebRTC Disconnection During Realtime Streaming

## Problem

When clicking "リアルタイム開始", the WebRTC connection would:

1. Establish successfully (`iceconnectionstatechange checking`)
2. Complete handshake (`Remote description is set`)
3. **Immediately disconnect** (`iceconnectionstatechange disconnected`)
4. **Restart** and repeat the cycle

Console logs showed:

```javascript
VM514 main.9c17b353.js:2 iceconnectionstatechange checking
VM514 main.9c17b353.js:2 Remote description is set
VM514 main.9c17b353.js:2 iceconnectionstatechange disconnected
// Page reloads (notice VM number change)
VM586 main.9c17b353.js:2 RTCConfiguration: {}  ← NEW instance
```

## Root Cause

**Training progress auto-polling was causing continuous page reruns:**

```python
if st.session_state.is_training:
    ...
    time.sleep(1)
    st.rerun()  ← Reruns every second!
```

When training is active, the app automatically reruns **every second** to update
progress bars. This continuous rerunning caused:

- WebRTC component to reinitialize
- Microphone connection to drop
- Audio streaming to fail
- User confusion

## Timeline Analysis

**What happens when you click "リアルタイム開始" with training active:**

```
T=0.0s: User clicks "リアルタイム開始"
        ↓ Button state: start_btn=True
        ↓ Flag set: _should_start_realtime=True
        ↓ Threads start successfully
        ↓ WebRTC establishes connection

T=0.5s: WebRTC handshake completes
        ✅ iceconnectionstatechange: checking → connected

T=1.0s: Training progress timer fires
        ↓ Condition: st.session_state.is_training = True
        ↓ Action: time.sleep(1) + st.rerun()
        ❌ Page reruns!

T=1.1s: Page reloads
        ❌ WebRTC component reinitializes
        ❌ Connection state: disconnected
        ❌ Threads lose WebRTC reference

T=1.5s: New WebRTC instance created
        ↓ Tries to reconnect
        ↓ Gets new offer/answer

T=2.0s: Training progress timer fires AGAIN
        ↓ Another st.rerun()
        ❌ Cycle repeats infinitely!
```

**Result:** WebRTC keeps disconnecting/reconnecting in an infinite loop.

## The Fix

**Skip training progress polling when realtime streaming is active:**

### Before

```python
if st.session_state.is_training:
    # Update progress
    ...
    time.sleep(1)
    st.rerun()  # Runs ALWAYS when training
```

### After

```python
if st.session_state.is_training and not st.session_state.get("realtime_running", False):
    # Update progress
    ...
    # Note: リアルタイムストリーミング中は自動ポーリングを停止（WebRTCの安定性のため）
    time.sleep(1)
    st.rerun()  # Only runs when NOT streaming
```

### Key Change

```python
# Added condition:
and not st.session_state.get("realtime_running", False)
```

When `realtime_running=True`, the training progress auto-polling is **paused**.

## Why This Works

### Conflicting Requirements

**Training Progress:**

- Needs regular updates (every 1 second)
- Uses `st.rerun()` to refresh UI
- Page reload is acceptable

**Realtime Streaming:**

- Needs stable WebRTC connection
- Cannot tolerate page reloads
- Component state must persist

**Solution:** Prioritize realtime streaming when both are active.

### New Behavior

```
Training Active + Streaming Inactive:
    → Auto-polling enabled
    → Progress updates every second
    → st.rerun() every second

Training Active + Streaming Active:
    → Auto-polling DISABLED
    → WebRTC stays stable
    → No st.rerun() during streaming
    → Progress frozen (acceptable trade-off)

Training Inactive + Streaming Active:
    → No auto-polling anyway
    → WebRTC stable
    → Streaming works perfectly
```

## Code Changes

**File:** `frontend/app.py`

**Line 980:** Modified condition

```python
# Before:
if st.session_state.is_training:

# After:
if st.session_state.is_training and not st.session_state.get("realtime_running", False):
```

**Added comment:** Line 998

```python
# Note: リアルタイムストリーミング中は自動ポーリングを停止（WebRTCの安定性のため）
```

## Impact

### Before Fix

❌ **WebRTC unstable during training:**

- Disconnects every second
- Audio streaming fails
- Cannot use realtime inference while training
- Confusing user experience

### After Fix

✅ **WebRTC stable during training:**

- Connection stays active
- Audio streaming works
- Can use realtime inference while training
- Smooth user experience

### Trade-off

⚠️ **Training progress updates pause during streaming:**

- Progress bars freeze while streaming
- Loss/metrics not updated in real-time
- **Acceptable** because:
  - User is focused on streaming task
  - Training continues in background
  - Progress resumes when streaming stops

## Testing

### Test Scenario 1: No Training Active

```bash
1. Start frontend without training
2. Click "START" → Enable microphone
3. Click "リアルタイム開始"
4. ✅ WebRTC connects and stays connected
5. ✅ Speak → See results
6. ✅ No disconnections
```

**Expected:** Works perfectly (no auto-polling).

### Test Scenario 2: Training Active

```bash
1. Start training
2. Wait for progress updates (every 1 second)
3. Click "START" → Enable microphone
4. Click "リアルタイム開始"
5. ✅ WebRTC connects and stays connected
6. ✅ Speak → See results
7. ⚠️ Progress bars freeze during streaming
8. Click "停止"
9. ✅ Progress updates resume
```

**Expected:** Streaming works, progress paused.

### Test Scenario 3: Start Streaming, Then Training

```bash
1. Start realtime streaming
2. Speak → See results working
3. Start training (from another session/API)
4. ✅ Streaming continues uninterrupted
5. ✅ WebRTC stays stable
6. ⚠️ Progress not visible (that's OK)
```

**Expected:** Streaming takes priority.

## Debugging

### Check Current States

```python
# In Streamlit UI debug panel:
is_training: True/False
realtime_running: True/False
```

### Verify Polling Behavior

```python
# When BOTH True:
if is_training and not realtime_running:
    # This block should NOT execute
    time.sleep(1)
    st.rerun()
```

### Console Logs

**Before fix:**

```
iceconnectionstatechange disconnected  ← Every second
(new VM instance)  ← Page reload
RTCConfiguration: {}  ← New WebRTC
```

**After fix:**

```
iceconnectionstatechange checking
iceconnectionstatechange connected  ← Stays connected!
(no more VM changes)  ← No reloads
```

## Alternative Solutions Considered

### Option 1: Use WebSocket Only (No WebRTC)

**Pros:**

- No browser WebRTC complexity
- More control over connection

**Cons:**

- Harder to access microphone
- More JavaScript/browser API work
- Security restrictions (HTTPS required)

**Decision:** Rejected (too much rework).

### Option 2: Separate Page for Realtime

**Pros:**

- Complete isolation from training page
- No state conflicts

**Cons:**

- Worse UX (switching pages)
- Duplicated UI elements
- More complex navigation

**Decision:** Rejected (current fix is simpler).

### Option 3: Pause Training During Streaming

**Pros:**

- No conflicts at all
- Simpler logic

**Cons:**

- Training interruption unacceptable
- Users should be able to do both

**Decision:** Rejected (too limiting).

### Option 4: Current Solution ✅

**Pros:**

- Minimal code change (one line)
- Both features work (streaming + training)
- Only UI polling paused (training continues)
- Acceptable trade-off

**Cons:**

- Progress not visible during streaming

**Decision:** ✅ **ACCEPTED** - Best balance.

## Future Improvements

### Option A: Manual Progress Refresh Button

```python
if st.session_state.is_training and st.session_state.get("realtime_running"):
    st.info("⏸️ 進捗表示を一時停止中（リアルタイムストリーミング実行中）")
    if st.button("🔄 進捗を更新"):
        update_progress_from_backend()
```

**Benefit:** User can manually check progress without auto-rerun.

### Option B: Server-Sent Events (SSE)

Use SSE instead of polling for progress updates:

- Backend pushes updates
- No need for `st.rerun()`
- WebRTC unaffected

**Benefit:** Best of both worlds.

### Option C: WebSocket for Progress

Similar to SSE, use WebSocket for progress:

- Real-time updates without polling
- No page reruns needed

**Benefit:** More standard protocol.

## Summary

**Fixed WebRTC disconnection by preventing training progress auto-polling during
realtime streaming.**

**Key change:**

```python
if st.session_state.is_training and not st.session_state.get("realtime_running", False):
```

**Result:**

- ✅ WebRTC stays connected during streaming
- ✅ Audio streaming works reliably
- ✅ Training continues in background
- ⚠️ Progress UI frozen during streaming (acceptable)

**Trade-off accepted:** Progress visibility sacrificed for streaming stability.
