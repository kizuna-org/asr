# How to Use the Realtime Web Interface

This guide explains how to use the realtime speech recognition feature in the
web interface.

## Prerequisites

1. **Deploy the application to GPU server:**
   ```bash
   cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
   ./run.sh
   ```

2. **Wait for deployment to complete** (shows "🎉
   全てのポート転送が完了しました")

3. **Keep the terminal open** - The `run.sh` script must stay running to
   maintain port forwarding

## Accessing the Web Interface

1. Open your browser and navigate to:
   ```
   http://localhost:58080
   ```

2. You should see the ASR Training POC application interface

## Using Realtime Speech Recognition

### Step 1: Navigate to Main Dashboard

The realtime feature is on the **main page** (should be the default view).

Scroll down to find the section titled: **"リアルタイム推論（マイク入力）"**
(Realtime Inference with Microphone Input)

### Step 2: Enable Microphone Access

1. You'll see a **WebRTC audio panel** on the left side
2. Click the **"START"** button to enable microphone access
3. Your browser will prompt you to allow microphone access - **click "Allow"**
4. The button should change to **"STOP"** when active
5. You should see a microphone indicator showing that audio is being captured

### Step 3: Select a Model

On the right side panel ("🎯 リアルタイム推論設定"):

1. **Select Model**: Choose a model from the dropdown
   - `conformer` - General purpose ASR model
   - `realtime` - Optimized for real-time processing (recommended)

2. The interface shows: `選択されたモデル: **model_name**`

### Step 4: Configure Audio Settings

Under "🔊 音声設定":

- **送信サンプルレート** (Transmission Sample Rate):
  - Default: `48000` Hz
  - Options: `16000`, `22050`, `44100`, `48000`
  - Recommendation: Keep at `48000` for best quality

### Step 5: Start Realtime Recognition

Under "🎮 制御" (Control):

1. Click the **"リアルタイム開始"** (Start Realtime) button
   - The button will be **disabled** (grayed out) until:
     - ✅ WebRTC is started (microphone enabled)
     - ✅ A model is selected
     - ✅ Audio receiver is initialized

2. Once clicked, the system will:
   - Start capturing audio from your microphone
   - Send audio chunks to the backend server
   - Display transcription results in real-time

### Step 6: View Transcription Results

Below the control panel, you'll see:

- **部分結果 (Partial Results)**: Real-time, incremental transcription
  - Updates approximately every 1 second
  - Shows what the model is currently recognizing

- **最終結果 (Final Results)**: Complete transcription when stopped
  - Displayed after you click "Stop"

- **ステータス情報 (Status Information)**:
  - Connection status
  - Audio chunks sent
  - Model performance metrics

### Step 7: Stop Realtime Recognition

1. Click the **"リアルタイム停止"** (Stop Realtime) button
2. The system will:
   - Stop sending audio to the server
   - Display the final transcription result
   - Reset for the next session

3. Click the **"STOP"** button in the WebRTC panel to release the microphone

## Troubleshooting

### Problem: "START" button doesn't work

**Solution:**

- Check browser console for errors (F12 → Console tab)
- Ensure you've granted microphone permissions
- Try refreshing the page
- Check if your browser supports WebRTC (Chrome, Firefox, Edge recommended)

### Problem: "リアルタイム開始" button is disabled

**Checklist:**

1. ✅ Did you click "START" in the WebRTC panel?
2. ✅ Did you select a model from the dropdown?
3. ✅ Is the WebRTC state showing "playing"?
4. ✅ Check the "デバッグ情報" (Debug Info) section for details

**Debug Info should show:**

- `rtc_ctx.state.playing: True`
- `rtc_ctx.audio_receiver: True`
- `realtime_running: False` (before starting)

### Problem: No transcription appears

**Possible causes:**

1. **Model not trained yet**
   - Untrained models will produce gibberish or random characters
   - This is normal - train the model first using the training feature

2. **No audio being captured**
   - Check if microphone indicator is showing activity
   - Test your microphone in system settings
   - Try speaking louder or closer to the microphone

3. **Backend connection issues**
   - Check if backend is running: `docker compose ps` on GPU server
   - Check backend logs:
     ```bash
     ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose logs -f asr-api"
     ```

4. **WebSocket connection failed**
   - Look for error messages in the "ステータス情報" section
   - Check if port 58081 is accessible: `curl http://localhost:58081/api/test`

### Problem: Transcription is gibberish

**This is normal for untrained models!**

The model needs to be trained first:

1. Go to the training section on the main page
2. Select a dataset (e.g., `ljspeech`)
3. Select the `realtime` model
4. Click "学習開始" (Start Training)
5. Wait for training to complete (or at least a few epochs)
6. Try realtime inference again

### Problem: Audio is choppy or delayed

**Solutions:**

- Reduce sample rate to `16000` Hz
- Check CPU/GPU usage on the server
- Ensure stable network connection
- Close other applications using the microphone

### Problem: "Error parsing WebSocket message"

**Solutions:**

- Refresh the page
- Stop and restart the realtime session
- Check backend logs for detailed error messages
- Ensure backend container is running properly

## Advanced Features

### Debug Information

The interface shows real-time debug info:

- `start_btn`: Whether start button was clicked
- `rtc_ctx`: WebRTC context status
- `rtc_ctx.state.playing`: Audio streaming status
- `rtc_ctx.audio_receiver`: Audio receiver status
- `realtime_running`: Current session status

### Status Messages

The system displays various status messages:

- `{"type": "status", "payload": {"status": "ready"}}` - Ready to receive audio
- `{"type": "partial", "payload": {"text": "..."}}` - Partial transcription
- `{"type": "final", "payload": {"text": "..."}}` - Final transcription
- `{"type": "error", "payload": {"message": "..."}}` - Error occurred

## Architecture Overview

```
┌─────────────┐     WebRTC      ┌──────────────┐
│  Browser    │────────────────>│  Streamlit   │
│  (Frontend) │                 │  Frontend    │
└─────────────┘                 └──────────────┘
                                       │
                                WebSocket
                                       │
                                       ▼
                                ┌──────────────┐
                                │   FastAPI    │
                                │   Backend    │
                                └──────────────┘
                                       │
                                   Inference
                                       │
                                       ▼
                                ┌──────────────┐
                                │ Realtime ASR │
                                │    Model     │
                                └──────────────┘
```

**Flow:**

1. Browser captures microphone audio via WebRTC
2. Streamlit frontend sends audio chunks via WebSocket
3. FastAPI backend receives and buffers audio
4. Model performs inference every ~1 second
5. Partial results sent back to frontend
6. Frontend displays transcription in real-time

## Tips for Best Results

1. **Train the model first** before expecting meaningful transcriptions
2. **Speak clearly** and at a normal pace
3. **Use a good microphone** for better audio quality
4. **Minimize background noise** for better accuracy
5. **Keep sessions short** (< 1 minute) for best performance
6. **Monitor the debug info** to understand what's happening

## Next Steps

After getting realtime inference working:

1. **Train the model** with your dataset
2. **Test with different audio** (various speakers, accents)
3. **Monitor performance metrics** (latency, accuracy)
4. **Adjust model parameters** in `config.yaml` if needed
5. **Export trained models** for production use

## Support

If you continue to have issues:

1. Check the detailed logs in the backend container
2. Review the `REALTIME_INFERENCE_FIX.md` document
3. Run the test script: `./test_realtime_inference.sh`
4. Check the GitHub issues for similar problems
