# Realtime Web Interface - Quick Start Guide

## 🚀 Quick Start (5 Steps)

### 1️⃣ Deploy to GPU Server

```bash
cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
./run.sh
# Keep this terminal open!
```

### 2️⃣ Open Web Interface

Open browser: **http://localhost:58080**

### 3️⃣ Enable Microphone

- Click **"START"** button in WebRTC panel
- Allow microphone access in browser prompt

### 4️⃣ Select Model

- Choose **"realtime"** from dropdown menu
- Keep sample rate at **48000 Hz**

### 5️⃣ Start Recognition

- Click **"リアルタイム開始"** button
- Speak into your microphone
- See transcription appear in real-time!

## 📸 Visual Guide

```
┌────────────────────────────────────────────────────────────┐
│  ASR Training POC                                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  リアルタイム推論（マイク入力）                                │
│                                                            │
│  ┌─────────────────────┐   ┌───────────────────────────┐  │
│  │  WebRTC Audio       │   │ 🎯 リアルタイム推論設定     │  │
│  │                     │   │                           │  │
│  │  [START] ◄─────────┼───┼─ 1. Click START          │  │
│  │                     │   │                           │  │
│  │  🎤 Microphone      │   │ リアルタイム用モデル:        │  │
│  │     Active          │   │ [realtime        ▼] ◄────┼──┼─ 2. Select Model
│  │                     │   │                           │  │
│  └─────────────────────┘   │ 🔊 音声設定                │  │
│                            │ 送信サンプルレート:          │  │
│                            │ [48000] Hz                │  │
│                            │                           │  │
│                            │ 🎮 制御                    │  │
│                            │ [リアルタイム開始] ◄───────┼──┼─ 3. Click Start
│                            │ [リアルタイム停止]           │  │
│                            └───────────────────────────┘  │
│                                                            │
│  📝 結果表示:                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 部分結果: "hello this is a test..."                 │ ◄─┼─ 4. See Results
│  │ 最終結果: (stopped時に表示)                          │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## ⚠️ Common Issues

### ❌ Button is Grayed Out

**Check:**

- ✅ WebRTC "START" clicked?
- ✅ Model selected?
- ✅ Microphone access granted?

**Debug Info should show:**

```
- rtc_ctx.state.playing: True  ← Must be True
- rtc_ctx.audio_receiver: True ← Must be True
- realtime_running: False      ← Should be False before starting
```

### ❌ No Transcription Appears

**Possible reasons:**

1. **Model not trained** → Train it first! (Normal to see gibberish)
2. **No audio detected** → Check microphone settings
3. **Backend error** → Check logs: `docker compose logs asr-api`

### ❌ Browser Won't Allow Microphone

**Solutions:**

- Use **Chrome**, **Firefox**, or **Edge** (Safari may have issues)
- Check browser permissions: `Settings → Privacy → Microphone`
- Try **HTTPS** or **localhost** (required for WebRTC)

## 🎯 Expected Behavior

### ✅ With Trained Model:

```
You say:  "Hello world"
Results:  "hello world" (or close approximation)
```

### ⚠️ With Untrained Model:

```
You say:  "Hello world"
Results:  "xjkl pqwe zmnb" (gibberish - this is NORMAL!)
```

**Why gibberish?** The model has random weights and needs training first!

## 🔄 Workflow

```
1. Deploy
   ↓
2. Open Browser → http://localhost:58080
   ↓
3. Enable Microphone (START button)
   ↓
4. Select Model ("realtime")
   ↓
5. Start Recognition (リアルタイム開始)
   ↓
6. Speak → See Results
   ↓
7. Stop Recognition (リアルタイム停止)
   ↓
8. Review Final Results
```

## 📋 Checklist Before Starting

- [ ] `./run.sh` is running (terminal open)
- [ ] Browser is open at `http://localhost:58080`
- [ ] Microphone is connected and working
- [ ] WebRTC "START" button is clicked
- [ ] Model is selected from dropdown
- [ ] "リアルタイム開始" button is enabled (not grayed out)

## 🆘 Need Help?

1. **Check the debug info** section on the page
2. **Read full guide**: `HOW_TO_USE_REALTIME_WEB_INTERFACE.md`
3. **Check backend logs**:
   ```bash
   ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && \
               sudo docker compose logs -f asr-api"
   ```
4. **Run test script**:
   ```bash
   ./test_realtime_inference.sh
   ```

## 💡 Tips

- **Speak clearly** and at normal pace
- **Minimize background noise**
- **Keep sessions under 1 minute** for best performance
- **Train the model first** for meaningful results
- **Monitor CPU/GPU usage** to avoid bottlenecks

## 🎓 Training the Model

Before realtime inference gives good results, train it:

1. Go to main page (scroll up)
2. Find "学習制御" section
3. Select dataset: `ljspeech`
4. Select model: `realtime`
5. Click "学習開始"
6. Wait for at least a few epochs
7. Try realtime inference again → Better results!

---

**Ready to start?** Follow the 5 steps above! 🚀
