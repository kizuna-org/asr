# Backend Logging via run.sh

## Overview

Modified `run.sh` to automatically display backend logs when running the
application. This makes it easier to debug and monitor the backend service in
real-time.

## Changes Made

### 1. Added Log Streaming

After containers are started, the script now automatically follows backend logs:

```bash
echo "📋 バックエンドのログを表示します..."
ssh ${SSH_HOST} "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f asr-api" &
LOGS_PID=$!
```

**Key features:**

- Runs in background (`&`) so port forwarding can continue
- Captures process ID (`LOGS_PID`) for cleanup
- Uses `docker compose logs -f` to follow logs in real-time
- Only shows `asr-api` container logs (backend)

### 2. Updated Cleanup Function

Added cleanup for the log streaming process:

```bash
# ログ表示プロセスを終了
if [ ! -z "${LOGS_PID}" ] && kill -0 "${LOGS_PID}" 2>/dev/null; then
    echo "📋 ログ表示プロセスを終了します..."
    kill "${LOGS_PID}" 2>/dev/null || true
fi
```

**Benefits:**

- Properly terminates log streaming when script exits
- No orphaned SSH processes
- Clean shutdown with Ctrl+C

### 3. Updated User Messages

```bash
echo "📋 バックエンドのログは自動的に表示されています。"
```

Informs users that logs are being displayed automatically.

## Usage

### Run the Script

```bash
cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
./run.sh
```

### Expected Output

```
📁 rsyncでファイルをサーバーにコピーします。
...
🚀 コンテナを起動します。
...
📋 バックエンドのログを表示します...

asr-api  | INFO:     Started server process [1]
asr-api  | INFO:     Waiting for application startup.
asr-api  | INFO:     Application startup complete.
asr-api  | INFO:     Uvicorn running on http://0.0.0.0:8000
asr-api  | {"timestamp": "2025-10-02T...", "level": "INFO", ...}

🎉 全てのポート転送が完了しました。
🌐 アプリケーションにアクセスする準備ができました。
📋 バックエンドのログは自動的に表示されています。
💡 このスクリプトを終了する（Ctrl+C）と、ポート転送も自動的に停止します。
```

### What You'll See

**Backend logs will show:**

- FastAPI startup messages
- Uvicorn server info
- API request logs
- WebSocket connections
- Model loading events
- Training progress (if active)
- Error messages and stack traces
- Realtime inference logs

### Stopping the Script

Press `Ctrl+C`:

```
^C🧹 クリーンアップを実行しています...
📋 ログ表示プロセスを終了します...
🔌 SSHマスターセッションを終了します...
✅ クリーンアップが完了しました。
```

## Benefits

### Before

❌ **No automatic log output:**

- Had to manually SSH to server
- Run `docker compose logs` separately
- Difficult to debug in real-time
- Easy to miss errors

### After

✅ **Automatic log streaming:**

- Logs appear immediately after deployment
- Real-time monitoring without extra steps
- Easy to spot errors and warnings
- Better development experience

## Log Format

### Uvicorn Logs

```
INFO:     127.0.0.1:45678 - "GET /status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45679 - "POST /train HTTP/1.1" 200 OK
```

### JSON Structured Logs

```json
{
  "timestamp": "2025-10-02T12:34:56.789Z",
  "level": "INFO",
  "message": "Model loaded successfully",
  "extra_fields": {
    "component": "model_loader",
    "model_name": "conformer"
  }
}
```

### WebSocket Logs

```
INFO:     ('127.0.0.1', 45680) - "WebSocket /realtime" [accepted]
DEBUG:    Received audio chunk: 1024 samples
INFO:     Partial result: "こんにちは"
```

## Troubleshooting

### Issue: Logs Not Appearing

**Check if containers are running:**

```bash
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose ps"
```

**Manually view logs:**

```bash
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose logs asr-api"
```

### Issue: Too Many Logs

**Filter by level (if needed):**

Modify the log command to grep for specific levels:

```bash
docker compose logs -f asr-api | grep "ERROR\|WARNING"
```

### Issue: Log Process Not Stopping

**Manual cleanup:**

```bash
# Find the SSH process
ps aux | grep "docker compose logs"

# Kill it
kill <PID>
```

## Advanced Usage

### View Frontend Logs Too

Modify `run.sh` to show both:

```bash
# Show both backend and frontend logs
ssh ${SSH_HOST} "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f" &
```

### Save Logs to File

```bash
# Redirect logs to file
ssh ${SSH_HOST} "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose logs -f asr-api" > backend.log 2>&1 &
```

### Filter Logs by Keyword

```bash
# Only show realtime-related logs
ssh ${SSH_HOST} "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose logs -f asr-api | grep realtime" &
```

## Integration with Realtime Streaming

### What to Look For

When testing realtime inference, watch for:

**Connection:**

```
INFO:     ('127.0.0.1', 45680) - "WebSocket /realtime" [accepted]
```

**Audio Reception:**

```
DEBUG:    Received audio chunk: 1024 samples, rate: 16000
```

**Inference:**

```
INFO:     Inference completed: duration=0.05s
```

**Results:**

```
INFO:     Partial result: "こんにちは"
INFO:     Final result: "こんにちは世界"
```

**Errors:**

```
ERROR:    WebSocket error: Connection lost
ERROR:    Model inference failed: CUDA out of memory
```

## Tips

### 1. Use Multiple Terminals

**Terminal 1:** Run `./run.sh` (shows logs) **Terminal 2:** Access web interface
(http://localhost:58080) **Terminal 3:** Manual SSH for debugging

### 2. Grep for Specific Issues

```bash
# In another terminal while run.sh is running
ssh edu-gpu "sudo docker compose -f /home/students/r03i/r03i18/asr-test/asr/asr-test/docker-compose.yml -f /home/students/r03i/r03i18/asr-test/asr/asr-test/docker-compose.gpu.yml logs asr-api" | grep "ERROR"
```

### 3. Timestamps

Logs include timestamps for debugging timing issues:

- WebRTC connection timing
- Inference latency
- Queue build-up

### 4. Component Tracking

Logs include component info:

```json
"extra_fields": {
  "component": "realtime_model",
  "action": "inference"
}
```

Use this to filter by component:

```bash
# Show only realtime model logs
... | grep "realtime_model"
```

## Summary

**Modified `run.sh` to automatically stream backend logs for better debugging
and monitoring.**

**Key improvements:**

- ✅ Automatic log display after deployment
- ✅ Real-time monitoring without extra commands
- ✅ Clean shutdown with proper cleanup
- ✅ Better development experience
- ✅ Easier to debug realtime streaming issues

**Usage:**

```bash
./run.sh
# Logs appear automatically
# Press Ctrl+C to stop
```
