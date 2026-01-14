# フロントエンド設計書

このドキュメントは、ASR学習POCアプリケーションのフロントエンド（Streamlit）の設計と実装について記述します。

## 1. 概要

フロントエンドはStreamlitを使用して構築されたWeb GUIで、以下の主要機能を提供します：

- **メインダッシュボード**: 学習制御、推論テスト、リアルタイム推論、進捗表示
- **モデル管理**: 学習済みモデルの一覧表示と削除、フィルタリング機能
- **チェックポイント管理**: チェックポイントの一覧表示と学習再開、詳細情報表示
- **リアルタイム推論**: マイク入力によるリアルタイム音声認識（WebRTC統合、conformer/realtimeモデル対応）
- **データセット管理**: データセットのダウンロード、自動展開
- **詳細なログ機能**: 構造化ログ、エラーハンドリング、プロキシ対応

## 2. アーキテクチャ

### 2.1. 技術スタック

- **フレームワーク**: Streamlit
- **通信**: HTTP API (requests), WebSocket (websockets)
- **リアルタイム音声**: WebRTC (streamlit-webrtc)
- **ログ**: 構造化ログ (JSON形式)
- **プロキシ対応**: HTTP_PROXY, HTTPS_PROXY, NO_PROXY
- **データ処理**: pandas, numpy
- **非同期処理**: asyncio, threading
- **音声処理**: torchaudio (バックエンド経由)

### 2.2. ディレクトリ構成

```
frontend/
├── app.py                 # メインアプリケーション
├── Dockerfile            # Docker設定
├── requirements.txt      # 依存関係
└── pages/               # ページ別の実装（将来の拡張用）
    └── __init__.py
```

## 3. 状態管理

### 3.1. セッション状態 (st.session_state)

```python
# 基本状態
"is_training": bool                    # 学習中フラグ
"logs": List[str]                     # ログメッセージ
"available_models": List[str]         # 利用可能なモデル
"available_datasets": List[str]       # 利用可能なデータセット

# 進捗データ
"progress_df": DataFrame              # 学習ロスデータ
"validation_df": DataFrame            # 検証ロスデータ
"lr_df": DataFrame                    # 学習率データ
"current_progress": float             # 現在の進捗率
"current_epoch": int                  # 現在のエポック
"current_step": int                   # 現在のステップ
"total_epochs": int                   # 総エポック数
"total_steps": int                    # 総ステップ数

# ページ管理
"current_page": str                   # 現在のページ
"initial_load": bool                  # 初期化完了フラグ

# リアルタイム推論
"realtime_running": bool              # リアルタイム推論実行中
"realtime_partial": str               # 部分結果
"realtime_final": str                 # 最終結果
"realtime_status": dict               # ステータス情報
"realtime_error": str                 # エラーメッセージ
"realtime_msg_queue": Queue           # メッセージキュー
```

### 3.2. 状態の初期化

```python
def init_session_state():
    defaults = {
        "logs": ["ダッシュボードへようこそ！"],
        "progress_df": pd.DataFrame(columns=["epoch", "step", "loss"]),
        "validation_df": pd.DataFrame(columns=["epoch", "val_loss"]),
        "lr_df": pd.DataFrame(columns=["step", "learning_rate"]),
        "is_training": False,
        "available_models": [],
        "available_datasets": [],
        # ... その他のデフォルト値
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

## 4. バックエンド通信

### 4.1. HTTP API通信

```python
# 設定取得
def get_config():
    response = requests.get(f"{BACKEND_URL}/config", timeout=10)
    if response.status_code == 200:
        config = response.json()
        st.session_state.available_models = config.get("available_models", [])
        st.session_state.available_datasets = config.get("available_datasets", [])

# 学習開始
def start_training(model_name, dataset_name, epochs, batch_size, **kwargs):
    params = {
        "model_name": model_name,  # "conformer" または "realtime"
        "dataset_name": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        **kwargs
    }
    response = requests.post(f"{BACKEND_URL}/train/start", json=params, timeout=30)
    return response.status_code == 200
```

### 4.2. WebSocket通信

```python
# リアルタイム推論用WebSocket
async def stream_audio_to_ws(q, model_name, sample_rate, running_flag_ref, msg_queue_ref):
    async with websockets.connect(WEBSOCKET_URL) as ws:
        # 開始メッセージ
        start_msg = {"type": "start", "model_name": model_name, "sample_rate": sample_rate, "format": "f32"}  # model_name: "conformer" または "realtime"
        await ws.send(json.dumps(start_msg))
        
        # 音声フレーム送信
        while running_flag_ref:
            try:
                chunk = q.get(timeout=0.1)
                await ws.send(chunk)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
```

### 4.3. エラーハンドリング

```python
def log_detailed_error(operation: str, error: Exception, response=None):
    """詳細なエラー情報をログに記録"""
    error_msg = f"❌ {operation} エラー:"
    error_msg += f"\n   - エラータイプ: {type(error).__name__}"
    error_msg += f"\n   - エラーメッセージ: {str(error)}"
    
    if response is not None:
        error_msg += f"\n   - ステータスコード: {response.status_code}"
        error_msg += f"\n   - レスポンスヘッダー: {dict(response.headers)}"
    
    st.session_state.logs.append(error_msg)
```

## 5. ページ構成

### 5.1. メインダッシュボード

- **学習制御**: モデル選択、データセット選択、学習パラメータ設定、チェックポイントからの再開
- **推論テスト**: 音声ファイルアップロード、推論実行、結果表示、パフォーマンス情報表示（3種類の時間計測）
- **リアルタイム推論**: マイク入力、リアルタイム音声認識（WebRTC統合、conformer/realtimeモデル対応）
- **進捗表示**: 学習ロス、学習率のグラフ表示、リアルタイム更新
- **ログ表示**: システムログの表示、詳細エラー情報、構造化ログ

### 5.2. モデル管理

- **モデル一覧**: 学習済みモデルの一覧表示、フィルタリング機能
- **モデル詳細**: サイズ、作成日時、ファイル構成、エポック情報
- **モデル削除**: 確認付きのモデル削除機能、安全な削除プロセス
- **ページナビゲーション**: サイドバーからのページ切り替え

### 5.3. チェックポイント管理

- **チェックポイント一覧**: フィルタリング機能付きの一覧表示、詳細情報
- **学習再開**: チェックポイントからの学習再開、パラメータ設定
- **詳細情報**: エポック、サイズ、作成日時、ファイル構成の表示
- **ページナビゲーション**: サイドバーからのページ切り替え

### 5.4. ページナビゲーション

```python
# ページ間のナビゲーション
current_page = st.session_state.get("current_page", "main")
if st.button("🏠 メインダッシュボード", use_container_width=True, disabled=(current_page == "main")):
    st.session_state.current_page = "main"
    st.rerun()
if st.button("🤖 モデル管理", use_container_width=True, disabled=(current_page == "model_management")):
    st.session_state.current_page = "model_management"
    st.rerun()
if st.button("📂 チェックポイント管理", use_container_width=True, disabled=(current_page == "checkpoint_management")):
    st.session_state.current_page = "checkpoint_management"
    st.rerun()
```

## 6. リアルタイム推論

### 6.1. WebRTC統合

```python
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

class MicAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frame_queue = None
        self.logger = logging.getLogger("ui-rt")
    
    def recv_audio(self, frames, **kwargs):
        # 音声フレームの処理
        for frame in frames:
            pcm = frame.to_ndarray(format="flt")
            # モノラル化
            if pcm.ndim == 2 and pcm.shape[0] > 1:
                pcm_mono = pcm.mean(axis=0)
            else:
                pcm_mono = pcm[0] if pcm.ndim == 2 else pcm
            
            # フレームキューに送信
            if self.frame_queue:
                self.frame_queue.put(pcm_mono.astype(np.float32).tobytes())
        
        return frames
```

### 6.2. 音声ストリーミング

```python
# WebRTCストリーマーの初期化
rtc_ctx = webrtc_streamer(
    key="asr-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=2048,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# 音声フレームの取得とWebSocket送信
def pull_audio_frames():
    while running_flag:
        if rtc_ctx.audio_receiver:
            frames = rtc_ctx.audio_receiver.get_frames()
            for frame in frames:
                # 音声フレームの処理とWebSocket送信
                process_and_send_audio_frame(frame)
```

### 6.3. リアルタイム推論の状態管理

```python
# リアルタイム推論用の状態管理
st.session_state.setdefault("realtime_running", False)
st.session_state.setdefault("realtime_partial", "")
st.session_state.setdefault("realtime_final", "")
st.session_state.setdefault("realtime_status", {})
st.session_state.setdefault("realtime_error", "")
st.session_state.setdefault("realtime_msg_queue", queue.Queue())
st.session_state.setdefault("realtime_stats", {})
st.session_state.setdefault("_rt_chunks_sent", 0)
```

### 6.4. 音声フレーム処理

```python
class MicAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frame_queue = None
        self.logger = logging.getLogger("ui-rt")
        self.msg_queue = None
        self._frames_sent = 0

    def recv_audio(self, frames, **kwargs):
        if self.frame_queue is None:
            return frames
            
        for frame in frames:
            # 32-bit float PCM, shape: (channels, samples)
            pcm = frame.to_ndarray(format="flt")
            
            # モノラル化
            if pcm.ndim == 2 and pcm.shape[0] > 1:
                pcm_mono = pcm.mean(axis=0)
            else:
                pcm_mono = pcm[0] if pcm.ndim == 2 else pcm
            
            # 送信は float32 little-endian bytes
            pcm_f32 = pcm_mono.astype(np.float32)
            
            try:
                self.frame_queue.put(pcm_f32.tobytes(), timeout=0.1)
                self._frames_sent += 1
            except queue.Full:
                self.logger.warning("Frame queue is full, dropping audio chunk")
        return frames
```

## 7. ログ機能

### 7.1. 構造化ログ

```python
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        return json.dumps(log_entry, ensure_ascii=False)
```

### 7.2. ログ表示

```python
# ログ表示エリア
st.header("ログ")
log_container = st.container()
with log_container:
    for log in st.session_state.logs[-50:]:  # 最新50件を表示
        st.text(log)
```

## 8. プロキシ対応

### 8.1. プロキシ設定

```python
# 環境変数からプロキシ設定を取得
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")
NO_PROXY = os.getenv("NO_PROXY", "localhost,127.0.0.1,asr-api")

# プロキシ設定を辞書形式で準備
proxies = {}
if HTTP_PROXY:
    proxies["http"] = HTTP_PROXY
if HTTPS_PROXY:
    proxies["https"] = HTTPS_PROXY
```

### 8.2. プロキシ判定

```python
def should_use_proxy(url):
    """URLがプロキシを使用すべきかどうかを判定"""
    if not proxies:
        return False
    
    no_proxy_hosts = [host.strip() for host in NO_PROXY.split(",")]
    for host in no_proxy_hosts:
        if host in url:
            return False
    return True
```

## 9. パフォーマンス最適化

### 9.1. 進捗更新の最適化

```python
# 進捗更新の頻度を制限（1秒ごと）
current_time = time.time()
if current_time - st.session_state.last_progress_update >= 1:
    progress_updated = update_progress_from_backend()
    st.session_state.last_progress_update = current_time
```

### 9.2. 連続エラー処理

```python
# 連続エラーが多すぎる場合は進捗更新をスキップ
if st.session_state.consecutive_errors >= st.session_state.max_consecutive_errors:
    st.session_state.logs.append("⚠️ 連続エラーが多すぎるため、進捗更新を一時停止します")
    return False
```

## 10. セキュリティ考慮事項

### 10.1. 入力検証

```python
# ファイルアップロード時の検証
uploaded = st.file_uploader(
    "音声ファイルを選択 (WAV/FLAC/MP3/M4A/OGG)", 
    type=["wav", "flac", "mp3", "m4a", "ogg"],
    help="推論対象の音声ファイルをアップロードしてください"
)
```

### 10.2. エラーハンドリング

```python
try:
    # API呼び出し
    response = requests.post(f"{BACKEND_URL}/train/start", json=params, timeout=30)
    response.raise_for_status()
except requests.exceptions.ConnectionError as e:
    log_detailed_error("学習開始", e)
except requests.exceptions.Timeout as e:
    log_detailed_error("学習開始", e)
except Exception as e:
    log_detailed_error("学習開始", e)
```

## 11. 実装済み機能

### 11.1. 主要機能

- **学習制御**: 新規学習、チェックポイントからの再開、学習停止（conformer/realtimeモデル対応）
- **推論機能**: ファイルアップロードによる推論、リアルタイム音声推論（WebRTC統合）
- **モデル管理**: 学習済みモデルの一覧表示、削除、フィルタリング機能
- **チェックポイント管理**: チェックポイントの一覧表示、学習再開、詳細情報表示
- **データセット管理**: データセットのダウンロード、自動展開
- **リアルタイム通信**: WebSocketによる学習進捗のリアルタイム更新、音声ストリーミング
- **ページナビゲーション**: メインダッシュボード、モデル管理、チェックポイント管理の分離
- **詳細なログ機能**: 構造化ログ、エラーハンドリング、プロキシ対応
- **パフォーマンス計測**: 推論時間の詳細計測（3種類の時間計測）
- **リアルタイム推論**: WebRTC統合によるマイク入力からのリアルタイム音声認識

### 11.2. 技術的特徴

- **WebRTC統合**: マイク入力によるリアルタイム音声認識（conformer/realtimeモデル対応）
- **構造化ログ**: JSON形式でのログ出力とエラーハンドリング
- **プロキシ対応**: HTTP_PROXY、HTTPS_PROXY、NO_PROXY環境変数のサポート
- **非同期処理**: WebSocket通信と音声ストリーミングの非同期処理
- **状態管理**: Streamlitのセッション状態を活用した複雑な状態管理
- **ページ管理**: 複数ページ間の状態保持とナビゲーション
- **パフォーマンス計測**: 推論時間の詳細計測（3種類の時間計測）

## 12. 今後の拡張予定

### 12.1. 機能拡張

- **バッチ推論**: 複数ファイルの一括推論
- **モデル比較**: 複数モデルの性能比較
- **学習履歴**: 過去の学習結果の管理
- **設定管理**: ユーザー設定の保存・復元

### 12.2. UI/UX改善

- **ダークモード**: テーマ切り替え機能
- **レスポンシブデザイン**: モバイル対応
- **アクセシビリティ**: キーボード操作対応
- **国際化**: 多言語対応

## 13. トラブルシューティング

### 13.1. よくある問題

1. **WebRTC接続エラー**: ブラウザの権限設定を確認
2. **WebSocket接続エラー**: ファイアウォール設定を確認
3. **プロキシエラー**: NO_PROXY設定を確認
4. **メモリ不足**: バッチサイズを調整
5. **音声フレーム取得エラー**: マイクの権限とWebRTC設定を確認

### 13.2. デバッグ情報

```python
# デバッグ情報の表示
st.write("**デバッグ情報:**")
st.write(f"- start_btn: {start_btn}")
st.write(f"- rtc_ctx: {rtc_ctx is not None}")
st.write(f"- rtc_ctx.state.playing: {rtc_ctx.state.playing if rtc_ctx else 'N/A'}")
st.write(f"- rtc_ctx.audio_receiver: {rtc_ctx.audio_receiver is not None if rtc_ctx else 'N/A'}")
st.write(f"- realtime_running: {st.session_state.get('realtime_running', False)}")
```

### 13.3. ログレベル調整

```python
# ログレベルの調整
logging.getLogger("ui-rt").setLevel(logging.INFO)
logging.getLogger("audio_puller").setLevel(logging.INFO)
logging.getLogger("websocket_loop").setLevel(logging.INFO)
logging.getLogger("websocket_sender").setLevel(logging.INFO)
```
