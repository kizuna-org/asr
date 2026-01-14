ASR学習POCアプリケーション 基本設計案 (詳細版)
1. 目的
既存の音声認識技術よりもリアルタイムかつ高速に動作するASRモデルの学習・評価サイクルを迅速に回すための、Proof of Concept（概念実証）アプリケーションを構築する。Web GUIを通じて、学習の制御、進捗の可視化、学習済みモデルのテスト、リアルタイム推論を直感的に行えるようにすることを目的とする。

2. 全体構成案
Docker Compose でバックエンド(FastAPI)とフロントエンド(Streamlit)を起動します。HTTP API は `/api` プレフィックス、WebSocket は `/ws` を使用します。ホスト公開ポートは `58081`(API) と `58080`(Frontend)。

**主要機能:**
- 学習制御: 新規学習、チェックポイントからの再開、学習停止（conformer/realtimeモデル対応）
- 推論機能: ファイルアップロードによる推論、リアルタイム音声推論（WebRTC統合）
- モデル管理: 学習済みモデルの一覧表示、削除、フィルタリング機能
- チェックポイント管理: チェックポイントの一覧表示、学習再開、詳細情報表示
- データセット管理: データセットのダウンロード、自動展開
- リアルタイム通信: WebSocketによる学習進捗のリアルタイム更新、音声ストリーミング
- 詳細なログ機能: 構造化ログ、エラーハンドリング、プロキシ対応

3. 各コンポーネント詳細
3.1. バックエンド (GPUサーバー上のDockerコンテナ)
使用技術
言語: Python 3.9+

Webフレームワーク: FastAPI

Webサーバー: Uvicorn

リアルタイム通信: fastapi.WebSocket

機械学習: PyTorch, torchaudio

ユーティリティ:

Hugging Face datasets (データセットのロードと前処理用)

Hugging Face transformers (トークナイザやビルディングブロックとして利用)

librosa (音声ファイルの前処理用)

PyYAML (設定ファイルの読み込み用)

詳細な内部構造
APIリクエストフロー (POST /api/train/start):

frontendからHTTPリクエストを`api.py`のエンドポイントが受信します（`/api` プレフィックス）。

FastAPIのBackgroundTasksを使い、学習プロセスを非同期のバックグラウンドタスクとしてtrainer.py内のstart_training関数に渡します。これにより、APIはすぐにレスポンスを返し、UIが固まるのを防ぎます。

trainer.pyは、リクエストで指定されたモデル名やデータセット名に基づき、config.yamlから詳細な設定（ハイパーパラメータなど）を読み込みます。

動的モジュール読み込み:

trainer.pyは、importlibモジュールを使い、app/models/{model_name}.pyから対応するモデルクラスを、app/datasets/{dataset_name}.pyからデータセットクラスを動的にインポートしてインスタンス化します。

学習プロセス (trainer.py):

学習ループを開始します。ループ内では、データローダーからバッチを取得し、モデルに渡して損失を計算し、逆伝播を行います。

一定ステップごとに、WebSocketマネージャー(websocket.py)を通じて現在の進捗状況（エポック、ロス値、学習率など）をfrontendにブロードキャストします。

数エポックごとに、モデルの状態辞書とオプティマイザの状態辞書をチェックポイントファイルとして./checkpointsディレクトリに保存します。

WebSocket通信 (websocket.py):

frontendからのWebSocket接続を待ち受けるエンドポイント/wsを定義します。

接続されたクライアントのリストを管理し、trainer.pyから進捗データを受け取るたびに、接続中の全クライアントにJSON形式でデータを送信します。

インターフェース定義
1. モデルインターフェース (backend/app/models/interface.py)

すべてのモデルがこのインターフェースを実装します。これによりtrainer.pyはモデルの具体的な実装を意識することなく、統一された方法で学習と推論を実行できます。

# backend/app/models/interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn

class BaseASRModel(nn.Module, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        学習・検証時のフォワードパス。
        Args:
            batch: データローダーからの出力 (例: {'waveforms': ..., 'tokens': ...})
        Returns:
            loss: 計算された損失 (スカラーテンソル)
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def inference(self, waveform: torch.Tensor) -> str:
        """
        推論処理。音声波形からテキストを生成する。
        Args:
            waveform: 単一の音声波形テンソル
        Returns:
            transcription: 文字起こし結果の文字列
        """
        pass

    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer, epoch: int):
        """チェックポイントを保存する"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None):
        """チェックポイントを読み込む"""
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)


2. データセットインターフェース (backend/app/datasets/interface.py)

torch.utils.data.Datasetを継承します。共通の前処理などをここに実装できます。

# backend/app/datasets/interface.py
from abc import ABC, abstractmethod
from typing import Dict
from torch.utils.data import Dataset

class BaseASRDataset(Dataset, ABC):
    def __init__(self, config: Dict, split: str = 'train'):
        self.config = config
        self.split = split
        self.data = self._load_data()

    @abstractmethod
    def _load_data(self):
        """データセットのメタデータを読み込む処理"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """
        音声波形とテキストのペアを辞書形式で返す
        (例: {'waveform': tensor, 'text': "hello world"})
        """
        pass

# さらに、Dataloaderに渡すための Collate Function も共通化すると便利
def collate_fn(batch):
    # バッチ内の音声波形をパディングし、テキストをトークンIDに変換するなどの処理
    pass

3.2. フロントエンド (Web GUI)

**主要機能:**
- メインダッシュボード: 学習制御、推論テスト、リアルタイム推論、進捗表示
- モデル管理: 学習済みモデルの一覧表示と削除、フィルタリング機能
- チェックポイント管理: チェックポイントの一覧表示と学習再開、詳細情報表示
- リアルタイム推論: マイク入力によるリアルタイム音声認識（WebRTC統合）
- データセット管理: データセットのダウンロード、自動展開
- 詳細なログ機能: 構造化ログ、エラーハンドリング、プロキシ対応

3.2.1. フロントエンド 詳細な内部構造

**状態管理 (st.session_state):**

Streamlitのセッション状態機能を利用して、UIの状態を保持します。

- `st.session_state.is_training`: 学習中かどうかのフラグ
- `st.session_state.logs`: バックエンドから受信したログメッセージのリスト
- `st.session_state.progress_df`: グラフ描画用のロス値などの時系列データフレーム
- `st.session_state.validation_df`: 検証ロスデータ
- `st.session_state.lr_df`: 学習率データ
- `st.session_state.current_page`: 現在のページ（main, model_management, checkpoint_management）
- `st.session_state.realtime_running`: リアルタイム推論の実行状態
- `st.session_state.realtime_partial`: リアルタイム推論の部分結果
- `st.session_state.realtime_final`: リアルタイム推論の最終結果
- `st.session_state.realtime_status`: リアルタイム推論のステータス情報
- `st.session_state.realtime_error`: リアルタイム推論のエラーメッセージ
- `st.session_state.realtime_msg_queue`: リアルタイム推論のメッセージキュー

**バックエンドとの通信 (app.py):**

- **制御系 (HTTP)**: 「学習開始」「停止」ボタンが押されると、requestsライブラリを使ってFastAPIの`/train/start`, `/train/stop`エンドポイントにPOSTリクエストを送信します。
- **進捗受信用 (WebSocket)**: 学習開始リクエストが成功した後、websocketsライブラリを使ってバックエンドの`/ws`エンドポイントに接続します。非同期処理を用いてWebSocketメッセージを継続的に待ち受け、受信したデータをst.session_stateに格納し、st.rerun()を呼び出して画面を再描画します。
- **リアルタイム推論**: WebRTCとWebSocketを組み合わせて、マイクからの音声をリアルタイムでサーバーに送信し、部分的な文字起こし結果を受信します。

**ログ機能:**
- 構造化ログ: JSON形式でのログ出力
- 詳細なエラーハンドリング: 接続エラー、タイムアウト、HTTPエラーの詳細表示
- プロキシ対応: HTTP_PROXY、HTTPS_PROXY、NO_PROXY環境変数のサポート

4. 開発・デプロイ環境
(変更なし)

5. ディレクトリ構成案
asr-poc/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPIアプリのエントリーポイント
│   │   ├── api.py               # HTTPエンドポイント定義
│   │   ├── websocket.py         # WebSocket関連
│   │   ├── trainer.py           # 学習プロセスの管理
│   │   ├── config_loader.py     # config.yamlの読み込みと検証
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── interface.py     # BaseASRDataset
│   │   │   └── ljspeech.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── interface.py     # BaseASRModel
│   │       └── conformer.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── config.yaml
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   └── ljspeech/
├── checkpoints/
└── docker-compose.yml

6. config.yaml 設定ファイル例
# backend/config.yaml
# 使用可能なモデルとデータセットを定義
available_models:
  - conformer
  - realtime
available_datasets:
  - ljspeech

# モデルごとの詳細設定
models:
  conformer:
    input_dim: 80 # メルスペクトログラムの次元数
    encoder_dim: 256
    num_heads: 4
    huggingface_model_name: "facebook/wav2vec2-base-960h"
    tokenizer:
      type: "SentencePiece"
      vocab_size: 5000

  realtime:
    encoder:
      input_dim: 80
      hidden_dim: 256
      num_layers: 3
      rnn_type: "GRU"
      dropout: 0.1
    decoder:
      input_dim: 256
      vocab_size: 1000
      blank_token: "_"
    processing:
      chunk_size_ms: 100
      sample_rate: 16000
      feature_type: "mel_spectrogram"
      n_mels: 80
      n_fft: 1024
      hop_length: 160
    optimization:
      precision: "fp16"
      batch_size: 1
      max_memory_mb: 512

# データセットごとの設定
datasets:
  ljspeech:
    path: "/app/data/ljspeech" # コンテナ内のパス（composeで ./data をマウント）
    sample_rate: 22050
    n_fft: 1024
    n_mels: 80
    text_cleaners: ['english_cleaners']

# 学習のグローバル設定
training:
  optimizer: "AdamW"
  learning_rate: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.98]
  eps: 1.0e-9
  scheduler: "WarmupLR"
  warmup_steps: 100
  batch_size: 32
  num_epochs: 100
  grad_clip_thresh: 1.0
  log_interval: 10
  checkpoint_interval: 1
  auto_resume: true
  checkpoint_retention: 5

7. 関連ドキュメント

- [リアルタイム性特化型音声認識モデル設計](./realtime_model_design.md): 精度を度外視し、リアルタイム性を最優先とした音声認識モデルの詳細設計仕様

8. 実装状況

## 8.1. 完了済み機能

**バックエンド実装:**
- FastAPI による REST API 実装（学習制御、推論、モデル管理）
- WebSocket によるリアルタイム通信
- 構造化ログ機能（JSON形式、詳細なエラーハンドリング）
- モデル管理機能（一覧、削除、フィルタリング）
- チェックポイント管理機能（一覧、詳細情報、学習再開）
- データセットダウンロード機能（自動展開）
- リアルタイム推論機能（WebSocket経由、conformer/realtimeモデル対応）
- 学習再開機能（チェックポイントからの再開、特定チェックポイント指定）
- 推論パフォーマンス計測機能（3種類の時間計測）
- テストエンドポイント（疎通確認）
- 詳細なエラーハンドリング（スタックトレース含む）
- モデルキャッシュ機能（推論性能向上）

**フロントエンド実装:**
- Streamlit による Web GUI
- 学習制御（新規学習、再開、停止、conformer/realtimeモデル対応）
- 推論テスト（ファイルアップロード、パフォーマンス情報表示）
- リアルタイム推論（WebRTC統合、conformer/realtimeモデル対応）
- モデル管理（一覧表示、削除、フィルタリング機能）
- チェックポイント管理（一覧表示、学習再開、詳細情報表示）
- データセット管理（ダウンロード、自動展開）
- 詳細なログ機能（構造化ログ、エラーハンドリング）
- プロキシ対応（HTTP_PROXY、HTTPS_PROXY、NO_PROXY）
- ページナビゲーション機能（メインダッシュボード、モデル管理、チェックポイント管理）
- 進捗表示（学習ロス、学習率のグラフ、リアルタイム更新）
- WebRTC統合によるリアルタイム音声認識
- 音声フレーム処理とWebSocket通信の非同期処理

**インフラ・デプロイ:**
- Docker Compose によるコンテナ化
- GPU 対応設定（NVIDIA Container Runtime）
- ローカル実行スクリプト（run-local.sh）
- ポートフォワーディング（SSH ControlMaster）
- 構造化ログのファイル出力機能

## 8.2. 次のステップ

**機能拡張:**
- バッチ推論機能の実装
- モデル比較機能の実装
- 学習履歴管理機能の実装
- 設定管理機能の実装
- リアルタイム推論の性能最適化
- 音声品質の自動評価機能

**UI/UX改善:**
- ダークモード対応
- レスポンシブデザイン対応
- アクセシビリティ向上
- 国際化対応
- リアルタイム推論のUI改善
- 音声波形の可視化機能

**技術的改善:**
- モデル推論の最適化
- WebSocket通信の安定性向上
- エラーハンドリングの強化
- ログ機能の拡張
