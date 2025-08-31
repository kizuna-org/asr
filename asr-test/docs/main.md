ASR学習POCアプリケーション 基本設計案 (詳細版)
1. 目的
既存の音声認識技術よりもリアルタイムかつ高速に動作するASRモデルの学習・評価サイクルを迅速に回すための、Proof of Concept（概念実証）アプリケーションを構築する。Web GUIを通じて、学習の制御、進捗の可視化、および学習済みモデルのテストを直感的に行えるようにすることを目的とする。

2. 全体構成案
(変更なし)

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
APIリクエストフロー (POST /train/start):

frontendからHTTPリクエストをapi.pyのエンドポイントが受信します。

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
(変更なし)

3.2.1. フロントエンド 詳細な内部構造
状態管理 (st.session_state):

Streamlitのセッション状態機能を利用して、UIの状態を保持します。

st.session_state.is_training: 学習中かどうかのフラグ

st.session_state.logs: バックエンドから受信したログメッセージのリスト

st.session_state.progress_data: グラフ描画用のロス値などの時系列データフレーム

バックエンドとの通信 (app.py):

制御系 (HTTP): 「学習開始」「停止」ボタンが押されると、requestsライブラリを使ってFastAPIの/train/start, /train/stopエンドポイントにPOSTリクエストを送信します。

進捗受信用 (WebSocket): 学習開始リクエストが成功した後、websocketsライブラリを使ってバックエンドの/wsエンドポイントに接続します。非同期処理を用いてWebSocketメッセージを継続的に待ち受け、受信したデータをst.session_stateに格納し、st.experimental_rerun()を呼び出して画面を再描画します。

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
  - rnn-t # (例)
available_datasets:
  - ljspeech

# モデルごとの詳細設定
models:
  conformer:
    input_dim: 80 # メルスペクトログラムの次元数
    encoder_dim: 256
    num_heads: 4
    # ... その他のハイパーパラメータ

# データセットごとの設定
datasets:
  ljspeech:
    path: "/data/ljspeech" # コンテナ内のパス
    sample_rate: 22050
    n_fft: 1024
    n_mels: 80

# 学習のグローバル設定
training:
  learning_rate: 0.001
  batch_size: 32
  optimizer: "Adam"

7. 次のステップ
環境構築: docker-compose.yml と各Dockerfileを記述し、コンテナを起動できる状態にする。

最小限の実装:

backend: conformer.pyとljspeech.pyの骨格をインターフェースに従って実装する。/inference APIがダミーデータで動作するようにする。

frontend: 音声ファイルをアップロードし、/inference APIを呼び出して結果を表示するUI部分を実装する。

学習機能の実装: trainer.pyに学習ループを実装し、フロントエンドの学習開始ボタンと連携させる。

リアルタイム更新の実装: WebSocketによる進捗通知機能を実装し、フロントエンドのグラフやステータスがリアルタイムに更新されるようにする。
