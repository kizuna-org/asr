# リアルタイム音声認識モデル学習システム

Dockerで動作する軽量な音声認識モデルの学習システムです。

## 特徴

- 🚀 光速でほぼリアルタイム動作
- 🐳 Dockerコンテナ化済み
- 🎯 軽量なCTCベースのモデル
- 📊 リアルタイム学習進捗表示
- 🎤 マイク入力でのリアルタイム推論
- 🎮 **CUDA対応でGPU高速化**

## システム要件

### GPU要件（推奨）
- NVIDIA GPU with CUDA support
- CUDA 12.3.2
- cuDNN 9
- Docker with NVIDIA Container Runtime

### CPU要件（最小）
- 4GB RAM
- 2 CPU cores

## アーキテクチャ

- **フロントエンド**: Streamlit (リアルタイムUI)
- **バックエンド**: FastAPI (推論API)
- **モデル**: 軽量CNN + LSTM + CTC
- **音声処理**: librosa + torchaudio
- **学習**: PyTorch with CUDA support

## セットアップ

### GPU環境での実行（推奨）

```bash
# プロジェクトのクローン
git clone <repository-url>
cd asr-test

# NVIDIA Container Runtimeの確認
nvidia-smi

# Docker Composeで起動（GPU対応）
sudo docker compose up --build
```

### CPU環境での実行

```bash
# CPU版のDockerfileを使用
sudo docker build -f Dockerfile.cpu -t asr-app-cpu .
sudo docker run -p 58080:8000 -p 58081:8501 asr-app-cpu
```

## GPU環境の確認

ビルド時に自動的にGPU環境がチェックされます：

```bash
# 手動でGPU環境をチェック
sudo docker run --gpus all asr-app python gpu_check.py
```

## 使用方法

### リアルタイム音声認識（推奨）
1. ブラウザで `http://localhost:58080/static/index.html` にアクセス
2. 「接続開始」ボタンをクリック
3. 「録音開始」ボタンをクリックしてマイクアクセスを許可
4. 音声を話すとリアルタイムで認識結果が表示されます

### モデル学習（Streamlit）
1. ブラウザで `http://localhost:58081` にアクセス
2. 音声データをアップロードまたはマイクで録音
3. モデルの学習を開始
4. リアルタイムで推論結果を確認

### API使用
- API ドキュメント: `http://localhost:58080/docs`
- ヘルスチェック: `http://localhost:58080/health`
- モデル情報: `http://localhost:58080/model_info`

## ディレクトリ構造

```
asr-test/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── gpu_check.py          # GPU環境チェックスクリプト
├── app/
│   ├── main.py
│   ├── model.py
│   ├── dataset.py
│   ├── trainer.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
└── models/
```

## トラブルシューティング

### GPU関連の問題

1. **nvidia-smiが動作しない**
   ```bash
   # NVIDIAドライバーの確認
   nvidia-smi
   ```

2. **DockerでGPUが認識されない**
   ```bash
   # NVIDIA Container Runtimeの確認
   docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
   ```

3. **CUDA out of memory**
   - バッチサイズを小さくする
   - モデルサイズを調整する

## ライセンス

MIT License
