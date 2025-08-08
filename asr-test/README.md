# リアルタイム音声認識モデル学習システム

Dockerで動作する軽量な音声認識モデルの学習システムです。

## 特徴

- 🚀 光速でほぼリアルタイム動作
- 🐳 Dockerコンテナ化済み
- 🎯 軽量なCTCベースのモデル
- 📊 リアルタイム学習進捗表示
- 🎤 マイク入力でのリアルタイム推論

## アーキテクチャ

- **フロントエンド**: Streamlit (リアルタイムUI)
- **バックエンド**: FastAPI (推論API)
- **モデル**: 軽量CNN + LSTM + CTC
- **音声処理**: librosa + torchaudio
- **学習**: PyTorch

## セットアップ

```bash
# プロジェクトのクローン
git clone <repository-url>
cd asr-test

# Docker Composeで起動
docker-compose up --build
```

## 使用方法

1. ブラウザで `http://localhost:8501` にアクセス
2. 音声データをアップロードまたはマイクで録音
3. モデルの学習を開始
4. リアルタイムで推論結果を確認

## ディレクトリ構造

```
asr-test/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
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

## ライセンス

MIT License
