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
sudo docker compose up --build
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
