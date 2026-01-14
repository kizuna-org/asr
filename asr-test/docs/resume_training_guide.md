# 学習再開機能ガイド

このドキュメントでは、ASR学習システムの学習再開機能の使用方法について説明します。

## 概要

学習再開機能により、以下のことが可能になります：

1. **自動再開**: 学習開始時に自動的に最新のチェックポイントから再開
2. **特定チェックポイントからの再開**: 指定したチェックポイントから学習を再開
3. **チェックポイント管理**: 利用可能なチェックポイントの一覧表示と管理
4. **自動クリーンアップ**: 古いチェックポイントの自動削除

## 機能詳細

### 1. 自動再開機能

学習開始時に、システムは自動的に最新のチェックポイントを検索し、存在する場合はそこから学習を再開します。

```bash
# 最新のチェックポイントから自動再開
curl -X POST "http://localhost:8000/api/train/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "conformer",
    "dataset_name": "ljspeech",
    "epochs": 10,
    "resume_from_checkpoint": true
  }'
```

### 2. 特定チェックポイントからの再開

特定のチェックポイントから学習を再開したい場合は、`specific_checkpoint`パラメータを使用します。

```bash
# 特定のチェックポイントから再開
curl -X POST "http://localhost:8000/api/train/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "conformer",
    "dataset_name": "ljspeech",
    "epochs": 10,
    "resume_from_checkpoint": true,
    "specific_checkpoint": "conformer-ljspeech-epoch-5.pt"
  }'
```

### 3. 学習再開専用API

学習再開専用のAPIエンドポイントも提供されています。

```bash
# 学習再開専用API
curl -X POST "http://localhost:8000/api/train/resume" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "conformer",
    "dataset_name": "ljspeech",
    "specific_checkpoint": "conformer-ljspeech-epoch-5.pt",
    "epochs": 10
  }'
```

### 4. チェックポイント一覧の取得

利用可能なチェックポイントの一覧を取得できます。

```bash
# 全てのチェックポイントを取得
curl "http://localhost:8000/api/checkpoints"

# 特定のモデルのチェックポイントを取得
curl "http://localhost:8000/api/checkpoints?model_name=conformer&dataset_name=ljspeech"
```

## 設定オプション

### config.yaml での設定

```yaml
training:
  # 学習再開設定
  auto_resume: true  # 学習開始時に自動的に最新のチェックポイントから再開するかどうか
  checkpoint_retention: 5  # 保持するチェックポイントの数（古いものは自動削除）
```

### API パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|----|-----------|----|
| `resume_from_checkpoint` | boolean | true | チェックポイントから学習を再開するかどうか |
| `specific_checkpoint` | string | null | 特定のチェックポイント名（省略時は最新を使用） |
| `epochs` | integer | 設定ファイルの値 | 学習エポック数 |
| `batch_size` | integer | 設定ファイルの値 | バッチサイズ |
| `lightweight` | boolean | false | 軽量実行モード（サンプル数を制限） |
| `limit_samples` | integer | null | 学習に使用するサンプル数の上限 |

## 使用例

### 例1: 基本的な学習再開

```python
import requests

# 学習を開始（自動的に最新のチェックポイントから再開）
response = requests.post("http://localhost:8000/api/train/start", json={
    "model_name": "conformer",
    "dataset_name": "ljspeech",
    "epochs": 20
})

print(response.json())
```

### 例2: 特定のエポックから再開

```python
import requests

# エポック5から学習を再開
response = requests.post("http://localhost:8000/api/train/resume", json={
    "model_name": "conformer",
    "dataset_name": "ljspeech",
    "specific_checkpoint": "conformer-ljspeech-epoch-5.pt",
    "epochs": 15
})

print(response.json())
```

### 例3: チェックポイント一覧の確認

```python
import requests

# チェックポイント一覧を取得
response = requests.get("http://localhost:8000/api/checkpoints")
checkpoints = response.json()["checkpoints"]

for checkpoint in checkpoints:
    print(f"Epoch {checkpoint['epoch']}: {checkpoint['name']} ({checkpoint['size_mb']}MB)")
```

## チェックポイントの自動管理

システムは以下の自動管理機能を提供します：

1. **自動保存**: 各エポック終了時にチェックポイントを自動保存
2. **自動クリーンアップ**: 設定された保持数を超える古いチェックポイントを自動削除
3. **最新リンク**: 最新のチェックポイントへのシンボリックリンクを自動更新

## トラブルシューティング

### よくある問題

1. **チェックポイントが見つからない**
   - エラー: `No checkpoint found for model 'conformer' and dataset 'ljspeech'`
   - 解決策: まず学習を実行してチェックポイントを作成してください

2. **指定されたチェックポイントが存在しない**
   - エラー: `Specified checkpoint 'invalid-checkpoint' not found`
   - 解決策: `/api/checkpoints`で利用可能なチェックポイントを確認してください

3. **学習が既に実行中**
   - エラー: `Training is already in progress`
   - 解決策: 現在の学習を停止してから再開してください

### ログの確認

学習再開の詳細なログは、WebSocket経由でリアルタイムに確認できます：

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'log') {
        console.log(`[${data.payload.level}] ${data.payload.message}`);
    }
};
```

## ベストプラクティス

1. **定期的なチェックポイント保存**: 長時間の学習では、適切な間隔でチェックポイントが保存されることを確認
2. **ストレージ管理**: チェックポイントの保持数を適切に設定してストレージ使用量を管理
3. **バックアップ**: 重要なチェックポイントは別途バックアップを取ることを推奨
4. **進捗監視**: WebSocketを使用して学習の進捗をリアルタイムで監視

## テスト

学習再開機能のテストには、提供されているテストスクリプトを使用できます：

```bash
cd /app
python test_resume_training.py
```

このスクリプトは以下のテストを実行します：
- チェックポイント一覧の取得
- 学習の開始と再開
- 進捗の監視
- エラーハンドリング

