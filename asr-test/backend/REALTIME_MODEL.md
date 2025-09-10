# リアルタイム音声認識モデル

## 概要

このドキュメントでは、asr-testプロジェクトに新規追加されたリアルタイム音声認識モデルについて説明します。

## 特徴

- **超低遅延**: 100msチャンクでの逐次処理
- **CTCベース**: ストリーミング対応のCTCデコーダ
- **軽量設計**: 100万パラメータ以下の軽量モデル
- **状態管理**: チャンク間での隠れ状態の永続化

## アーキテクチャ

### モデル構成

```
[音声チャンク] -> [メルスペクトログラム] -> [GRUエンコーダ] -> [CTCデコーダ] -> [認識結果]
```

### 主要コンポーネント

1. **RealtimeEncoder**: GRUベースのエンコーダ
   - 入力次元: 80 (メルスペクトログラム)
   - 隠れ次元: 256
   - 層数: 3
   - 状態の永続化対応

2. **RealtimeCTCDecoder**: CTCベースのデコーダ
   - 語彙サイズ: 1000
   - リアルタイムデコード機能
   - 閾値ベースの文字検出

3. **RealtimeASRPipeline**: ストリーミング処理パイプライン
   - チャンクベースの処理
   - テキストの蓄積
   - 状態管理

## 設定

### config.yaml

```yaml
models:
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
```

## 使用方法

### 1. API経由での使用

```bash
# 音声ファイルでの推論
curl -X POST "http://localhost:8000/api/inference" \
  -F "file=@audio.wav" \
  -F "model_name=realtime"
```

### 2. WebSocket経由での使用

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// ストリーミング開始
ws.send(JSON.stringify({
  type: "start",
  model_name: "realtime",
  sample_rate: 16000,
  format: "f32"
}));

// 音声データ送信
ws.send(audioData);

// ストリーミング停止
ws.send(JSON.stringify({type: "stop"}));
```

### 3. プログラムでの使用

```python
from app.models.realtime import RealtimeASRModel, RealtimeASRPipeline
from app import config_loader

# 設定読み込み
config = config_loader.load_config()
realtime_config = config['models']['realtime']

# モデル初期化
model = RealtimeASRModel(realtime_config)
pipeline = RealtimeASRPipeline(model)

# 音声チャンク処理
chunk_text = pipeline.process_audio_chunk(audio_chunk)

# 蓄積されたテキスト取得
accumulated_text = pipeline.get_accumulated_text()

# 状態リセット
pipeline.reset()
```

## テスト

### テストスクリプトの実行

```bash
# 基本テスト
python test_realtime_model.py

# デモ実行
python demo_realtime.py
```

### テスト内容

1. **モデルコンポーネントテスト**
   - エンコーダの動作確認
   - デコーダの動作確認
   - 特徴抽出の動作確認

2. **音声チャンキングテスト**
   - チャンク分割の動作確認
   - ストリーミング処理のシミュレーション

3. **リアルタイムモデルテスト**
   - 推論処理の動作確認
   - パイプラインの動作確認
   - 状態管理の動作確認

## パフォーマンス

### 目標値

- **チャンク処理時間**: 50ms以下
- **エンドツーエンド遅延**: 100ms以下
- **メモリ使用量**: 512MB以下
- **CPU使用率**: 単一コアで50%以下

### 最適化

- FP16精度の使用
- バッチサイズ1での推論
- 不要なテンソルの即座解放
- GPU並列処理の活用

## 制限事項

### 精度の制限

- 文脈理解なし（長期的な文脈を考慮しない）
- 同音異義語の判別困難
- 漢字変換なし（読み仮名のみ）
- 句読点の自動挿入なし

### 適用範囲

#### 適している用途
- リアルタイム字幕表示
- 音声入力の即時フィードバック
- 会議の議事録作成支援
- 音声コマンド認識

#### 適していない用途
- 高精度な文字起こし
- 文脈依存の言語理解
- 専門用語の正確な認識
- 多言語混在の音声

## 開発ロードマップ

### Phase 1: 基本実装 ✅
- [x] CTCベースの基本モデル実装
- [x] チャンクベースの前処理実装
- [x] リアルタイム推論パイプライン実装

### Phase 2: 最適化
- [ ] モデル軽量化
- [ ] 推論速度最適化
- [ ] メモリ使用量削減

### Phase 3: 統合
- [x] 既存システムとの統合
- [x] WebSocket対応
- [x] フロントエンド連携

### Phase 4: 評価・改善
- [ ] 遅延測定
- [ ] 精度評価（参考値）
- [ ] ユーザビリティテスト

## トラブルシューティング

### よくある問題

1. **モデル読み込みエラー**
   - 設定ファイルの確認
   - 依存関係の確認

2. **推論結果が空**
   - 音声データの品質確認
   - サンプリングレートの確認

3. **メモリ不足**
   - バッチサイズの調整
   - 精度設定の確認

### ログ確認

```bash
# アプリケーションログの確認
docker-compose logs -f backend

# モデル固有のログ
grep "model" /app/logs/asr-api.log
```

## 参考資料

- [リアルタイムモデル設計ドキュメント](../docs/realtime_model_design.md)
- [API仕様書](../docs/api_spec.md)
- [設定仕様書](../docs/config_spec.md)
