# リアルタイム性特化型音声認識モデル設計

## 1. 概要

このドキュメントでは、**リアルタイム性を最優先**とし、精度は度外視した音声認識モデルの設計仕様を定義します。

### 1.1 基本コンセプト

- **超低遅延**: 未来の音声情報を一切参照しない「ゼロ遅延」設計
- **逐次処理**: 音声チャンクを受け取るたびに即座に処理・出力
- **シンプルアーキテクチャ**: 複雑な機構を排除し、高速な順伝播のみで完結

### 1.2 設計方針

```
[音声ストリーム] -> [前処理] -> [エンコーダ] -> [CTCデコーダ] -> [認識結果テキスト]
     (マイク入力)     (チャンク化)   (音響特徴抽出)   (文字確率計算)     (逐次表示)
```

## 2. アーキテクチャ詳細

### 2.1 前処理 (Chunking)

#### 仕様
- **チャンクサイズ**: 100ms固定
- **オーバーラップ**: なし（完全独立処理）
- **サンプリングレート**: 16kHz
- **チャンクあたりのサンプル数**: 1,600サンプル

#### 実装
```python
def create_audio_chunks(audio_stream, chunk_size_ms=100, sample_rate=16000):
    """
    音声ストリームを固定サイズのチャンクに分割
    
    Args:
        audio_stream: 連続音声ストリーム
        chunk_size_ms: チャンクサイズ（ミリ秒）
        sample_rate: サンプリングレート
    
    Returns:
        Generator yielding audio chunks
    """
    chunk_samples = int(sample_rate * chunk_size_ms / 1000)
    
    while True:
        chunk = audio_stream.read(chunk_samples)
        if len(chunk) < chunk_samples:
            break
        yield chunk
```

### 2.2 エンコーダ (Encoder)

#### アーキテクチャ選択理由
- **RNN/GRU採用**: 過去の文脈を内部状態で保持可能
- **一方向処理**: Bi-directional RNNは使用しない（未来参照禁止）
- **軽量設計**: 計算量を最小限に抑制

#### モデル構造
```python
class RealtimeEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 音響特徴抽出
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # RNN層（GRU使用）
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # 出力投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, chunk_features, hidden_state=None):
        """
        Args:
            chunk_features: [batch_size, seq_len, input_dim]
            hidden_state: 前のチャンクの隠れ状態
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            new_hidden_state: 次のチャンク用の隠れ状態
        """
        # 特徴抽出
        features = self.feature_extractor(chunk_features)
        
        # RNN処理
        output, new_hidden_state = self.rnn(features, hidden_state)
        
        # 出力投影
        output = self.output_projection(output)
        
        return output, new_hidden_state
```

#### 状態管理
- **隠れ状態の永続化**: チャンク間で隠れ状態を引き継ぎ
- **メモリ効率**: 過去の全チャンクを保持せず、状態のみ保持
- **初期化**: セッション開始時に隠れ状態をゼロ初期化

### 2.3 CTCデコーダ (CTC Decoder)

#### CTCの利点
- **ストリーミング対応**: 各時間ステップで独立した文字確率計算
- **アライメント不要**: 音声とテキストの時間的対応を自動学習
- **リアルタイム出力**: 遅延なしで文字列を生成可能

#### 実装
```python
class RealtimeCTCDecoder(nn.Module):
    def __init__(self, input_dim=256, vocab_size=1000):
        super().__init__()
        self.vocab_size = vocab_size
        
        # CTC出力層
        self.ctc_head = nn.Linear(input_dim, vocab_size + 1)  # +1 for blank token
        
        # 文字確率の正規化
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, encoder_output):
        """
        Args:
            encoder_output: [batch_size, seq_len, input_dim]
        
        Returns:
            log_probs: [batch_size, seq_len, vocab_size + 1]
        """
        # CTC出力
        logits = self.ctc_head(encoder_output)
        
        # 確率正規化
        log_probs = self.softmax(logits)
        
        return log_probs
    
    def decode_realtime(self, log_probs, threshold=0.5):
        """
        リアルタイムデコード（簡易版）
        
        Args:
            log_probs: [seq_len, vocab_size + 1]
            threshold: 文字検出の閾値
        
        Returns:
            detected_chars: 検出された文字のリスト
        """
        detected_chars = []
        blank_id = self.vocab_size  # 最後のインデックスがblank
        
        for t in range(log_probs.size(0)):
            # 最大確率の文字を取得
            max_prob, max_char = torch.max(log_probs[t], dim=-1)
            
            # 閾値を超え、かつblankでない場合
            if max_prob > threshold and max_char != blank_id:
                detected_chars.append(max_char.item())
        
        return detected_chars
```

### 2.4 後処理 (Post-processing)

#### CTC出力の処理ルール
1. **重複除去**: 連続する同じ文字を1つにまとめる
2. **ブランク削除**: 空白トークンをすべて削除
3. **文字列構築**: 処理済み文字列を結合

#### 実装
```python
def post_process_ctc_output(ctc_sequence, blank_token='_'):
    """
    CTC出力を最終的な文字列に変換
    
    Args:
        ctc_sequence: CTCデコーダの出力文字列
        blank_token: 空白トークン
    
    Returns:
        final_text: 処理済みの文字列
    """
    # 1. 重複除去
    deduplicated = []
    prev_char = None
    
    for char in ctc_sequence:
        if char != prev_char:
            deduplicated.append(char)
        prev_char = char
    
    # 2. ブランク削除
    final_text = ''.join([char for char in deduplicated if char != blank_token])
    
    return final_text
```

## 3. リアルタイム処理フロー

### 3.1 処理パイプライン

```python
class RealtimeASRPipeline:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.encoder = RealtimeEncoder().to(device)
        self.decoder = RealtimeCTCDecoder().to(device)
        self.hidden_state = None
        self.load_model(model_path)
    
    def process_audio_chunk(self, audio_chunk):
        """
        単一の音声チャンクを処理
        
        Args:
            audio_chunk: 音声データ（1,600サンプル）
        
        Returns:
            recognized_text: 認識された文字列
        """
        # 1. 音響特徴抽出
        features = self.extract_features(audio_chunk)
        
        # 2. エンコーダ処理
        encoder_output, self.hidden_state = self.encoder(
            features, self.hidden_state
        )
        
        # 3. CTCデコード
        log_probs = self.decoder(encoder_output)
        
        # 4. リアルタイムデコード
        detected_chars = self.decoder.decode_realtime(log_probs[0])
        
        # 5. 後処理
        recognized_text = self.post_process(detected_chars)
        
        return recognized_text
    
    def reset_state(self):
        """セッション開始時に状態をリセット"""
        self.hidden_state = None
```

### 3.2 ストリーミング処理

```python
def realtime_recognition_loop(audio_stream):
    """
    リアルタイム音声認識のメインループ
    
    Args:
        audio_stream: 音声入力ストリーム
    """
    pipeline = RealtimeASRPipeline(model_path="realtime_model.pt")
    accumulated_text = ""
    
    try:
        for audio_chunk in create_audio_chunks(audio_stream):
            # チャンク処理
            chunk_text = pipeline.process_audio_chunk(audio_chunk)
            
            # 結果を蓄積
            if chunk_text:
                accumulated_text += chunk_text
                
                # リアルタイム表示
                print(f"\r認識結果: {accumulated_text}", end="", flush=True)
    
    except KeyboardInterrupt:
        print(f"\n最終結果: {accumulated_text}")
        pipeline.reset_state()
```

## 4. 性能最適化

### 4.1 計算量削減

#### モデル軽量化
- **パラメータ数**: 100万パラメータ以下を目標
- **レイヤー数**: エンコーダ3層、デコーダ1層
- **隠れ次元**: 256次元（必要に応じて128次元まで削減可能）

#### 推論最適化
- **バッチサイズ**: 1（ストリーミング処理）
- **精度**: FP16使用でメモリ使用量半減
- **並列化**: 可能な限りGPU並列処理を活用

### 4.2 メモリ効率

#### 状態管理
- **隠れ状態のみ保持**: 過去の全チャンクは保持しない
- **定数メモリ使用量**: チャンク数に依存しないメモリ使用量
- **ガベージコレクション**: 不要なテンソルを即座に解放

### 4.3 遅延最小化

#### 処理時間目標
- **チャンク処理時間**: 50ms以下（100msチャンクに対して）
- **エンドツーエンド遅延**: 100ms以下
- **CPU使用率**: 単一コアで50%以下

## 5. 実装仕様

### 5.1 モデル設定

```yaml
# realtime_model_config.yaml
model:
  name: "realtime_asr"
  type: "ctc_based"
  
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

### 5.2 インターフェース

```python
class RealtimeASRModel(BaseASRModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = RealtimeEncoder(config['encoder'])
        self.decoder = RealtimeCTCDecoder(config['decoder'])
        self.hidden_state = None
    
    def forward(self, batch):
        """学習時のフォワードパス"""
        # 学習用の実装
        pass
    
    @torch.no_grad()
    def inference(self, waveform):
        """推論処理（リアルタイム用）"""
        # ストリーミング推論の実装
        pass
    
    def reset_state(self):
        """状態リセット"""
        self.hidden_state = None
```

## 6. 制限事項と割り切り

### 6.1 精度の制限

- **文脈理解**: 長期的な文脈を考慮しない
- **同音異義語**: 「橋」と「箸」の判別が困難
- **漢字変換**: 読み仮名のみの出力
- **句読点**: 句読点の自動挿入なし

### 6.2 適用範囲

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

## 7. 開発ロードマップ

### Phase 1: 基本実装
- [ ] CTCベースの基本モデル実装
- [ ] チャンクベースの前処理実装
- [ ] リアルタイム推論パイプライン実装

### Phase 2: 最適化
- [ ] モデル軽量化
- [ ] 推論速度最適化
- [ ] メモリ使用量削減

### Phase 3: 統合
- [ ] 既存システムとの統合
- [ ] WebSocket対応
- [ ] フロントエンド連携

### Phase 4: 評価・改善
- [ ] 遅延測定
- [ ] 精度評価（参考値）
- [ ] ユーザビリティテスト

## 8. まとめ

このリアルタイム性特化型音声認識モデルは、精度よりも**即座性**を重視した設計となっています。CTCアーキテクチャと軽量なRNNエンコーダを組み合わせることで、理論上最小限の遅延で音声認識を実現します。

このモデルは、リアルタイム性が絶対条件となるアプリケーションの初期プロトタイプとして最適であり、段階的な改善を通じて実用性を高めていくことが可能です。
