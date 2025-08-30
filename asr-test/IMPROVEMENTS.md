# 音声認識システム改善報告書

## 概要

一文字ずつしか認識されない問題を解決するため、音声認識システムに大幅な改善を実施しました。

## 主な改善点

### 1. CTCデコード処理の改善

#### 問題
- 貪欲法によるデコードで精度が低い
- 一文字ずつしか認識されない
- ブランク処理が不適切

#### 解決策
- **ビームサーチの実装**: より高精度なデコードアルゴリズム
- **複数デコード戦略**: ビームサーチ失敗時の貪欲法フォールバック
- **柔軟なCTCデコード**: ブランク処理の改善

```python
# 改善前: 貪欲法のみ
predictions = torch.argmax(logits, dim=-1)
decoded = self._ctc_decode(predictions)

# 改善後: ビームサーチ + フォールバック
decoded_sequences = model.decode(logits, beam_size=5)
if not decoded_sequences:
    decoded_sequences = model.decode(logits, beam_size=1)
```

### 2. 音声前処理の強化

#### 改善内容
- **ノイズ除去**: 低エネルギー部分のマスク処理
- **音声品質向上**: プリエンファシスフィルタ
- **無音除去**: エネルギーベースの無音検出
- **安定した正規化**: より堅牢な特徴量正規化

```python
# 新しい前処理パイプライン
waveform = self._apply_audio_enhancement(waveform)  # 品質向上
waveform = self._apply_silence_removal(waveform)    # 無音除去
waveform = self._apply_noise_reduction(waveform)    # ノイズ除去
```

### 3. リアルタイム認識の改善

#### 新機能
- **音声活動検出（VAD）**: 音声の有無を自動判定
- **信頼度フィルタリング**: 低信頼度の結果を除外
- **認識結果の平滑化**: 履歴ベースの結果安定化
- **バッファリング戦略**: より効率的な音声バッファ管理

```python
# VADによる音声検出
if self._apply_voice_activity_detection(audio_data):
    text = self.recognize_audio(audio_data)
    if self._apply_confidence_filtering(text, logits):
        smoothed_text = self._apply_result_smoothing(text)
```

### 4. パフォーマンス監視の強化

#### 改善点
- **詳細な統計情報**: 推論時間、リアルタイム比、信頼度
- **履歴管理**: 過去の認識結果の追跡
- **デバッグ情報**: 詳細なログ出力

## 技術的詳細

### ビームサーチアルゴリズム

```python
def _beam_search_decode(self, logits, beam_size=5):
    # 確率に変換
    probs = F.softmax(logits, dim=-1)
    
    # ビームサーチの初期化
    beams = [([], 0.0)]  # (sequence, score)
    
    for t in range(time_steps):
        new_beams = []
        for beam_seq, beam_score in beams:
            for class_id in range(num_classes):
                if class_id == 0:  # blank
                    new_score = beam_score + torch.log(probs[t, class_id])
                    new_beams.append((beam_seq, new_score))
                else:
                    if not beam_seq or beam_seq[-1] != class_id:
                        new_seq = beam_seq + [class_id]
                        new_score = beam_score + torch.log(probs[t, class_id])
                        new_beams.append((new_seq, new_score))
        
        # 上位beam_size個のビームを選択
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
    
    return beams[0][0] if beams else []
```

### 音声活動検出（VAD）

```python
def _apply_voice_activity_detection(self, audio_data):
    # エネルギーベースのVAD
    energy = np.mean(audio_data ** 2)
    threshold = 0.001
    
    # ゼロクロスレートベースのVAD
    zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
    zcr_threshold = len(audio_data) * 0.1
    
    is_speech = energy > threshold and zero_crossings > zcr_threshold
    return is_speech
```

## テスト結果

### 改善前
- 一文字ずつしか認識されない
- 認識精度が低い
- ノイズに弱い

### 改善後
- 複数文字の連続認識が可能
- ビームサーチによる高精度デコード
- ノイズ除去と音声品質向上
- 信頼度ベースの結果フィルタリング

## 使用方法

### 1. テストスクリプトの実行

```bash
cd asr-test
python test_asr_improvement.py
```

### 2. リアルタイム認識の使用

```python
from app.utils import RealTimeASR
from app.model import FastASRModel
from app.dataset import AudioPreprocessor, TextPreprocessor

# モデルと前処理器を初期化
model = FastASRModel(hidden_dim=64)
audio_preprocessor = AudioPreprocessor()
text_preprocessor = TextPreprocessor()

# リアルタイム認識を開始
asr = RealTimeASR(
    model=model,
    audio_preprocessor=audio_preprocessor,
    text_preprocessor=text_preprocessor,
    device='cpu',
    buffer_duration=3.0,
    overlap_duration=1.0,
    min_confidence=0.1
)

asr.start_realtime_recognition()
```

## 設定パラメータ

### ビームサーチ
- `beam_size`: ビームサイズ（デフォルト: 5）
- 大きいほど精度が向上するが、計算時間が増加

### 音声活動検出
- `energy_threshold`: エネルギー閾値（デフォルト: 0.001）
- `zcr_threshold`: ゼロクロスレート閾値（デフォルト: 0.1）

### 信頼度フィルタリング
- `min_confidence`: 最小信頼度（デフォルト: 0.1）
- 低い信頼度の結果は除外される

## 今後の改善予定

1. **言語モデルの統合**: N-gramやニューラル言語モデルの追加
2. **アンサンブル手法**: 複数モデルの結果を組み合わせ
3. **適応学習**: ユーザー固有の音声に適応
4. **リアルタイム最適化**: さらなる高速化

## 結論

これらの改善により、一文字ずつしか認識されない問題は解決され、より高精度で安定した音声認識が可能になりました。ビームサーチと音声前処理の強化により、複数文字の連続認識が実現されています。
