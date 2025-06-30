# Scripts フォルダ - Text-to-Speech (TTS) モデル学習・実験スクリプト

このフォルダには、LJSpeechデータセットを使用したText-to-Speech（TTS）モデルの学習・実験用スクリプトが含まれています。

## 📋 目次

- [環境セットアップ](#環境セットアップ)
- [主要スクリプト](#主要スクリプト)
- [学習用スクリプト](#学習用スクリプト)
- [テスト用スクリプト](#テスト用スクリプト)
- [モデル実装](#モデル実装)
- [データセット操作](#データセット操作)
- [使用例](#使用例)
- [トラブルシューティング](#トラブルシューティング)

## 🔧 環境セットアップ

### 依存関係のインストール

```bash
# 仮想環境をアクティベート
source ../.venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 必要なライブラリ

- **tensorflow** (>=2.13.0) - 深層学習フレームワーク
- **tensorflow-datasets** (>=4.9.0) - LJSpeechデータセット用
- **librosa** (>=0.10.0) - 音声処理
- **matplotlib** (>=3.7.0) - 可視化
- **soundfile** (>=0.12.0) - 音声ファイル読み書き
- **numpy** (>=1.24.0) - 数値計算
- **transformers** (>=4.21.0) - Transformerモデル
- **torch** (>=1.13.0) - PyTorchサポート

## 🚀 主要スクリプト

### 1. メイン学習スクリプト: `ljspeech_demo.py`

**最も重要なスクリプト** - LJSpeechデータセットでのTTSモデル学習

#### 基本使用法

```bash
# 高速テスト（10サンプル、自動エポック数）- 推奨：初回テスト用
python ljspeech_demo.py

# FastSpeech2で小データセット学習
python ljspeech_demo.py --limit-samples 10 --epochs 100 --model fastspeech2

# Transformer TTSで小データセット学習
python ljspeech_demo.py --limit-samples 10 --epochs 100 --model transformer_tts

# 中規模データセット学習
python ljspeech_demo.py --limit-samples 50 --epochs 200 --model fastspeech2

# フルデータセット学習（本格トレーニング）
python ljspeech_demo.py --limit-samples None --epochs 2000 --model fastspeech2

# 超高速テスト（3サンプル、20エポック）
python ljspeech_demo.py --limit-samples 3 --epochs 20
```

#### オプション詳細

| オプション | 選択肢 | デフォルト | 説明 |
|-----------|--------|-----------|------|
| `--limit-samples` | 整数 または `None` | `10` | 使用サンプル数（10=高速テスト、None=全データ） |
| `--model` | `fastspeech2`, `transformer_tts` | `fastspeech2` | 使用するTTSモデル |
| `--epochs` | 整数 | 自動最適化 | 学習エポック数（Noneで自動設定） |

#### 機能

- ✅ **複数モデルサポート**: FastSpeech2, Transformer TTS
- ✅ **自動チェックポイント**: 学習途中でCtrl+Cで安全に中断・再開可能
- ✅ **音声サンプル生成**: 各エポック終了時に音声ファイル生成
- ✅ **学習時間予測**: 残り時間と完了予定時刻を表示
- ✅ **階層化出力**: モデル＆サンプル数ごとに独立したディレクトリ
- ✅ **エポック数自動最適化**: データセットサイズに応じた自動設定
- ✅ **効率的データロード**: 小データセット用最適化（split最適化）
- ✅ **学習パラメータ自動調整**: 小データセット用の高学習率・小バッチサイズ

### 2. 簡単学習ランチャー: `run_training.py`

**初心者向け** - コマンドを覚えなくても簡単に学習開始

```bash
# ヘルプ表示
python run_training.py help

# ミニモード学習（10サンプル、2000エポック）
python run_training.py mini

# ミニモード学習（10サンプル、100エポック）
python run_training.py mini 100

# フルモード学習（全データセット、2000エポック）
python run_training.py full

# フルモード学習（全データセット、5000エポック）
python run_training.py full 5000

# モデルテストのみ
python run_training.py test
```

## 🧪 テスト用スクリプト

### 1. FastSpeech2モデルテスト: `test_fastspeech2.py`

FastSpeech2モデルの動作確認

```bash
python test_fastspeech2.py
```

**テスト内容:**
- ✅ モデル構築テスト
- ✅ 個別コンポーネントテスト
- ✅ トレーナー統合テスト
- ✅ 損失関数テスト

### 2. Transformer系モデルテスト: `test_transformer_models.py`

Transformer TTS、VITSモデルの動作確認

```bash
python test_transformer_models.py
```

**テスト内容:**
- ✅ Transformer TTS モデルテスト
- ✅ VITS モデルテスト
- ✅ ljspeech_demo.py との統合テスト

### 3. データセットテスト: `test_dataset.py`

LJSpeechデータセットの読み込み確認

```bash
python test_dataset.py
```

## 🏗️ モデル実装

### 1. FastSpeech2: `fastspeech2_model.py`

高品質な非自己回帰型TTSモデル

**特徴:**
- 🎯 Duration Predictor（継続時間予測）
- 🎵 Pitch Predictor（ピッチ予測）
- ⚡ Energy Predictor（エネルギー予測）
- 🔧 PostNet（後処理ネットワーク）

### 2. Simple Transformer TTS: `simple_transformer_tts.py`

シンプルなTransformerベースTTSモデル

**特徴:**
- 🤖 Multi-Head Attention
- 🔄 Position-wise Feed Forward
- 📊 Layer Normalization
- 🎛️ PostNet

### 3. Simple VITS: `simple_vits.py`

VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）の簡易実装

### 4. 汎用モデル: `transformer_tts_model.py`, `vits_model.py`

その他のモデル実装

## 📊 データセット操作

### 1. データセットダウンロード: `download_dataset.py`

LJSpeechデータセットの事前ダウンロード（Dockerコンテナ用）

```bash
python download_dataset.py
```

## 📁 出力ディレクトリ構造

学習実行後、モデルとサンプル数ごとに整理された以下のディレクトリ構造が作成されます：

```
outputs/
├── fastspeech2/                    # FastSpeech2専用
│   ├── samples_10/                 # 10サンプル学習
│   │   ├── checkpoints/            # チェックポイント
│   │   │   ├── model.keras        # 保存されたモデル
│   │   │   ├── vocabulary.json    # 語彙ファイル
│   │   │   ├── training_state.json # 学習状態
│   │   │   └── dataset_processed.cache
│   │   ├── epoch_samples/          # エポック毎の音声サンプル
│   │   │   ├── epoch_1_20240101_120000.wav
│   │   │   └── ...
│   │   ├── ljspeech_synthesis_model.keras  # 最終モデル
│   │   ├── synthesized_audio_final.wav     # 最終合成音声
│   │   └── synthesis_visualization_final.png # 最終可視化
│   ├── samples_3/                  # 3サンプル学習
│   ├── samples_50/                 # 50サンプル学習
│   └── full/                       # フルデータセット学習
├── transformer_tts/                # Transformer TTS専用
│   ├── samples_10/
│   ├── samples_20/
│   └── full/
└── vits/                          # VITS専用
    └── （同様の構造）
```

**メリット:**
- ✅ **実験管理**: 異なる条件での結果を簡単に比較
- ✅ **並行実験**: 複数の設定で同時実験可能
- ✅ **結果保護**: 実験結果が上書きされない

## 🚀 効率化機能

### 自動最適化機能

新しいシステムでは、データセットサイズに応じて学習パラメータが自動最適化されます：

#### 小データセット（20サンプル以下）の場合
- **エポック数**: 自動計算 `min(500, max(100, 1000 // samples))`
- **学習率**: 1e-3（通常の10倍高速）
- **バッチサイズ**: 小さく調整（過学習促進）
- **データロード**: `split="train[:N]"` で効率化
- **データリピート**: 複数回リピートで過学習促進

#### 例：自動設定値
```bash
# 3サンプル → 333エポック、高学習率
python ljspeech_demo.py --limit-samples 3

# 10サンプル → 100エポック、高学習率  
python ljspeech_demo.py --limit-samples 10

# 50サンプル → 2000エポック、通常学習率
python ljspeech_demo.py --limit-samples 50
```

### 段階的学習戦略

```bash
# Phase 1: 超高速検証（数分）
python ljspeech_demo.py --limit-samples 3 --epochs 20

# Phase 2: 高速テスト（10-30分）
python ljspeech_demo.py --limit-samples 10 --epochs 100

# Phase 3: 中規模実験（1-3時間）
python ljspeech_demo.py --limit-samples 50 --epochs 300

# Phase 4: 本格学習（数時間〜数日）
python ljspeech_demo.py --limit-samples None --epochs 2000
```

## 💡 使用例

### 1. 初心者向け：まずはテスト

```bash
# 0. ヘルプ確認
python ljspeech_demo.py --help

# 1. モデルテスト
python test_fastspeech2.py

# 2. データセットテスト  
python test_dataset.py

# 3. 超高速テスト（3サンプル、自動エポック）
python ljspeech_demo.py --limit-samples 3

# 4. 標準テスト（10サンプル、自動エポック）
python ljspeech_demo.py

# 5. 簡単学習（従来方式との互換性）
python run_training.py mini 10
```

### 2. 開発者向け：モデル比較実験

```bash
# FastSpeech2で学習（10サンプル）
python ljspeech_demo.py --limit-samples 10 --epochs 100 --model fastspeech2

# Transformer TTSで学習（10サンプル）
python ljspeech_demo.py --limit-samples 10 --epochs 100 --model transformer_tts

# 結果比較
ls outputs/fastspeech2/samples_10/epoch_samples/
ls outputs/transformer_tts/samples_10/epoch_samples/

# 異なるサンプル数での比較
python ljspeech_demo.py --limit-samples 3 --epochs 50 --model fastspeech2
python ljspeech_demo.py --limit-samples 20 --epochs 150 --model fastspeech2
```

### 3. 本格運用：フルデータセット学習

```bash
# 長時間学習（中断・再開可能）
python ljspeech_demo.py --limit-samples None --epochs 5000 --model fastspeech2

# 中断された場合（Ctrl+C後）
python ljspeech_demo.py --limit-samples None --epochs 5000 --model fastspeech2
# → 自動的に前回のエポックから再開
```

## 🔄 学習の中断・再開

**安全な中断方法:**
```bash
# 学習中にCtrl+Cを押す
^C
# → 現在のエポック完了後、自動的にチェックポイント保存
```

**再開方法:**
```bash
# 同じコマンドを再実行
python ljspeech_demo.py --limit-samples 10 --epochs 100 --model fastspeech2
# → 自動的に保存されたエポックから再開
```

## ⚠️ トラブルシューティング

### GPU関連

**問題:** GPU out of memory
```bash
# 解決策1: バッチサイズを小さくする（コード内で調整）
# 解決策2: より小さなモデルを使用
python ljspeech_demo.py --model transformer_tts  # FastSpeech2より軽量
```

### データセット関連

**問題:** データセットダウンロードエラー
```bash
# 手動でデータセットディレクトリ作成
mkdir -p ./datasets
python download_dataset.py
```

### モデル関連

**問題:** モデル互換性エラー
```bash
# チェックポイントディレクトリをクリア
rm -rf outputs/[model_name]/checkpoints/
```

## 📈 パフォーマンス最適化

### 1. GPU使用率向上

- `MAX_FRAMES = 430` の調整（メモリ使用量とのトレードオフ）
- バッチサイズの調整

### 2. 学習速度向上

```bash
# 小データセットで最適なエポック数を見つける
python ljspeech_demo.py --limit-samples 10 --epochs 50

# その後フルデータセットで実行
python ljspeech_demo.py --limit-samples None --epochs [optimal_epochs]

# 段階的スケールアップ
python ljspeech_demo.py --limit-samples 3 --epochs 20    # 超高速テスト
python ljspeech_demo.py --limit-samples 10 --epochs 100  # 通常テスト  
python ljspeech_demo.py --limit-samples 50 --epochs 300  # 中規模テスト
python ljspeech_demo.py --limit-samples None --epochs 2000 # 本格学習
```

## 🎵 音声品質向上のヒント

1. **エポック数**: 最低100エポック、推奨500-2000エポック
2. **モデル選択**: 
   - **FastSpeech2**: 最高品質、計算量大
   - **Transformer TTS**: バランス型
3. **データ量**: フルデータセット推奨（13,100サンプル）

## 📞 サポート

問題が発生した場合：

1. まず該当するテストスクリプトを実行
2. エラーメッセージを確認
3. GPU/CPU使用率とメモリ使用量を確認
4. チェックポイントのクリアを試行

---

**🎉 Happy TTS Training! 🎉** 
