# 設定ファイル仕様書 (config.yaml)

このドキュメントは、バックエンドアプリケーションの設定ファイル `backend/config.yaml` の構造と各パラメータについて詳述します。

## 1. ルートレベル

```yaml
# 使用可能なモデルとデータセットのリスト
available_models:
  - conformer
  - rnn-t

available_datasets:
  - ljspeech

# モデルごとの詳細設定
models:
  # ... (セクション 2 を参照)

# データセットごとの設定
datasets:
  # ... (セクション 3 を参照)

# 学習のグローバル設定
training:
  # ... (セクション 4 を参照)
```

-   `available_models` (list[string], required): API経由で学習を開始できるモデル名のリスト。このリストにないモデルは `POST /train/start` で指定できません。
-   `available_datasets` (list[string], required): API経由で利用できるデータセット名のリスト。

## 2. `models` セクション

各モデルのハイパーパラメータやアーキテクチャに関する設定を記述します。キーはモデル名（`available_models` と一致）です。

```yaml
models:
  conformer:
    # --- アーキテクチャ設定 ---
    input_dim: 80         # 入力特徴量の次元数 (例: メルスペクトログラムの次元)
    encoder_dim: 256      # エンコーダの隠れ層の次元数
    num_encoder_layers: 4 # エンコーダブロックの数
    num_heads: 4          # Multi-Head Attention のヘッド数
    kernel_size: 31       # Convolutionモジュールのカーネルサイズ
    dropout: 0.1          # ドロップアウト率

    # --- トークナイザ設定 ---
    tokenizer:
      type: "SentencePiece" # "Character", "Word" など
      vocab_size: 5000
      model_path: "/path/to/tokenizer.model" # SentencePieceモデルのパス

  rnn-t:
    # ... (RNN-Tモデル用の設定)
```

-   各パラメータは、対応するモデルクラス (`app/models/{model_name}.py`) の `__init__` メソッドで解釈されます。
-   モデルごとに必要なパラメータは異なります。

## 3. `datasets` セクション

各データセットのパスや前処理に関する設定を記述します。キーはデータセット名（`available_datasets` と一致）です。

```yaml
datasets:
  ljspeech:
    # --- データパス ---
    path: "/data/ljspeech" # Dockerコンテナ内のデータセットルートパス

    # --- 音声前処理設定 ---
    sample_rate: 22050    # リサンプリングするサンプルレート
    n_fft: 1024           # STFTのウィンドウサイズ
    win_length: 1024      # STFTのウィンドウ長
    hop_length: 256       # STFTのホップ長
    n_mels: 80            # 生成するメルフィルタバンクの数
    f_min: 0              # メルスペクトログラムの最小周波数
    f_max: 8000           # メルスペクトログラムの最大周波数

    # --- テキスト前処理 ---
    text_cleaners: ['english_cleaners'] # 適用するテキストクリーナーのリスト
```

-   `path`: データセットのファイルが格納されているディレクトリへのパス。
-   音声前処理設定は、`torchaudio` や `librosa` を使ってメルスペクトログラムを計算する際に使用されます。
-   `text_cleaners`: テキストを正規化するためのクリーナー関数の名前。

## 4. `training` セクション

学習プロセス全体に適用されるグローバルな設定を記述します。

```yaml
training:
  # --- オプティマイザ設定 ---
  optimizer: "AdamW"      # 使用するオプティマイザ名 (torch.optim内のクラス名)
  learning_rate: 0.001    # 学習率
  weight_decay: 0.01      # AdamWのweight decay
  betas: [0.9, 0.98]      # Adam/AdamWのbetaパラメータ
  eps: 1.0e-9             # Adam/AdamWのepsilon

  # --- スケジューラ設定 ---
  scheduler: "WarmupLR"   # 学習率スケジューラの名前 (オプション)
  warmup_steps: 4000      # WarmupLRのウォームアップステップ数

  # --- 学習ループ設定 ---
  batch_size: 32          # バッチサイズ
  num_epochs: 100         # 総エポック数
  grad_clip_thresh: 1.0   # 勾配クリッピングの閾値
  log_interval: 10        # ログを記録するステップ間隔 (steps)
  checkpoint_interval: 1  # チェックポイントを保存する間隔 (epochs)
```

-   `optimizer`: `torch.optim` で利用可能なオプティマイザの名前を指定します。
-   `scheduler`: カスタム実装またはライブラリの学習率スケジューラを指定します。
-   `checkpoint_interval`: 何エポックごとにチェックポイントを保存するかを指定します。
