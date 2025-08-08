import streamlit as st
import gc
import os
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import librosa
import soundfile as sf

# ALSAエラーを抑制
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'
os.environ['ALSA_CONFIG_PATH'] = '/dev/null'
os.environ['ALSA_PCM_NAME'] = 'null'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PULSE_SERVER'] = 'unix:/tmp/pulse-socket'
os.environ['PULSE_COOKIE'] = '/tmp/pulse-cookie'
os.environ['AUDIODEV'] = 'null'
os.environ['AUDIODRIVER'] = 'null'

# オーディオライブラリの警告を抑制
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# PyTorchの初期化を最適化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# PyTorchのインポート（メモリ効率化）
try:
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except ImportError as e:
    st.error(f"PyTorchのインポートエラー: {e}")
    st.stop()

import numpy as np

# ローカルモジュールのインポート
from app.model import LightweightASRModel, FastASRModel, CHAR_TO_ID, ID_TO_CHAR
from app.dataset import AudioPreprocessor, TextPreprocessor, ASRDataset, create_dataloader, SyntheticDataset
from app.trainer import ASRTrainer, FastTrainer
from app.controlled_trainer import ControlledASRTrainer
from app.ljspeech_dataset import create_ljspeech_dataloader
from app.utils import (
    AudioRecorder, RealTimeASR, AudioProcessor, ModelManager, 
    PerformanceMonitor, create_sample_audio_data, save_sample_dataset
)

# ページ設定
st.set_page_config(
    page_title="リアルタイム音声認識モデル学習",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自動モデルロード機能
def auto_load_latest_model():
    """最新のモデルを自動的に読み込む"""
    try:
        # モデルディレクトリを確認
        model_dir = "models"
        if not os.path.exists(model_dir):
            print("モデルディレクトリが存在しません")
            return False
        
        # 利用可能なモデルファイルを検索
        model_files = []
        for file in os.listdir(model_dir):
            if file.endswith('.pth') or file.endswith('.pt'):
                model_path = os.path.join(model_dir, file)
                # ファイルの作成時刻を取得
                creation_time = os.path.getctime(model_path)
                model_files.append((model_path, creation_time))
        
        if not model_files:
            print("利用可能なモデルファイルが見つかりません")
            return False
        
        # 最新のモデルを選択
        latest_model = max(model_files, key=lambda x: x[1])[0]
        
        # モデル情報ファイルを確認
        model_info_path = latest_model.replace('.pth', '_info.json').replace('.pt', '_info.json')
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
        else:
            # モデル情報ファイルがない場合は、チェックポイントから推測
            print("モデル情報ファイルが見つかりません。チェックポイントから推測します...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                checkpoint = torch.load(latest_model, map_location=device)
                # パラメータのサイズからモデルタイプを推測
                if 'output_layer.weight' in checkpoint['model_state_dict']:
                    output_shape = checkpoint['model_state_dict']['output_layer.weight'].shape
                    if output_shape[1] == 64:  # FastASRModel
                        model_info = {'model_type': 'FastASRModel', 'hidden_dim': 64}
                    elif output_shape[1] == 256:  # LightweightASRModel
                        model_info = {'model_type': 'LightweightASRModel', 'hidden_dim': 128, 'num_layers': 2}
                    else:
                        model_info = {'model_type': 'LightweightASRModel', 'hidden_dim': 128, 'num_layers': 2}
                else:
                    model_info = {'model_type': 'LightweightASRModel', 'hidden_dim': 128, 'num_layers': 2}
            except Exception as e:
                print(f"チェックポイントの読み込みに失敗: {e}")
                return False
        
        print(f"推測されたモデル情報: {model_info}")
        
        # モデルを初期化
        if model_info.get('model_type', '').startswith('Fast'):
            model = FastASRModel(
                hidden_dim=model_info.get('hidden_dim', 64),
                num_classes=len(CHAR_TO_ID)
            )
        else:
            model = LightweightASRModel(
                hidden_dim=model_info.get('hidden_dim', 128),
                num_layers=model_info.get('num_layers', 2),
                num_classes=len(CHAR_TO_ID)
            )
        
        # モデルを読み込み
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(latest_model, map_location=device)
        
        # モデルの状態辞書を読み込み
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("モデルの状態辞書を正常に読み込みました")
        except Exception as e:
            print(f"モデルの状態辞書の読み込みに失敗: {e}")
            print("新しいモデルとして初期化します")
            # 読み込みに失敗した場合は、新しいモデルとして初期化
            model = model.to(device)
            model.eval()
        
        # 前処理器を初期化
        audio_preprocessor = AudioPreprocessor()
        text_preprocessor = TextPreprocessor()
        
        # セッション状態に保存
        st.session_state.model = model
        st.session_state.audio_preprocessor = audio_preprocessor
        st.session_state.text_preprocessor = text_preprocessor
        
        print(f"自動ロード完了: {os.path.basename(latest_model)}")
        return True
        
    except Exception as e:
        print(f"自動モデルロードエラー: {e}")
        return False

# セッション状態の初期化
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'audio_preprocessor' not in st.session_state:
    st.session_state.audio_preprocessor = None
if 'text_preprocessor' not in st.session_state:
    st.session_state.text_preprocessor = None
if 'performance_monitor' not in st.session_state:
    st.session_state.performance_monitor = PerformanceMonitor()
if 'training_history' not in st.session_state:
    st.session_state.training_history = {'loss': [], 'wer': [], 'epoch': []}
if 'controlled_trainer' not in st.session_state:
    st.session_state.controlled_trainer = None
if 'training_status' not in st.session_state:
    st.session_state.training_status = {}
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = {'current_epoch': 0, 'current_batch': 0, 'total_batches': 0}
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None

# モデルが初期化されていない場合は自動ロードを試行
if st.session_state.model is None:
    print("モデルが初期化されていません。自動ロードを試行します...")
    auto_load_result = auto_load_latest_model()
    if not auto_load_result:
        print("自動ロードに失敗しました。新しいモデルを初期化する必要があります。")

# メモリ管理
def clear_memory():
    """メモリをクリア"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 定期的なメモリクリア
if 'last_memory_clear' not in st.session_state:
    st.session_state.last_memory_clear = time.time()

# 5分ごとにメモリクリア
if time.time() - st.session_state.last_memory_clear > 300:
    clear_memory()
    st.session_state.last_memory_clear = time.time()

# タイトル
st.title("🎤 リアルタイム音声認識モデル学習システム")
st.markdown("---")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    # モデル選択
    model_type = st.selectbox(
        "モデルタイプ",
        ["FastASRModel (超軽量)", "LightweightASRModel (軽量)"],
        key="model_type_sidebar",
        help="FastASRModelはリアルタイム推論に最適化されています"
    )
    
    # デバイス選択
    device = st.selectbox(
        "デバイス",
        ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        key="device_sidebar",
        help="GPUが利用可能な場合はcudaを選択してください"
    )
    
    # 学習パラメータ
    st.subheader("🎯 学習パラメータ")
    learning_rate = st.slider("学習率", 1e-5, 1e-2, 1e-3, format="%.5f")
    batch_size = st.slider("バッチサイズ", 1, 32, 8)
    max_epochs = st.slider("最大エポック数", 10, 200, 50)
    
    # 高度な学習パラメータ
    with st.expander("🔧 高度な設定"):
        weight_decay = st.slider("Weight Decay", 0.0, 0.01, 0.0001, format="%.4f")
        gradient_clip = st.slider("Gradient Clipping", 0.0, 10.0, 1.0, format="%.1f")
        enable_early_stopping = st.checkbox("Early Stopping を有効にする", value=False)
        early_stopping_patience = st.slider("Early Stopping Patience", 5, 50, 10, disabled=not enable_early_stopping)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, format="%.1f")
    
    # モデルパラメータ
    st.subheader("🧠 モデルパラメータ")
    hidden_dim = st.slider("隠れ層サイズ", 32, 256, 64 if model_type.startswith("Fast") else 128)
    num_layers = st.slider("LSTM層数", 1, 4, 1 if model_type.startswith("Fast") else 2)
    
    # データセット設定
    st.subheader("📁 データセット設定")
    dataset_type = st.selectbox(
        "データセットタイプ",
        ["サンプルデータ", "LJSpeechデータセット", "カスタムデータ"],
        key="dataset_type_sidebar",
        help="使用するデータセットを選択してください"
    )
    
    if dataset_type == "カスタムデータ":
        custom_data_path = st.text_input(
            "カスタムデータパス",
            value="data/custom",
            help="カスタムデータセットのパスを入力してください"
        )

# メインコンテンツ
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 ホーム", "🎯 モデル学習", "📊 学習進捗", "🎤 リアルタイム認識", "📈 結果分析", "🚀 学習制御"])

with tab1:
    st.header("🚀 システム概要")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✨ 特徴")
        st.markdown("""
        - **光速でほぼリアルタイム動作**
        - **軽量CNN + LSTM + CTCアーキテクチャ**
        - **Dockerコンテナ化済み**
        - **リアルタイム学習進捗表示**
        - **マイク入力でのリアルタイム推論**
        - **高度な学習制御機能**
        """)
    
    with col2:
        st.subheader("📈 パフォーマンス")
        if st.session_state.performance_monitor:
            stats = st.session_state.performance_monitor.get_statistics()
            if stats:
                st.metric("平均推論時間", f"{stats['avg_inference_time']:.4f}s")
                st.metric("リアルタイム比", f"{stats['avg_realtime_ratio']:.2f}x")
                st.metric("総推論回数", stats['total_inferences'])
    
    # システム情報
    st.subheader("💻 システム情報")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PyTorch", torch.__version__)
        st.metric("CUDA利用可能", "✅" if torch.cuda.is_available() else "❌")
    
    with col2:
        if torch.cuda.is_available():
            st.metric("GPU", torch.cuda.get_device_name(0))
            st.metric("GPUメモリ", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    with col3:
        st.metric("デバイス", device)
        if st.session_state.model:
            params = sum(p.numel() for p in st.session_state.model.parameters())
            st.metric("モデルパラメータ", f"{params:,}")

with tab2:
    st.header("🎯 モデル学習")
    
    # ステップ1: モデル初期化
    st.subheader("1️⃣ モデル初期化")
    
    # 自動ロードの結果を表示
    if st.session_state.model is not None:
        st.success("✅ モデルが自動的に読み込まれました")
        model_params = sum(p.numel() for p in st.session_state.model.parameters())
        
        # モデルの学習状態をチェック
        is_trained = False
        if hasattr(st.session_state.model, 'is_trained'):
            is_trained = st.session_state.model.is_trained()
        
        st.info(f"📊 モデル情報: {st.session_state.model.__class__.__name__}, パラメータ数: {model_params:,}")
        
        if is_trained:
            st.success("✅ モデルは学習済みです")
        else:
            st.warning("⚠️ モデルは未学習です。学習を実行してください。")
    else:
        st.warning("⚠️ モデルが初期化されていません。下のボタンで初期化してください。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 モデルを初期化", type="primary"):
            with st.spinner("モデルを初期化中..."):
                try:
                    # 前処理器の初期化
                    st.session_state.audio_preprocessor = AudioPreprocessor()
                    st.session_state.text_preprocessor = TextPreprocessor()
                    
                    # モデルの初期化
                    if model_type.startswith("Fast"):
                        st.session_state.model = FastASRModel(
                            hidden_dim=hidden_dim,
                            num_classes=len(CHAR_TO_ID)
                        )
                    else:
                        st.session_state.model = LightweightASRModel(
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            num_classes=len(CHAR_TO_ID)
                        )
                    
                    st.success("✅ モデルが初期化されました！")
                    
                    # モデル情報の表示
                    params = sum(p.numel() for p in st.session_state.model.parameters())
                    st.info(f"📊 モデル情報: {params:,}パラメータ")
                    
                except Exception as e:
                    st.error(f"❌ モデル初期化に失敗しました: {str(e)}")
    
    with col2:
        if st.session_state.model:
            st.success("✅ モデルが初期化済みです")
        else:
            st.warning("⚠️ モデルが初期化されていません")
    
    # ステップ2: データセット準備
    st.subheader("2️⃣ データセット準備")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**サンプルデータ生成**")
        if st.button("🎵 サンプルデータセット生成"):
            with st.spinner("サンプルデータセットを生成中..."):
                try:
                    samples = create_sample_audio_data(num_samples=50, duration=3.0)
                    save_sample_dataset(samples, "data/raw")
                    st.success(f"✅ {len(samples)}個のサンプルデータを生成しました！")
                    st.session_state.dataset_info = {
                        'type': 'sample',
                        'samples': len(samples),
                        'path': 'data/raw'
                    }
                    st.info(f"✅ データセット情報を設定しました: {st.session_state.dataset_info}")
                except Exception as e:
                    st.error(f"❌ サンプルデータ生成に失敗しました: {str(e)}")
    
    with col2:
        st.write("**カスタムデータアップロード**")
        uploaded_files = st.file_uploader(
            "音声ファイルをアップロード",
            type=['wav', 'mp3', 'flac', 'm4a'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)}個のファイルがアップロードされました")
            # ファイルの保存処理をここに追加
            st.session_state.dataset_info = {
                'type': 'custom',
                'samples': len(uploaded_files),
                'path': 'data/custom'
            }
            st.info(f"✅ カスタムデータセット情報を設定しました: {st.session_state.dataset_info}")
    
    # データセット選択
    st.write("**データセット選択**")
    dataset_selection = st.selectbox(
        "使用するデータセットを選択",
        ["サンプルデータ", "LJSpeechデータセット", "カスタムデータ"],
        key="dataset_selection_tab2",
        help="学習に使用するデータセットを選択してください"
    )
    
    if dataset_selection == "LJSpeechデータセット":
        ljspeech_dir = "/app/datasets/ljspeech/1.1.1"
        if os.path.exists(ljspeech_dir):
            st.success("✅ LJSpeechデータセットが利用可能です")
            st.session_state.dataset_info = {
                'type': 'ljspeech',
                'samples': 'unknown',
                'path': ljspeech_dir
            }
            st.info(f"✅ LJSpeechデータセット情報を設定しました: {st.session_state.dataset_info}")
        else:
            st.error("❌ LJSpeechデータセットが見つかりません")
            st.info("ℹ️ サンプルデータを生成するか、カスタムデータをアップロードしてください")
            st.session_state.dataset_info = None
    elif dataset_selection == "カスタムデータ":
        if not st.session_state.dataset_info or st.session_state.dataset_info['type'] != 'custom':
            st.warning("⚠️ カスタムデータをアップロードしてください")
    else:  # サンプルデータ
        if not st.session_state.dataset_info or st.session_state.dataset_info['type'] != 'sample':
            st.info("ℹ️ サンプルデータを生成してください")
    
    # データセット情報の表示
    if st.session_state.dataset_info and isinstance(st.session_state.dataset_info, dict):
        st.info(f"📊 現在のデータセット: {st.session_state.dataset_info['type']} ({st.session_state.dataset_info['samples']}サンプル)")
    elif st.session_state.dataset_info:
        st.warning("⚠️ データセット情報が不正です")
        st.session_state.dataset_info = None  # 不正なデータをクリア
    
    # ステップ3: 学習実行
    st.subheader("3️⃣ 学習実行")
    
    if not st.session_state.model:
        st.warning("⚠️ まずモデルを初期化してください。")
    elif not st.session_state.dataset_info or not isinstance(st.session_state.dataset_info, dict):
        st.warning("⚠️ データセットを準備してください。")
    else:
        # 学習設定の確認
        st.write("**学習設定の確認**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("学習率", f"{learning_rate:.5f}")
            st.metric("バッチサイズ", batch_size)
        
        with col2:
            st.metric("最大エポック", max_epochs)
            st.metric("隠れ層サイズ", hidden_dim)
        
        with col3:
            st.metric("デバイス", device)
            st.metric("データセット", st.session_state.dataset_info['type'])
        
        # 学習制御ボタン
        st.write("**学習制御**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ 学習開始", type="primary"):
                try:
                    # データセット情報の検証
                    if not st.session_state.dataset_info or not isinstance(st.session_state.dataset_info, dict):
                        st.error("❌ データセット情報が不正です")
                        st.stop()
                    
                    # データローダーの作成
                    if st.session_state.dataset_info['type'] == 'sample':
                        dataset = ASRDataset(
                            data_dir=st.session_state.dataset_info['path'],
                            audio_preprocessor=st.session_state.audio_preprocessor,
                            text_preprocessor=st.session_state.text_preprocessor
                        )
                        train_loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)
                    elif st.session_state.dataset_info['type'] == 'ljspeech':
                        ljspeech_dir = "/app/datasets/ljspeech/1.1.1"
                        if os.path.exists(ljspeech_dir):
                            try:
                                train_loader, dataset_info = create_ljspeech_dataloader(
                                    data_dir=ljspeech_dir,
                                    audio_preprocessor=st.session_state.audio_preprocessor,
                                    text_preprocessor=st.session_state.text_preprocessor,
                                    batch_size=batch_size
                                )
                                st.success(f"✅ LJSpeechデータセット読み込み完了: {dataset_info['total_samples']}サンプル")
                            except ValueError as e:
                                st.error(f"❌ LJSpeechデータセットエラー: {str(e)}")
                                st.info("ℹ️ サンプルデータを生成するか、カスタムデータをアップロードしてください")
                                st.stop()
                        else:
                            st.error("❌ LJSpeechデータセットが見つかりません")
                            st.info("ℹ️ サンプルデータを生成するか、カスタムデータをアップロードしてください")
                            st.stop()
                    elif st.session_state.dataset_info['type'] == 'custom':
                        # カスタムデータセットの処理
                        if isinstance(st.session_state.dataset_info, dict):
                            custom_path = st.session_state.dataset_info.get('path', 'data/custom')
                        else:
                            custom_path = 'data/custom'
                        if os.path.exists(custom_path):
                            dataset = ASRDataset(
                                data_dir=custom_path,
                                audio_preprocessor=st.session_state.audio_preprocessor,
                                text_preprocessor=st.session_state.text_preprocessor
                            )
                            train_loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)
                            st.success(f"✅ カスタムデータセット読み込み完了: {len(dataset)}サンプル")
                        else:
                            st.error(f"❌ カスタムデータセットが見つかりません: {custom_path}")
                            st.stop()
                    
                    # 制御可能なトレーナーの初期化
                    st.session_state.controlled_trainer = ControlledASRTrainer(
                        model=st.session_state.model,
                        train_loader=train_loader,
                        device=device,
                        learning_rate=learning_rate,
                        max_epochs=max_epochs,
                        model_save_dir="models",
                        weight_decay=weight_decay,
                        gradient_clip=gradient_clip,
                        early_stopping_patience=early_stopping_patience if enable_early_stopping else None,
                        validation_split=validation_split
                    )
                    
                    # 学習開始
                    st.session_state.training_start_time = time.time()
                    result = st.session_state.controlled_trainer.start_training()
                    st.success(result["message"])
                    
                except Exception as e:
                    import traceback
                    st.error(f"❌ 学習開始に失敗しました: {str(e)}")
                    st.error(f"詳細: {traceback.format_exc()}")
                    st.error(f"dataset_info: {st.session_state.dataset_info}")
                    st.error(f"dataset_info type: {type(st.session_state.dataset_info)}")
        
        with col2:
            if st.button("⏸️ 一時停止"):
                if st.session_state.controlled_trainer:
                    result = st.session_state.controlled_trainer.pause_training()
                    st.info(result["message"])
                else:
                    st.warning("⚠️ トレーナーが初期化されていません")
        
        with col3:
            if st.button("▶️ 再開"):
                if st.session_state.controlled_trainer:
                    result = st.session_state.controlled_trainer.resume_training()
                    st.success(result["message"])
                else:
                    st.warning("⚠️ トレーナーが初期化されていません")
        
        with col4:
            if st.button("⏹️ 停止"):
                if st.session_state.controlled_trainer:
                    result = st.session_state.controlled_trainer.stop_training()
                    st.warning(result["message"])
                else:
                    st.warning("⚠️ トレーナーが初期化されていません")

with tab3:
    st.header("📊 学習進捗")
    
    # 自動更新の設定
    auto_refresh = st.checkbox("🔄 自動更新", value=True, help="学習進捗を自動的に更新します")
    
    if auto_refresh and st.session_state.controlled_trainer:
        status = st.session_state.controlled_trainer.get_training_status()
        if status["is_training"]:
            # 自動更新のためのJavaScript
            st.markdown(
                """
                <script>
                    setTimeout(function(){
                        window.location.reload();
                    }, 5000);
                </script>
                """,
                unsafe_allow_html=True
            )
    
    if not st.session_state.controlled_trainer:
        st.info("ℹ️ 学習を開始すると進捗が表示されます")
    else:
        # 学習状態の表示
        status = st.session_state.controlled_trainer.get_training_status()
        
        # リアルタイム進捗バー
        if status["is_training"]:
            progress = (status["current_epoch"] * status["total_batches"] + status["current_batch"]) / (status["max_epochs"] * status["total_batches"])
            st.progress(progress)
            st.write(f"進捗: {progress:.1%}")
            
            # 推定残り時間
            if status["current_batch"] > 0:
                elapsed_time = time.time() - getattr(st.session_state, 'training_start_time', time.time())
                avg_time_per_batch = elapsed_time / (status["current_epoch"] * status["total_batches"] + status["current_batch"])
                remaining_batches = (status["max_epochs"] * status["total_batches"]) - (status["current_epoch"] * status["total_batches"] + status["current_batch"])
                estimated_remaining_time = remaining_batches * avg_time_per_batch
                
                st.info(f"⏱️ 推定残り時間: {estimated_remaining_time/60:.1f}分")
        
        # 学習状態メトリクス
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("学習中", "✅" if status["is_training"] else "❌")
            st.metric("一時停止", "✅" if status["is_paused"] else "❌")
        
        with col2:
            st.metric("現在のエポック", f"{status['current_epoch'] + 1}/{status['max_epochs']}")
            st.metric("現在のバッチ", f"{status['current_batch']}/{status['total_batches']}")
        
        with col3:
            st.metric("ベスト損失", f"{status['best_val_loss']:.4f}")
            st.metric("ベストエポック", status["best_epoch"] + 1)
        
        with col4:
            st.metric("学習率", f"{learning_rate:.5f}")
            st.metric("残り時間", "計算中..." if status["is_training"] else "停止中")
        
        # リアルタイム学習曲線
        if status["train_losses"]:
            st.subheader("📈 学習曲線")
            
            # Plotlyを使用したインタラクティブなグラフ
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Loss', 'Validation Loss', 'Training WER', 'Validation WER'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 損失曲線
            epochs = list(range(1, len(status["train_losses"]) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=status["train_losses"], name="Train Loss", line=dict(color='blue')),
                row=1, col=1
            )
            if status["val_losses"]:
                fig.add_trace(
                    go.Scatter(x=epochs, y=status["val_losses"], name="Val Loss", line=dict(color='red')),
                    row=1, col=2
                )
            
            # WER曲線
            if status["train_wers"]:
                fig.add_trace(
                    go.Scatter(x=epochs, y=status["train_wers"], name="Train WER", line=dict(color='green')),
                    row=2, col=1
                )
            if status["val_wers"]:
                fig.add_trace(
                    go.Scatter(x=epochs, y=status["val_wers"], name="Val WER", line=dict(color='orange')),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # チェックポイント管理
        st.subheader("💾 チェックポイント管理")
        
        checkpoints = st.session_state.controlled_trainer.get_available_checkpoints()
        
        if checkpoints:
            selected_checkpoint = st.selectbox(
                "チェックポイントを選択",
                checkpoints,
                key="checkpoint_select_tab3",
                help="読み込むチェックポイントを選択してください"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 チェックポイント読み込み"):
                    checkpoint_path = os.path.join("models", selected_checkpoint)
                    result = st.session_state.controlled_trainer.load_checkpoint(checkpoint_path)
                    st.success(result["message"])
            
            with col2:
                if st.button("💾 現在の状態を保存"):
                    result = st.session_state.controlled_trainer.save_checkpoint()
                    st.success(result["message"])
        else:
            st.info("ℹ️ 利用可能なチェックポイントがありません")

with tab4:
    st.header("🎤 リアルタイム音声認識")
    
    if not st.session_state.model:
        st.warning("⚠️ まずモデルを初期化してください。")
    else:
        # モデルの学習状態をチェック
        if hasattr(st.session_state.model, 'is_trained') and not st.session_state.model.is_trained():
            st.error("❌ **モデルが学習されていません！**")
            st.warning("⚠️ 現在のモデルは初期化されたばかりで、実際の音声データで学習されていません。")
            st.info("💡 **解決方法**:")
            st.info("1. 「🎯 モデル学習」タブに移動")
            st.info("2. 音声データをアップロードまたは録音")
            st.info("3. モデルの学習を実行")
            st.info("4. 学習完了後に再度音声認識を試してください")
            
            # 学習状態の詳細情報
            with st.expander("🔍 モデル詳細情報"):
                params = sum(p.numel() for p in st.session_state.model.parameters())
                st.write(f"**モデルタイプ**: {st.session_state.model.__class__.__name__}")
                st.write(f"**パラメータ数**: {params:,}")
                st.write(f"**学習状態**: 未学習")
                st.write(f"**推奨アクション**: モデル学習の実行")
        else:
            st.success("✅ モデルが学習済みです。音声認識を開始できます。")
        
        st.subheader("🎙️ マイク入力")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 録音開始", type="primary"):
                st.session_state.recording = True
                st.session_state.recognized_text = []
                
                # 録音機の初期化
                recorder = AudioRecorder()
                
                # 録音開始を試行
                if recorder.start_recording():
                    # リアルタイム認識の実行
                    realtime_asr = RealTimeASR(
                        model=st.session_state.model,
                        audio_preprocessor=st.session_state.audio_preprocessor,
                        text_preprocessor=st.session_state.text_preprocessor,
                        device=device
                    )
                    
                    # 認識結果の表示
                    result_placeholder = st.empty()
                    
                    try:
                        for i in range(10):  # 10回の認識を実行
                            audio_data = recorder.get_audio_data(3.0)  # 3秒間の音声
                            
                            if len(audio_data) > 0:
                                start_time = time.time()
                                text = realtime_asr.recognize_audio(audio_data)
                                inference_time = time.time() - start_time
                                
                                if text.strip():
                                    st.session_state.recognized_text.append(text)
                                    result_placeholder.write(f"🎯 認識結果: **{text}**")
                                    
                                    # パフォーマンス記録
                                    st.session_state.performance_monitor.record_inference(
                                        inference_time, 3.0
                                    )
                            
                            time.sleep(0.1)  # 少し待機
                    
                    finally:
                        recorder.close()
                        st.session_state.recording = False
                else:
                    st.error("❌ マイクアクセスに失敗しました")
                    st.info("ℹ️ Dockerコンテナ内ではマイクアクセスが制限されています")
                    st.info("ℹ️ 代わりに音声ファイルをアップロードして認識してください")
                    st.session_state.recording = False
        
        with col2:
            if st.button("⏹️ 録音停止"):
                st.session_state.recording = False
                st.success("✅ 録音を停止しました")
        
        # 音声ファイルアップロード（代替手段）
        st.subheader("📁 音声ファイルアップロード")
        st.info("ℹ️ Dockerコンテナ内ではマイクアクセスが制限されているため、音声ファイルをアップロードして認識してください")
        
        # デモ用音声生成
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎵 デモ音声生成", help="テスト用の音声データを生成します"):
                try:
                    from app.utils import create_sample_audio_data
                    samples = create_sample_audio_data(num_samples=1, duration=3.0)
                    audio_data = samples[0][0]  # 最初のサンプルの音声データ
                    
                    # 音声データの情報を表示
                    st.info(f"📊 生成された音声データ: 長さ={len(audio_data)}サンプル, 範囲=[{audio_data.min():.4f}, {audio_data.max():.4f}]")
                    
                    # 一時ファイルとして保存
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        sf.write(tmp_file.name, audio_data, 16000)
                        temp_path = tmp_file.name
                    
                    # 認識実行
                    realtime_asr = RealTimeASR(
                        model=st.session_state.model,
                        audio_preprocessor=st.session_state.audio_preprocessor,
                        text_preprocessor=st.session_state.text_preprocessor,
                        device=device
                    )
                    
                    start_time = time.time()
                    text = realtime_asr.recognize_audio(audio_data)
                    inference_time = time.time() - start_time
                    
                    # 結果表示
                    if text.strip():
                        st.success(f"🎯 認識結果: **{text}**")
                    else:
                        st.warning("⚠️ 認識結果が空です。")
                        st.info("🔍 デバッグ情報:")
                        st.code(f"音声データ長: {len(audio_data)}サンプル")
                        st.code(f"推論時間: {inference_time:.4f}秒")
                        st.info("💡 ヒント: モデルが学習済みでも、音声の品質や内容によって認識できない場合があります。")
                    
                    st.info(f"⏱️ 推論時間: {inference_time:.4f}秒")
                    
                    # パフォーマンス記録
                    st.session_state.performance_monitor.record_inference(
                        inference_time, len(audio_data) / 16000
                    )
                    
                    # 履歴に追加
                    if not hasattr(st.session_state, 'recognized_text'):
                        st.session_state.recognized_text = []
                    st.session_state.recognized_text.append(text)
                    
                    # 一時ファイル削除
                    os.unlink(temp_path)
                    
                except Exception as e:
                    st.error(f"❌ デモ音声生成に失敗しました: {str(e)}")
                    import traceback
                    st.error(f"詳細: {traceback.format_exc()}")
        
        with col2:
            st.info("💡 ヒント: デモ音声生成ボタンでテスト用の音声データを生成できます")
        
        uploaded_audio = st.file_uploader(
            "音声ファイルをアップロードして認識",
            type=['wav', 'mp3', 'flac', 'm4a'],
            key="audio_upload_tab4"
        )
        
        if uploaded_audio and st.session_state.model:
            if st.button("🎯 音声認識実行", type="primary"):
                try:
                    # 音声ファイルを一時保存
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_audio.getvalue())
                        temp_path = tmp_file.name
                    
                    # 音声を読み込み
                    audio, sr = librosa.load(temp_path, sr=16000)
                    
                    # 認識実行
                    realtime_asr = RealTimeASR(
                        model=st.session_state.model,
                        audio_preprocessor=st.session_state.audio_preprocessor,
                        text_preprocessor=st.session_state.text_preprocessor,
                        device=device
                    )
                    
                    start_time = time.time()
                    text = realtime_asr.recognize_audio(audio)
                    inference_time = time.time() - start_time
                    
                    # 結果表示
                    if text.strip():
                        st.success(f"🎯 認識結果: **{text}**")
                    else:
                        st.warning("⚠️ 認識結果が空です。")
                        st.info("🔍 デバッグ情報:")
                        st.code(f"音声ファイル: {uploaded_audio.name}")
                        st.code(f"音声データ長: {len(audio)}サンプル")
                        st.code(f"サンプリングレート: {sr}Hz")
                        st.code(f"推論時間: {inference_time:.4f}秒")
                        st.info("💡 ヒント: 音声の品質や内容によって認識できない場合があります。")
                    
                    st.info(f"⏱️ 推論時間: {inference_time:.4f}秒")
                    
                    # パフォーマンス記録
                    st.session_state.performance_monitor.record_inference(
                        inference_time, len(audio) / sr
                    )
                    
                    # 履歴に追加
                    if not hasattr(st.session_state, 'recognized_text'):
                        st.session_state.recognized_text = []
                    st.session_state.recognized_text.append(text)
                    
                    # 一時ファイル削除
                    os.unlink(temp_path)
                    
                except Exception as e:
                    st.error(f"❌ 音声認識に失敗しました: {str(e)}")
        
        # 認識結果の履歴
        if hasattr(st.session_state, 'recognized_text') and st.session_state.recognized_text:
            st.subheader("📝 認識履歴")
            for i, text in enumerate(st.session_state.recognized_text):
                st.write(f"{i+1}. {text}")

with tab5:
    st.header("📈 結果分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 学習曲線")
        
        if st.session_state.training_history['loss']:
            # 学習曲線のプロット
            df = pd.DataFrame(st.session_state.training_history)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Training Loss', 'Training WER'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['loss'], name='Loss'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['wer'], name='WER'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ 学習履歴がありません。学習を実行してください。")
    
    with col2:
        st.subheader("⚡ パフォーマンス統計")
        
        stats = st.session_state.performance_monitor.get_statistics()
        if stats:
            # パフォーマンス指標の表示
            st.metric("総推論回数", stats['total_inferences'])
            st.metric("平均推論時間", f"{stats['avg_inference_time']:.4f}s")
            st.metric("推論時間標準偏差", f"{stats['std_inference_time']:.4f}s")
            st.metric("平均リアルタイム比", f"{stats['avg_realtime_ratio']:.2f}x")
            st.metric("最小リアルタイム比", f"{stats['min_realtime_ratio']:.2f}x")
            st.metric("最大リアルタイム比", f"{stats['max_realtime_ratio']:.2f}x")
            
            # パフォーマンス詳細ボタン
            if st.button("📊 詳細統計を表示"):
                st.session_state.performance_monitor.print_statistics()
        else:
            st.info("ℹ️ パフォーマンスデータがありません。リアルタイム認識を実行してください。")
    
    # モデル情報
    if st.session_state.model:
        st.subheader("🤖 モデル情報")
        
        model_manager = ModelManager()
        
        col1, col2 = st.columns(2)
        
        with col1:
            params = sum(p.numel() for p in st.session_state.model.parameters())
            trainable_params = sum(p.numel() for p in st.session_state.model.parameters() if p.requires_grad)
            model_size_mb = sum(p.numel() * p.element_size() for p in st.session_state.model.parameters()) / (1024 * 1024)
            
            st.metric("総パラメータ数", f"{params:,}")
            st.metric("学習可能パラメータ数", f"{trainable_params:,}")
            st.metric("モデルサイズ", f"{model_size_mb:.2f}MB")
        
        with col2:
            # 保存されたモデルの一覧
            saved_models = model_manager.list_models()
            if saved_models:
                st.write("💾 保存されたモデル:")
                for model_file in saved_models:
                    st.write(f"- {model_file}")
            else:
                st.info("ℹ️ 保存されたモデルがありません。")

with tab6:
    st.header("🚀 学習制御")
    
    if not st.session_state.model:
        st.warning("⚠️ まずモデルを初期化してください。")
    else:
        st.subheader("🎛️ 学習制御パネル")
        
        # 学習状態の表示
        if st.session_state.controlled_trainer:
            status = st.session_state.controlled_trainer.get_training_status()
            
            # リアルタイム更新
            if st.button("🔄 状態更新"):
                # ページを更新
                st.markdown(
                    """
                    <script>
                        window.location.reload();
                    </script>
                    """,
                    unsafe_allow_html=True
                )
            
            # 学習状態の詳細表示
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "学習状態", 
                    "🟢 実行中" if status["is_training"] else "🔴 停止中",
                    delta="一時停止中" if status["is_paused"] else ""
                )
            
            with col2:
                st.metric(
                    "現在のエポック", 
                    f"{status['current_epoch'] + 1}/{status['max_epochs']}",
                    delta=f"バッチ {status['current_batch']}"
                )
            
            with col3:
                st.metric(
                    "ベスト損失", 
                    f"{status['best_val_loss']:.4f}",
                    delta=f"エポック {status['best_epoch'] + 1}"
                )
            
            with col4:
                progress = (status['current_epoch'] + 1) / status['max_epochs'] * 100
                st.metric(
                    "進捗", 
                    f"{progress:.1f}%",
                    delta=f"残り {status['max_epochs'] - (status['current_epoch'] + 1)} エポック"
                )
        
        # 学習制御ボタン
        st.subheader("🎮 制御ボタン")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ 学習開始", type="primary", use_container_width=True):
                if st.session_state.controlled_trainer:
                    result = st.session_state.controlled_trainer.start_training()
                    st.success(result["message"])
                else:
                    st.error("❌ トレーナーが初期化されていません")
        
        with col2:
            if st.button("⏸️ 一時停止", use_container_width=True):
                if st.session_state.controlled_trainer:
                    result = st.session_state.controlled_trainer.pause_training()
                    st.info(result["message"])
                else:
                    st.error("❌ トレーナーが初期化されていません")
        
        with col3:
            if st.button("▶️ 再開", use_container_width=True):
                if st.session_state.controlled_trainer:
                    result = st.session_state.controlled_trainer.resume_training()
                    st.success(result["message"])
                else:
                    st.error("❌ トレーナーが初期化されていません")
        
        with col4:
            if st.button("⏹️ 停止", use_container_width=True):
                if st.session_state.controlled_trainer:
                    result = st.session_state.controlled_trainer.stop_training()
                    st.warning(result["message"])
                else:
                    st.error("❌ トレーナーが初期化されていません")
        
        # チェックポイント管理
        st.subheader("💾 チェックポイント管理")
        
        if st.session_state.controlled_trainer:
            checkpoints = st.session_state.controlled_trainer.get_available_checkpoints()
            
            if checkpoints:
                selected_checkpoint = st.selectbox(
                    "チェックポイントを選択",
                    checkpoints,
                    key="checkpoint_select_tab4",
                    help="読み込むチェックポイントを選択してください"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 チェックポイント読み込み", use_container_width=True):
                        checkpoint_path = os.path.join("models", selected_checkpoint)
                        result = st.session_state.controlled_trainer.load_checkpoint(checkpoint_path)
                        st.success(result["message"])
                
                with col2:
                    if st.button("💾 現在の状態を保存", use_container_width=True):
                        result = st.session_state.controlled_trainer.save_checkpoint()
                        st.success(result["message"])
                
                # チェックポイント一覧
                st.write("📋 利用可能なチェックポイント:")
                for i, checkpoint in enumerate(checkpoints[-5:]):  # 最新5個を表示
                    st.write(f"{i+1}. {checkpoint}")
            else:
                st.info("ℹ️ 利用可能なチェックポイントがありません")
        
        # リアルタイム学習進捗
        if st.session_state.controlled_trainer and status.get("is_training", False):
            st.subheader("📈 リアルタイム学習進捗")
            
            # 学習曲線の表示
            if status.get("train_losses"):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # 損失曲線
                ax1.plot(status["train_losses"], label='Train Loss', color='blue')
                if status.get("val_losses"):
                    ax1.plot(status["val_losses"], label='Val Loss', color='red')
                ax1.set_title('Training and Validation Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
                
                # WER曲線
                ax2.plot(status["train_wers"], label='Train WER', color='green')
                if status.get("val_wers"):
                    ax2.plot(status["val_wers"], label='Val WER', color='orange')
                ax2.set_title('Training and Validation WER')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('WER')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # 最新の学習結果
                if status["train_losses"]:
                    latest_epoch = len(status["train_losses"])
                    latest_loss = status["train_losses"][-1]
                    latest_wer = status["train_wers"][-1] if status["train_wers"] else 0.0
                    
                    st.write(f"📊 最新の学習結果 (エポック {latest_epoch}):")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("損失", f"{latest_loss:.4f}")
                    with col2:
                        st.metric("WER", f"{latest_wer:.4f}")
        
        # 学習設定
        st.subheader("⚙️ 学習設定")
        
        if st.session_state.controlled_trainer:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**現在の設定:**")
                st.write(f"- 学習率: {st.session_state.controlled_trainer.optimizer.param_groups[0]['lr']:.6f}")
                st.write(f"- 最大エポック数: {st.session_state.controlled_trainer.max_epochs}")
                st.write(f"- デバイス: {st.session_state.controlled_trainer.device}")
            
            with col2:
                st.write("**モデル情報:**")
                params = sum(p.numel() for p in st.session_state.model.parameters())
                st.write(f"- パラメータ数: {params:,}")
                st.write(f"- モデルタイプ: {st.session_state.model.__class__.__name__}")
                st.write(f"- 学習可能パラメータ: {sum(p.numel() for p in st.session_state.model.parameters() if p.requires_grad):,}")

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 リアルタイム音声認識モデル学習システム | 光速でほぼリアルタイム動作</p>
    </div>
    """,
    unsafe_allow_html=True
)
