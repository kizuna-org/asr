import streamlit as st
import gc
import os
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

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
    st.session_state.dataset_info = {}

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
        help="FastASRModelはリアルタイム推論に最適化されています"
    )
    
    # デバイス選択
    device = st.selectbox(
        "デバイス",
        ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
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
        early_stopping_patience = st.slider("Early Stopping Patience", 5, 50, 10)
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
    
    # データセット選択
    st.write("**データセット選択**")
    dataset_selection = st.selectbox(
        "使用するデータセットを選択",
        ["サンプルデータ", "LJSpeechデータセット", "カスタムデータ"],
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
        else:
            st.error("❌ LJSpeechデータセットが見つかりません")
            st.info("ℹ️ サンプルデータを生成するか、カスタムデータをアップロードしてください")
    elif dataset_selection == "カスタムデータ":
        if not st.session_state.dataset_info or st.session_state.dataset_info['type'] != 'custom':
            st.warning("⚠️ カスタムデータをアップロードしてください")
    else:  # サンプルデータ
        if not st.session_state.dataset_info or st.session_state.dataset_info['type'] != 'sample':
            st.info("ℹ️ サンプルデータを生成してください")
    
    # データセット情報の表示
    if st.session_state.dataset_info:
        st.info(f"📊 現在のデータセット: {st.session_state.dataset_info['type']} ({st.session_state.dataset_info['samples']}サンプル)")
    
    # ステップ3: 学習実行
    st.subheader("3️⃣ 学習実行")
    
    if not st.session_state.model:
        st.warning("⚠️ まずモデルを初期化してください。")
    elif not st.session_state.dataset_info:
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
                            train_loader, dataset_info = create_ljspeech_dataloader(
                                data_dir=ljspeech_dir,
                                audio_preprocessor=st.session_state.audio_preprocessor,
                                text_preprocessor=st.session_state.text_preprocessor,
                                batch_size=batch_size
                            )
                            st.success(f"✅ LJSpeechデータセット読み込み完了: {dataset_info['total_samples']}サンプル")
                        else:
                            st.error("❌ LJSpeechデータセットが見つかりません")
                            st.stop()
                    elif st.session_state.dataset_info['type'] == 'custom':
                        # カスタムデータセットの処理
                        custom_path = st.session_state.dataset_info.get('path', 'data/custom')
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
                        early_stopping_patience=early_stopping_patience,
                        validation_split=validation_split
                    )
                    
                    # 学習開始
                    st.session_state.training_start_time = time.time()
                    result = st.session_state.controlled_trainer.start_training()
                    st.success(result["message"])
                    
                except Exception as e:
                    st.error(f"❌ 学習開始に失敗しました: {str(e)}")
        
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
            st.rerun()
    
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
        st.subheader("🎙️ マイク入力")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 録音開始", type="primary"):
                st.session_state.recording = True
                st.session_state.recognized_text = []
                
                # 録音機の初期化
                recorder = AudioRecorder()
                recorder.start_recording()
                
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
        
        with col2:
            if st.button("⏹️ 録音停止"):
                st.session_state.recording = False
                st.success("✅ 録音を停止しました")
        
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
                st.rerun()
            
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
