import streamlit as st
import torch
import numpy as np
import os
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ローカルモジュールのインポート
from model import LightweightASRModel, FastASRModel, CHAR_TO_ID, ID_TO_CHAR
from dataset import AudioPreprocessor, TextPreprocessor, ASRDataset, create_dataloader, SyntheticDataset
from trainer import ASRTrainer, FastTrainer
from utils import (
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
    st.subheader("学習パラメータ")
    learning_rate = st.slider("学習率", 1e-5, 1e-2, 1e-3, format="%.5f")
    batch_size = st.slider("バッチサイズ", 1, 32, 8)
    max_epochs = st.slider("最大エポック数", 10, 200, 50)
    
    # モデルパラメータ
    st.subheader("モデルパラメータ")
    hidden_dim = st.slider("隠れ層サイズ", 32, 256, 64 if model_type.startswith("Fast") else 128)
    num_layers = st.slider("LSTM層数", 1, 4, 1 if model_type.startswith("Fast") else 2)

# メインコンテンツ
tab1, tab2, tab3, tab4 = st.tabs(["🏠 ホーム", "🎯 モデル学習", "🎤 リアルタイム認識", "📊 結果分析"])

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
    
    # モデル初期化
    if st.button("🔄 モデルを初期化"):
        with st.spinner("モデルを初期化中..."):
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
    
    # データセット準備
    st.subheader("📁 データセット準備")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎵 サンプルデータセット生成"):
            with st.spinner("サンプルデータセットを生成中..."):
                samples = create_sample_audio_data(num_samples=20, duration=3.0)
                save_sample_dataset(samples, "data/raw")
                st.success(f"✅ {len(samples)}個のサンプルデータを生成しました！")
    
    with col2:
        uploaded_files = st.file_uploader(
            "音声ファイルをアップロード",
            type=['wav', 'mp3', 'flac', 'm4a'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)}個のファイルがアップロードされました")
    
    # 学習実行
    if st.session_state.model and st.session_state.audio_preprocessor:
        st.subheader("🚀 学習実行")
        
        # データセットの読み込み
        data_dir = "data/raw"
        if os.path.exists(data_dir) and os.listdir(data_dir):
            try:
                dataset = ASRDataset(
                    data_dir=data_dir,
                    audio_preprocessor=st.session_state.audio_preprocessor,
                    text_preprocessor=st.session_state.text_preprocessor
                )
                
                if len(dataset) > 0:
                    st.success(f"✅ データセット読み込み完了: {len(dataset)}サンプル")
                    
                    # データローダーの作成
                    train_loader = create_dataloader(
                        dataset, 
                        batch_size=batch_size, 
                        shuffle=True
                    )
                    
                    # トレーナーの初期化
                    if model_type.startswith("Fast"):
                        st.session_state.trainer = FastTrainer(
                            model=st.session_state.model,
                            train_loader=train_loader,
                            device=device,
                            learning_rate=learning_rate,
                            max_epochs=max_epochs,
                            model_save_dir="models"
                        )
                    else:
                        st.session_state.trainer = ASRTrainer(
                            model=st.session_state.model,
                            train_loader=train_loader,
                            device=device,
                            learning_rate=learning_rate,
                            max_epochs=max_epochs,
                            model_save_dir="models"
                        )
                    
                    # 学習開始ボタン
                    if st.button("🎯 学習開始", type="primary"):
                        st.subheader("📈 学習進捗")
                        
                        # プログレスバー
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 学習履歴の表示用
                        chart_placeholder = st.empty()
                        
                        # 学習実行
                        try:
                            history = st.session_state.trainer.train()
                            
                            # 学習履歴を更新
                            st.session_state.training_history = {
                                'loss': history['train_losses'],
                                'wer': history['train_wers'],
                                'epoch': list(range(1, len(history['train_losses']) + 1))
                            }
                            
                            st.success("✅ 学習が完了しました！")
                            
                        except Exception as e:
                            st.error(f"❌ 学習中にエラーが発生しました: {str(e)}")
                
                else:
                    st.warning("⚠️ データセットが空です。サンプルデータを生成するか、音声ファイルをアップロードしてください。")
            
            except Exception as e:
                st.error(f"❌ データセットの読み込みに失敗しました: {str(e)}")
        else:
            st.info("ℹ️ データディレクトリが空です。サンプルデータを生成するか、音声ファイルをアップロードしてください。")

with tab3:
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

with tab4:
    st.header("📊 結果分析")
    
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
