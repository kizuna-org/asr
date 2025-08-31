# frontend/app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("ASR Training Dashboard")

# --- サイドバー --- #
st.sidebar.header("Control Panel")

model_name = st.sidebar.selectbox("モデルを選択", ["conformer", "rnn-t"])
dataset_name = st.sidebar.selectbox("データセットを選択", ["ljspeech"])

if st.sidebar.button("学習開始"):
    st.sidebar.success(f"{model_name} の学習を開始しました。")
    # TODO: バックエンドに /train/start APIをコール

if st.sidebar.button("学習停止"):
    st.sidebar.warning("学習を停止します。")
    # TODO: バックエンドに /train/stop APIをコール

st.sidebar.header("Status")
is_training = st.sidebar.toggle("学習中", False)
st.sidebar.metric("現在のステータス", "学習中" if is_training else "待機中")

# --- メインエリア --- #
col1, col2 = st.columns(2)

with col1:
    st.header("学習ロス")
    chart_data = pd.DataFrame(
        np.random.randn(20, 2) / 10 + 1.5, columns=['train_loss', 'val_loss']
    )
    st.line_chart(chart_data)

with col2:
    st.header("学習率")
    lr_chart_data = pd.DataFrame(
        np.random.randn(20, 1) / 1000 + 0.001, columns=['learning_rate']
    )
    st.line_chart(lr_chart_data)

st.header("進捗")
progress_text = "Epoch 5/10, Step 100/200"

st.progress(50, text=progress_text)

st.header("ログ")
st.text_area("logs", "[INFO] Epoch 4 finished.\n[INFO] Starting validation...", height=300)


st.header("推論テスト")
uploaded_file = st.file_uploader("音声ファイルをアップロード", type=["wav", "flac", "mp3"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("推論実行"):
        # TODO: バックエンドに /inference APIをコール
        st.success("推論結果: a piece of cake")
