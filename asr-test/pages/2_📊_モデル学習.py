import streamlit as st
import os
import sys

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.trainer import ASRTrainer, FastTrainer
from app.controlled_trainer import ControlledASRTrainer

st.title("📊 モデル学習")

st.markdown("""
このページでは、音声認識モデルの学習を実行できます。
""")

# 学習設定
st.subheader("学習設定")

col1, col2 = st.columns(2)

with col1:
    epochs = st.number_input("エポック数", min_value=1, max_value=100, value=10)
    batch_size = st.number_input("バッチサイズ", min_value=1, max_value=128, value=32)
    
with col2:
    learning_rate = st.number_input("学習率", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    model_type = st.selectbox("モデルタイプ", ["LightweightASR", "FastASR"])

# 学習開始
if st.button("🚀 学習を開始"):
    st.info("学習機能は準備中です...")
    
    # ここに学習のロジックを実装
    # ASRTrainer, FastTrainer, ControlledASRTrainer クラスを使用
