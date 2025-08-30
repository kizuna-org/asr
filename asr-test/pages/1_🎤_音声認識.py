import streamlit as st
import os
import sys

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.utils import AudioRecorder, RealTimeASR, AudioProcessor

st.title("🎤 リアルタイム音声認識")

st.markdown("""
このページでは、リアルタイム音声認識機能を利用できます。
""")

# 音声認識機能の実装
if st.button("🎤 音声認識を開始"):
    st.info("音声認識機能は準備中です...")
    
    # ここに音声認識のロジックを実装
    # AudioRecorder, RealTimeASR, AudioProcessor クラスを使用
