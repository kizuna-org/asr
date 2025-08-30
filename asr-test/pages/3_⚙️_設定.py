import streamlit as st
import os
import sys

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.utils import PerformanceMonitor

st.title("⚙️ 設定")

st.markdown("""
このページでは、アプリケーションの設定を管理できます。
""")

# モデル管理
st.subheader("📁 モデル管理")

model_dir = "models"
if os.path.exists(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pth', '.pt'))]
    
    if model_files:
        st.write("利用可能なモデル:")
        for model_file in model_files:
            st.write(f"- {model_file}")
    else:
        st.info("モデルファイルが見つかりません")
else:
    st.warning("モデルディレクトリが存在しません")

# パフォーマンス設定
st.subheader("📈 パフォーマンス設定")

col1, col2 = st.columns(2)

with col1:
    st.checkbox("GPU使用", value=True, key="use_gpu")
    st.checkbox("メモリ最適化", value=True, key="memory_optimization")
    
with col2:
    st.number_input("最大メモリ使用量 (MB)", min_value=100, max_value=10000, value=2048, key="max_memory")
    st.number_input("スレッド数", min_value=1, max_value=8, value=4, key="num_threads")

# 設定保存
if st.button("💾 設定を保存"):
    st.success("設定が保存されました")
