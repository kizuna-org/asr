import streamlit as st
import requests
import os
from datetime import datetime
import time
import traceback

# 設定
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = os.getenv("BACKEND_PORT", "58081")
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api"

# プロキシ設定
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")
NO_PROXY = os.getenv("NO_PROXY", "localhost,127.0.0.1,asr-api")

proxies = {}
if HTTP_PROXY:
    proxies["http"] = HTTP_PROXY
if HTTPS_PROXY:
    proxies["https"] = HTTPS_PROXY

def should_use_proxy(url):
    """URLがプロキシを使用すべきかどうかを判定"""
    if not proxies:
        return False
    
    no_proxy_hosts = [host.strip() for host in NO_PROXY.split(",")]
    for host in no_proxy_hosts:
        if host in url:
            return False
    return True

def get_models():
    """学習済みモデル一覧を取得"""
    try:
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.get(f"{BACKEND_URL}/models", timeout=10, proxies=request_proxies)
        
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            st.error(f"モデル一覧の取得に失敗しました: HTTP {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("バックエンドに接続できません。バックエンドサービスが起動しているか確認してください。")
        return []
    except requests.exceptions.Timeout:
        st.error("リクエストがタイムアウトしました。")
        return []
    except Exception as e:
        st.error(f"モデル一覧の取得中にエラーが発生しました: {str(e)}")
        return []

def delete_model(model_name):
    """指定されたモデルを削除"""
    try:
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.delete(f"{BACKEND_URL}/models/{model_name}", timeout=30, proxies=request_proxies)
        
        if response.status_code == 200:
            return True, "モデルが正常に削除されました。"
        else:
            error_detail = response.json().get("detail", "不明なエラー")
            return False, f"削除に失敗しました: {error_detail}"
    except requests.exceptions.ConnectionError:
        return False, "バックエンドに接続できません。"
    except requests.exceptions.Timeout:
        return False, "リクエストがタイムアウトしました。"
    except Exception as e:
        return False, f"削除中にエラーが発生しました: {str(e)}"

def format_file_size(size_mb):
    """ファイルサイズを適切な単位でフォーマット"""
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.1f} GB"

def format_timestamp(timestamp):
    """タイムスタンプを読みやすい形式でフォーマット"""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

# ページ設定
st.set_page_config(
    page_title="モデル管理 - ASR学習ダッシュボード",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🤖 学習済みモデル管理")

# ナビゲーション
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    if st.button("🏠 メインダッシュボード", use_container_width=True):
        st.switch_page("app.py")
with col_nav2:
    if st.button("🤖 モデル管理", use_container_width=True, disabled=True):
        pass  # 現在のページなので無効化
with col_nav3:
    st.markdown("### 📊 現在のページ: モデル管理")
st.markdown("---")

# サイドバー - ナビゲーション
with st.sidebar:
    st.header("📋 ナビゲーション")
    
    # ページ間のナビゲーション
    if st.button("🏠 メインダッシュボード", use_container_width=True):
        st.switch_page("app.py")
    if st.button("🤖 モデル管理", use_container_width=True, disabled=True):
        pass  # 現在のページなので無効化
    
    st.markdown("---")
    st.header("ℹ️ 情報")
    st.markdown("""
    **モデル管理ページ**
    
    このページでは以下の操作が可能です：
    - 学習済みモデルの一覧表示
    - モデルの詳細情報確認
    - 不要なモデルの削除
    
    ⚠️ モデル削除は復元できません。
    """)

# 説明
st.markdown("""
このページでは、学習済みモデルの一覧表示と削除を行うことができます。
モデルは学習完了時に自動的に保存され、ここで管理できます。
""")

# モデル一覧の取得
if st.button("🔄 モデル一覧を更新", type="primary"):
    st.rerun()

models = get_models()

if not models:
    st.info("学習済みモデルが見つかりません。学習を実行してモデルを作成してください。")
else:
    st.success(f"{len(models)}個の学習済みモデルが見つかりました。")
    
    # モデル一覧をテーブル形式で表示
    st.subheader("📋 モデル一覧")
    
    # データフレーム用のデータを準備
    model_data = []
    for model in models:
        model_data.append({
            "モデル名": model["name"],
            "エポック": model["epoch"] if model["epoch"] else "不明",
            "サイズ": format_file_size(model["size_mb"]),
            "ファイル数": model["file_count"],
            "作成日時": format_timestamp(model["created_at"]),
            "パス": model["path"]
        })
    
    # テーブル表示
    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True
    )
    
    # モデル詳細と削除機能
    st.subheader("🗑️ モデル削除")
    st.warning("⚠️ モデルを削除すると復元できません。削除前に十分確認してください。")
    
    # モデル選択
    model_names = [model["name"] for model in models]
    selected_model = st.selectbox(
        "削除するモデルを選択してください:",
        model_names,
        index=None,
        placeholder="モデルを選択..."
    )
    
    if selected_model:
        # 選択されたモデルの詳細を表示
        selected_model_info = next((m for m in models if m["name"] == selected_model), None)
        if selected_model_info:
            st.markdown("### 選択されたモデルの詳細")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("モデル名", selected_model_info["name"])
                st.metric("エポック", selected_model_info["epoch"] if selected_model_info["epoch"] else "不明")
            
            with col2:
                st.metric("サイズ", format_file_size(selected_model_info["size_mb"]))
                st.metric("ファイル数", selected_model_info["file_count"])
            
            with col3:
                st.metric("作成日時", format_timestamp(selected_model_info["created_at"]))
            
            # ファイル一覧
            st.markdown("#### 含まれるファイル:")
            for file_name in selected_model_info["files"]:
                st.text(f"• {file_name}")
            
            # 削除確認
            st.markdown("### 削除確認")
            confirm_text = st.text_input(
                f"削除を確認するために、モデル名 '{selected_model}' を入力してください:",
                placeholder="モデル名を入力..."
            )
            
            # 削除ボタン
            if st.button("🗑️ モデルを削除", type="secondary", disabled=confirm_text != selected_model):
                if confirm_text == selected_model:
                    with st.spinner("モデルを削除中..."):
                        success, message = delete_model(selected_model)
                        
                        if success:
                            st.success(message)
                            st.balloons()
                            # 削除後、ページを更新
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.error("モデル名が一致しません。正確に入力してください。")

# フッター
st.markdown("---")
st.markdown("💡 **ヒント**: モデルを削除する前に、必要に応じてバックアップを取ることをお勧めします。")
