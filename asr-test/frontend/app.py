import streamlit as st
import pandas as pd
import requests
import asyncio
import websockets
import json
from typing import Dict, Any

# --- 設定 ---
BACKEND_URL = "http://localhost:58081/api"
WEBSOCKET_URL = "ws://localhost:58081/ws"

# --- 状態管理の初期化 ---
def init_session_state():
    defaults = {
        "logs": ["ダッシュボードへようこそ！"],
        "progress_data": pd.DataFrame(columns=["step", "loss", "learning_rate"]),
        "is_training": False,
        "available_models": [],
        "available_datasets": [],
        "current_progress": 0,
        "progress_text": "待機中",
        "websocket_task": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- バックエンドAPI通信 --- #
def get_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        response.raise_for_status()
        st.session_state.is_training = response.json().get("is_training", False)
    except requests.RequestException:
        st.session_state.is_training = False # 接続失敗時は学習していないとみなす

def get_config():
    try:
        response = requests.get(f"{BACKEND_URL}/config")
        response.raise_for_status()
        config = response.json()
        st.session_state.available_models = config.get("available_models", [])
        st.session_state.available_datasets = config.get("available_datasets", [])
    except requests.RequestException as e:
        st.error(f"設定の取得に失敗しました: {e}")

def handle_start_training(model: str, dataset: str):
    try:
        response = requests.post(f"{BACKEND_URL}/train/start", json={"model_name": model, "dataset_name": dataset})
        response.raise_for_status()
        st.session_state.is_training = True
        st.session_state.logs.append(f"[INFO] 学習開始リクエスト成功")
        st.rerun()
    except requests.RequestException as e:
        st.error(f"学習の開始に失敗しました: {e.response.json() if e.response else e}")

def handle_stop_training():
    try:
        response = requests.post(f"{BACKEND_URL}/train/stop")
        response.raise_for_status()
        st.session_state.logs.append(f"[INFO] 学習停止リクエスト成功")
    except requests.RequestException as e:
        st.error(f"学習の停止に失敗しました: {e.response.json() if e.response else e}")

def handle_inference(file, model_name):
    files = {"file": (file.name, file.getvalue(), file.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/inference?model_name={model_name}", files=files)
        response.raise_for_status()
        st.success(f"推論結果: {response.json()['transcription']}")
    except requests.RequestException as e:
        st.error(f"推論に失敗しました: {e.response.json() if e.response else e}")

# --- WebSocketリスナー ---
async def websocket_listener():
    st.session_state.logs.append("[WS] WebSocketリスナーを開始します。")
    try:
        async with websockets.connect(WEBSOCKET_URL) as ws:
            st.session_state.logs.append("[WS] WebSocketに接続しました。")
            st.rerun()
            while st.session_state.is_training:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(message)
                    handle_ws_message(data)
                    st.rerun()
                except asyncio.TimeoutError:
                    get_status() # タイムアウトしたらHTTPで状態確認
                except websockets.ConnectionClosed:
                    st.session_state.logs.append("[WS] 接続が切れました。再接続を試みます...")
                    break # ループを抜けて再接続
    except Exception as e:
        st.session_state.logs.append(f"[WS] エラー: {e}")
    st.session_state.is_training = False # 接続が完全に切れたら学習終了とみなす
    st.rerun()

def handle_ws_message(data: Dict[str, Any]):
    type = data.get("type")
    payload = data.get("payload", {})
    if type == "log":
        st.session_state.logs.append(f"[BACKEND] {payload.get('message')}")
    elif type == "progress":
        st.session_state.progress_data.loc[len(st.session_state.progress_data)] = payload
        st.session_state.current_progress = payload['step'] / payload['total_steps']
        st.session_state.progress_text = f"Epoch {payload['epoch']}/{payload['total_epochs']}, Step {payload['step']}/{payload['total_steps']}"
    elif type == "status":
        if payload.get("status") in ["completed", "stopped"]:
            st.session_state.is_training = False
            st.session_state.logs.append(f"[INFO] 学習が終了しました: {payload.get('message')}")
    elif type == "error":
        st.session_state.is_training = False
        st.error(f"学習中にエラーが発生しました: {payload.get('message')}")

# --- UI描画 ---
st.set_page_config(layout="wide")
init_session_state()

# 起動時に一度だけ設定とステータスを取得
if "initial_load" not in st.session_state:
    get_config()
    get_status()
    st.session_state.initial_load = True

# 学習中ならWebSocketリスナーを起動
if st.session_state.is_training and st.session_state.websocket_task is None:
    st.session_state.websocket_task = asyncio.ensure_future(websocket_listener())

st.title("ASR Training Dashboard")

# --- サイドバー ---
st.sidebar.header("Control Panel")
model_name = st.sidebar.selectbox("モデルを選択", st.session_state.available_models, disabled=st.session_state.is_training)
dataset_name = st.sidebar.selectbox("データセットを選択", st.session_state.available_datasets, disabled=st.session_state.is_training)

if st.sidebar.button("学習開始", disabled=st.session_state.is_training):
    handle_start_training(model_name, dataset_name)

if st.sidebar.button("学習停止", disabled=not st.session_state.is_training):
    handle_stop_training()

st.sidebar.header("Status")
st.sidebar.metric("現在のステータス", "学習中" if st.session_state.is_training else "待機中")

# --- メインエリア ---
col1, col2 = st.columns(2)
with col1:
    st.header("学習ロス")
    st.line_chart(st.session_state.progress_data, x="step", y="loss")
with col2:
    st.header("学習率")
    st.line_chart(st.session_state.progress_data, x="step", y="learning_rate")

st.header("進捗")
st.progress(st.session_state.current_progress, text=st.session_state.progress_text)

st.header("ログ")
st.text_area("logs", "\n".join(st.session_state.logs), height=300, key="log_area")

st.header("推論テスト")
uploaded_file = st.file_uploader("音声ファイルをアップロード", type=["wav", "flac", "mp3"], disabled=st.session_state.is_training)
if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)
    if st.button("推論実行", disabled=st.session_state.is_training):
        handle_inference(uploaded_file, model_name)