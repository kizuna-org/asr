#!/bin/bash
# asr-test/backend, frontend をローカル環境で起動するスクリプト
# 必要なPythonパッケージのインストールや環境変数の設定も行います

set -e

# backend
cd "$(dirname "$0")/backend"
echo "[asr-test] backend: venv作成・有効化・依存インストール"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# lint対応: 変数代入とexportを分離
PYTHONPATH=$(pwd)/app
export PYTHONPATH
CONFIG_PATH=$(pwd)/config.yaml
export CONFIG_PATH
# backend起動（例: main.py）
echo "[asr-test] backend: サーバー起動 (バックグラウンド)"
nohup python app/main.py &
BACKEND_PID=$!
cd ..

# frontend
cd frontend
echo "[asr-test] frontend: venv作成・有効化・依存インストール"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# frontend起動（例: app.py）
echo "[asr-test] frontend: サーバー起動 (バックグラウンド)"
nohup python app.py &
FRONTEND_PID=$!
cd ..


# 終了時にプロセスをkill
trap 'kill $BACKEND_PID $FRONTEND_PID' EXIT

echo "[asr-test] backend(PID=$BACKEND_PID), frontend(PID=$FRONTEND_PID) を起動しました。"
echo "終了するには Ctrl+C を押してください。"
wait
