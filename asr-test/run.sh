#!/bin/bash

# これは開発PCで実行し、サーバーにデプロイするためのスクリプトです。

# --- 設定項目 ---
SSH_HOST="edu-gpu"
# ControlMasterで使うソケットファイルのパス
CONTROL_PATH="/tmp/ssh-master-${SSH_HOST}-%r@%h:%p"
PORTS_TO_FORWARD=(58080 58081)

# --- 関数定義 ---

# マスターセッションを開始する関数
start_ssh_master_session() {
    echo "SSHマスターセッションを開始します..."
    # -f: バックグラウンド実行, -n:標準入力をリダイレクトしない, -N:リモートコマンドを実行しない
    # -M: マスターモード, -S: コントロールソケットのパス
    ssh -f -n -N -M -S "${CONTROL_PATH}" "${SSH_HOST}"
    
    # セッションが確立されたか確認
    if ! ssh -S "${CONTROL_PATH}" -O check "${SSH_HOST}" >/dev/null 2>&1; then
        echo "❌ エラー: SSHマスターセッションの確立に失敗しました。"
        echo "   'ssh ${SSH_HOST}' を手動で実行して接続できるか確認してください。"
        exit 1
    fi
    echo "✅ SSHマスターセッションが確立されました。"
}

# ポート転送を実行する関数
forward_port() {
    local port=$1
    echo "ポート${port}の転送を開始します..."
    # -O forward: 既存のマスターセッションを使ってポート転送を追加
    ssh -S "${CONTROL_PATH}" -O forward -L "${port}:localhost:${port}" "${SSH_HOST}"
    
    # 転送が成功したか確認
    if ! lsof -ti:"${port}" > /dev/null 2>&1; then
        echo "❌ エラー: ポート${port}の転送に失敗しました。"
        cleanup # 失敗したらクリーンアップ
        exit 1
    fi
    echo "✅ ポート${port}は正常に転送されています (http://localhost:${port})"
}

# クリーンアップ関数
cleanup() {
    echo "クリーンアップを実行しています..."
    # マスター接続が存在すれば終了させる
    if ssh -S "${CONTROL_PATH}" -O check "${SSH_HOST}" >/dev/null 2>&1; then
        echo "SSHマスターセッションを終了します..."
        ssh -S "${CONTROL_PATH}" -O exit "${SSH_HOST}"
    fi

    # lsofでポートを強制的に解放 (念のため)
    for port in "${PORTS_TO_FORWARD[@]}"; do
        if lsof -ti:"${port}" > /dev/null 2>&1; then
            echo "ポート${port}のプロセスを終了します..."
            lsof -ti:"${port}" | xargs kill -9 2>/dev/null || true
        fi
    done
    echo "クリーンアップが完了しました。"
}

# --- メイン処理 ---

# 終了時に必ずクリーンアップが実行されるように設定
trap cleanup EXIT INT TERM

# 最初に既存の接続をクリーンアップ
cleanup
sleep 1

# [STEP 1] rsync, docker build, up などのデプロイ処理
echo "rsyncでファイルをサーバーにコピーします。"
rsync -avz \
  --exclude='__pycache__/' \
  --exclude='models/' \
  ./ ${SSH_HOST}:/home/students/r03i/r03i18/asr-test/asr/asr-test

echo "コンテナを停止します。"
ssh ${SSH_HOST} "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose down"

echo "イメージをビルドします。"
ssh ${SSH_HOST} "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker build . -t asr-app"

echo "コンテナを起動します。"
ssh ${SSH_HOST} "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose up -d"

echo "デプロイが完了しました。"
echo ""

# [STEP 2] SSHマスターセッションを開始
start_ssh_master_session

# [STEP 3] 各ポートの転送を開始
for port in "${PORTS_TO_FORWARD[@]}"; do
    forward_port "${port}"
done

echo ""
echo "全てのポート転送が完了しました。"
echo "アプリケーションにアクセスする準備ができました。"
echo "このスクリプトを終了する（Ctrl+C）と、ポート転送も自動的に停止します。"
echo ""

# マスターセッションが終了するまで待機
echo "ポート転送を実行中です。停止するには Ctrl+C を押してください..."
# -O check でマスター接続が生きているか監視
while ssh -S "${CONTROL_PATH}" -O check "${SSH_HOST}" >/dev/null 2>&1; do
    sleep 5
done

echo "ポート転送セッションが終了しました。"
