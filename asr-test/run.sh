#!/bin/bash

# これは開発PCで実行し、サーバーにデプロイするためのスクリプトです。

# 関数定義
cleanup() {
    echo "クリーンアップを実行しています..."
    
    # 既存のポート転送プロセスを終了
    if lsof -ti:8080 > /dev/null 2>&1; then
        echo "ポート8080の既存プロセスを終了します..."
        lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    fi
    
    if lsof -ti:8081 > /dev/null 2>&1; then
        echo "ポート8081の既存プロセスを終了します..."
        lsof -ti:8081 | xargs kill -9 2>/dev/null || true
    fi
    
    # バックグラウンドプロセスを終了
    if [ ! -z "$PORT_FORWARD_PID1" ] && kill -0 $PORT_FORWARD_PID1 2>/dev/null; then
        kill $PORT_FORWARD_PID1 2>/dev/null || true
    fi
    
    if [ ! -z "$PORT_FORWARD_PID2" ] && kill -0 $PORT_FORWARD_PID2 2>/dev/null; then
        kill $PORT_FORWARD_PID2 2>/dev/null || true
    fi
    
    echo "クリーンアップが完了しました。"
}

# 終了時のクリーンアップ関数
final_cleanup() {
    cleanup
    exit 0
}

# シグナルハンドラーを設定
trap final_cleanup EXIT INT TERM

echo "rsyncでファイルをサーバーにコピーします。"
rsync -avz \
  --exclude='__pycache__/' \
  --exclude='models/' \
  ./ edu-gpu:/home/students/r03i/r03i18/asr-test/asr/asr-test

echo "コンテナを停止します。"
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose down"

echo "イメージをビルドします。"
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker build . -t asr-app"

echo "コンテナを起動します。"
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose up -d"

echo "デプロイが完了しました。"

echo ""
echo "ポート転送を開始します..."
echo "Ctrl+Cで停止するまで、以下のポートでアクセスできます："
echo "  - Streamlit: http://localhost:8080"
echo "  - API: http://localhost:8081"
echo ""
echo "ポート転送を停止するには Ctrl+C を押してください。"
echo ""

# 既存のポート転送プロセスをクリーンアップ
cleanup

# 少し待機してから新しいポート転送を開始
sleep 2

# ポート転送をバックグラウンドで開始し、プロセスIDを記録
echo "ポート8080の転送を開始します..."
ssh -f -N -L 8080:localhost:58080 edu-gpu &
PORT_FORWARD_PID1=$!

echo "ポート8081の転送を開始します..."
ssh -f -N -L 8081:localhost:58081 edu-gpu &
PORT_FORWARD_PID2=$!

# プロセスが正常に開始されたか確認
sleep 2
if ! kill -0 $PORT_FORWARD_PID1 2>/dev/null; then
    echo "エラー: ポート8080の転送プロセスが開始できませんでした。"
    exit 1
fi

if ! kill -0 $PORT_FORWARD_PID2 2>/dev/null; then
    echo "エラー: ポート8081の転送プロセスが開始できませんでした。"
    kill $PORT_FORWARD_PID1 2>/dev/null || true
    exit 1
fi

echo "ポート転送が開始されました。"
echo "アプリケーションにアクセスする準備ができました。"
echo ""
echo "停止するには Ctrl+C を押してください。"

# ユーザーがCtrl+Cを押すまで待機
echo "ポート転送が実行中です。Ctrl+Cで停止してください..."
while true; do
    # プロセスが生きているかチェック
    if ! kill -0 $PORT_FORWARD_PID1 2>/dev/null || ! kill -0 $PORT_FORWARD_PID2 2>/dev/null; then
        echo "ポート転送プロセスが予期せず終了しました。"
        break
    fi
    sleep 5
done
