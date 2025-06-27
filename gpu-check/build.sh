#!/bin/bash

# GPU確認ツールのDockerイメージをビルドするスクリプト

set -e

# プロキシ環境変数を設定
export HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
export HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"

IMAGE_NAME="gpu-check"
IMAGE_TAG="latest"

echo "🔨 GPU確認ツールのDockerイメージをビルドしています..."
echo "   イメージ名: ${IMAGE_NAME}:${IMAGE_TAG}"
echo

# カレントディレクトリがgpu-checkかチェック
if [[ ! -f "Dockerfile" || ! -f "check_gpu.py" ]]; then
    echo "❌ エラー: gpu-checkディレクトリで実行してください"
    echo "   必要なファイル: Dockerfile, check_gpu.py, quick_gpu_check.py"
    exit 1
fi

# Dockerイメージをビルド
echo "📦 Dockerイメージをビルドしています..."
if sudo docker build \
    --build-arg HTTP_PROXY="${HTTP_PROXY}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" . ; then
    echo "✅ Dockerイメージのビルドが完了しました"
else
    echo "❌ Dockerイメージのビルドに失敗しました"
    echo "💡 トラブルシューティング:"
    echo "   1. インターネット接続を確認してください"
    echo "   2. プロキシ設定を確認してください"
    echo "   3. Docker デーモンが実行中か確認してください"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo
    echo "✅ ビルド完了!"
    echo
    
    # コンテナの実行
    echo "🚀 コンテナを実行しています..."
    echo
    
    echo
    echo "⚡ 簡易チェックを実行中..."
    if sudo docker run --gpus all --rm "${IMAGE_NAME}:${IMAGE_TAG}" python quick_gpu_check.py ; then
        echo "✅ GPU確認テストが正常に完了しました"
    else
        echo "⚠️ 簡易GPU確認テストでエラーが発生しました"
        echo "🔍 詳細チェックを実行します..."
        echo
        if sudo docker run --gpus all --rm "${IMAGE_NAME}:${IMAGE_TAG}" python check_gpu.py ; then
            echo "✅ 詳細GPU確認テストが正常に完了しました"
        else
            echo "⚠️ 詳細GPU確認テストでもエラーが発生しました（GPU環境がない可能性があります）"
            echo "💡 CPU環境でのテストを実行します..."
            sudo docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python check_gpu.py
        fi
    fi
    
    echo
    echo "✅ すべてのテストが完了しました!"
    echo
         echo "📝 手動で再実行する場合:"
     echo "   詳細チェック:"
     echo "     sudo docker run --gpus all --rm ${IMAGE_NAME}:${IMAGE_TAG}"
     echo
     echo "   簡易チェック:"
     echo "     sudo docker run --gpus all --rm ${IMAGE_NAME}:${IMAGE_TAG} python quick_gpu_check.py"
     echo
     echo "   CPU環境でのテスト:"
     echo "     sudo docker run --rm ${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo "❌ ビルドに失敗しました"
    exit 1
fi
