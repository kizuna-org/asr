#!/bin/bash

# --- 設定 ---
GIT_USER="poc_user"
GIT_PASS="poc_password"
REPO_NAME="my-poc-repo.git"
GIT_SERVER_HOST_URL="http://localhost:8081/git" # ホストからアクセスするURL
BRANCH_NAME="main"

# ホスト上のGitデータディレクトリとbareリポジトリのパス
HOST_GIT_DATA_DIR="./git-data"
BARE_REPO_PATH="${HOST_GIT_DATA_DIR}/${REPO_NAME}"

# 一時的な作業ディレクトリ
WORK_DIR="temp_repo_work"

# --- スクリプト本体 ---

set -e

echo "--- リポジトリセットアップを開始します ---"

# 1. bareリポジトリをホスト上に作成
if [ -d "$BARE_REPO_PATH" ]; then
    echo "Bareリポジトリは既に存在します: $BARE_REPO_PATH"
else
    echo "Bareリポジトリをホスト上に作成します: $BARE_REPO_PATH"
    git init --bare "$BARE_REPO_PATH"
fi

# 2. ローカル作業ディレクトリの準備
if [ -d "$WORK_DIR" ]; then
  echo "古い作業ディレクトリ ($WORK_DIR) を削除します。"
  rm -rf "$WORK_DIR"
fi
mkdir "$WORK_DIR"
cd "$WORK_DIR"

# 3. ローカルリポジトリを初期化
echo "ローカルリポジトリを初期化しています..."
git init
git branch -m "$BRANCH_NAME" # デフォルトブランチをmainに設定

# 4. Jenkinsfileを作成
echo "Jenkinsfileを作成しています..."
cat << 'EOF' > Jenkinsfile
pipeline {
    agent any
    stages {
        stage('Run from Corrected Script') {
            steps {
                echo "Repository was correctly initialized by the script!"
                echo "This pipeline is for branch: ${env.BRANCH_NAME}"
            }
        }
    }
}
EOF

# 5. コミットとプッシュ
echo "変更をコミットし、リモートにプッシュしています..."
git add Jenkinsfile
git commit -m "Initial commit after bare repo creation"

# リモートリポジトリを登録
git remote add origin "http://${GIT_USER}:${GIT_PASS}@${GIT_SERVER_HOST_URL#http://}/${REPO_NAME}"

# リモートにプッシュ
git push -u origin "$BRANCH_NAME"

# 6. クリーンアップ
cd ..
rm -rf "$WORK_DIR"

echo "--- セットアップが正常に完了しました ---"
