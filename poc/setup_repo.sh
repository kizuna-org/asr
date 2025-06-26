#!/bin/bash

# --- 設定 ---
# このスクリプトは、Docker上のGitサーバーに初期リポジトリとJenkinsfileを作成します。
#
# compose.yamlで設定した値に合わせてください。
GIT_USER="poc_user"
GIT_PASS="poc_password"
REPO_NAME="my-poc-repo.git"
GIT_SERVER_URL="http://localhost:8081/git"
BRANCH_NAME="main"

# 一時的な作業ディレクトリ
WORK_DIR="temp_repo_work"

# --- スクリプト本体 ---

# スクリプトのいずれかのコマンドでエラーが発生したら処理を中断する
set -e

echo "--- リポジトリセットアップを開始します ---"

# 作業ディレクトリが既に存在すれば削除
if [ -d "$WORK_DIR" ]; then
  echo "古い作業ディレクトリ ($WORK_DIR) を削除します。"
  rm -rf "$WORK_DIR"
fi

# 作業ディレクトリを作成
mkdir "$WORK_DIR"
cd "$WORK_DIR"

# 1. リポジトリをクローン
echo "リポジトリ ($REPO_NAME) をクローンしています..."
git clone "http://${GIT_USER}:${GIT_PASS}@${GIT_SERVER_URL#http://}/${REPO_NAME}"

# 2. Jenkinsfileを作成
cd "${REPO_NAME%.git}" # ".git"拡張子を除いたディレクトリ名に移動
echo "Jenkinsfileを作成しています..."

cat << 'EOF' > Jenkinsfile
pipeline {
    agent any
    stages {
        stage('Run from Script') {
            steps {
                echo "This repository was set up by the setup_repo.sh script!"
                echo "This pipeline is for branch: ${env.BRANCH_NAME}"
            }
        }
    }
}
EOF

# 3. コミットとプッシュ
echo "変更をコミットし、プッシュしています..."
git add Jenkinsfile
git commit -m "Initial setup by script"
git push origin "$BRANCH_NAME"

# 4. クリーンアップ
cd ../..
rm -rf "$WORK_DIR"

echo "--- セットアップが正常に完了しました ---"
echo "リポジトリ名: $REPO_NAME"
echo "ブランチ名: $BRANCH_NAME"
