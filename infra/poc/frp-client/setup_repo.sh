#!/bin/bash
# setup_repo.sh (Gitea版)

set -e

# --- Giteaで作成したユーザー名、パスワード、リポジトリ名に合わせて変更してください ---
GITEA_USER="poc_user"
GITEA_PASS="poc_password" # Giteaで設定したパスワード
REPO_NAME="my-poc-repo"   # Giteaで作成したリポジトリ名 (例: my-poc-repo.gitではない)
# ---------------------------------------------------------------------------------

GITEA_URL="http://localhost:3000/${GITEA_USER}/${REPO_NAME}.git"
WORK_DIR="temp_gitea_work"

echo "--- GiteaリポジトリにJenkinsfileをプッシュします ---"

rm -rf "$WORK_DIR"
git clone "http://${GITEA_USER}:${GITEA_PASS}@${GITEA_URL#http://}" "$WORK_DIR"

cd "$WORK_DIR"

cat << 'EOF' > Jenkinsfile
pipeline {
    agent any
    stages {
        stage('Success with Gitea!') {
            steps {
                echo "Successfully cloned from the Gitea server!"
                echo "This is the final, working solution."
            }
        }
    }
}
EOF

git add Jenkinsfile
git commit -m "Add Jenkinsfile"
git push origin main

cd ..
rm -rf "$WORK_DIR"

echo "--- 正常にプッシュが完了しました ---"
