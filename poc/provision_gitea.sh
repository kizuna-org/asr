#!/bin/bash

# --- 設定 ---
GITEA_URL="http://localhost:3000"
GITEA_USER="poc_user"
GITEA_PASS="poc_password"
GITEA_EMAIL="user@example.com"
REPO_NAME="my-poc-repo"

# --- スクリプト本体 ---
set -e

# 1. Giteaが起動するのを待つ
echo "--- Giteaの起動を待っています... ---"
until $(curl --output /dev/null --silent --head --fail ${GITEA_URL}); do
    printf '.'
    sleep 2
done
echo -e "\nGiteaが起動しました。"

# 2. 管理者ユーザーを作成する
echo "--- Giteaユーザー '${GITEA_USER}' を作成しています... ---"
# --- 修正点: --user git を追加 ---
if docker compose exec --user git gitea gitea admin user list | grep -q "${GITEA_USER}"; then
    echo "ユーザー '${GITEA_USER}' は既に存在します。"
else
    # --- 修正点: --user git を追加 ---
    docker compose exec --user git gitea gitea admin user create \
        --username "${GITEA_USER}" \
        --password "${GITEA_PASS}" \
        --email "${GITEA_EMAIL}" \
        --admin \
        --must-change-password=false
    echo "ユーザー '${GITEA_USER}' を作成しました。"
fi

# 3. リポジトリを作成する
echo "--- Giteaリポジトリ '${REPO_NAME}' を作成しています... ---"
REPO_EXISTS=$(curl -u "${GITEA_USER}:${GITEA_PASS}" -s -o /dev/null -w "%{http_code}" "${GITEA_URL}/api/v1/repos/${GITEA_USER}/${REPO_NAME}")

if [ "$REPO_EXISTS" = "200" ]; then
    echo "リポジトリ '${REPO_NAME}' は既に存在します。"
else
    HTTP_STATUS=$(curl -u "${GITEA_USER}:${GITEA_PASS}" -s -o /dev/null -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"name": "'"${REPO_NAME}"'", "private": false}' \
        "${GITEA_URL}/api/v1/user/repos")

    if [ "$HTTP_STATUS" = "201" ]; then
        echo "リポジトリ '${REPO_NAME}' を作成しました。"
    else
        echo "リポジトリの作成に失敗しました。HTTPステータス: ${HTTP_STATUS}"
        exit 1
    fi
fi

echo "--- Giteaのプロビジョニングが完了しました ---"
