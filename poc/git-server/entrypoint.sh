#!/bin/bash
# git-server/entrypoint.sh

set -e

# Gitリポジトリ用のディレクトリを作成
mkdir -p /git

# ユーザー認証ファイル（.htpasswd）の準備
HTPASSWD_FILE="/etc/apache2/git.htpasswd"
# 起動時にファイルを初期化
> "$HTPASSWD_FILE"

echo "Setting up Git users..."
# GIT_USER_1, GIT_PASS_1 のような環境変数をループで処理
for i in {1..10}; do
    user_var="GIT_USER_${i}"
    pass_var="GIT_PASS_${i}"

    # ユーザー名とパスワードの両方が設定されている場合のみ処理
    if [ -n "${!user_var}" ] && [ -n "${!pass_var}" ]; then
        echo "-> Adding user: ${!user_var}"
        # パスワードファイルにユーザーを追加（-B: bcrypt, -b: パスワードを引数で指定）
        htpasswd -B -b "$HTPASSWD_FILE" "${!user_var}" "${!pass_var}"
    fi
done

# ファイルの所有権を設定
# PUID/PGIDが設定されていなければデフォルトで1000を使用
PUID=${PUID:-1000}
PGID=${PGID:-1000}
echo "Setting ownership of /git to ${PUID}:${PGID}"
chown -R "${PUID}:${PGID}" /git

# Apacheサーバーをフォアグラウンドで起動
echo "Starting Apache server..."
exec httpd -D FOREGROUND
