#!/bin/bash
# git-server/entrypoint.sh (最終版・変更なし)

set -e

# PUID/PGIDが設定されていなければデフォルトで1000を使用
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Apacheの設定ファイル内の実行ユーザーとグループを、環境変数で指定されたPUID/PGIDに書き換える
echo "Configuring Apache to run as UID: ${PUID} and GID: ${PGID}"
sed -i "s/User apache/User #${PUID}/g" /etc/apache2/httpd.conf
sed -i "s/Group apache/Group #${PGID}/g" /etc/apache2/httpd.conf

# ApacheのServerName警告を抑制する
echo "ServerName localhost" >> /etc/apache2/httpd.conf

# Gitリポジトリ用のディレクトリを作成
mkdir -p /git

# ユーザー認証ファイル（.htpasswd）の準備
HTPASSWD_FILE="/etc/apache2/git.htpasswd"
> "$HTPASSWD_FILE"

echo "Setting up Git users..."
for i in {1..10}; do
    user_var="GIT_USER_${i}"
    pass_var="GIT_PASS_${i}"
    if [ -n "${!user_var}" ] && [ -n "${!pass_var}" ]; then
        echo "-> Adding user: ${!user_var}"
        htpasswd -B -b "$HTPASSWD_FILE" "${!user_var}" "${!pass_var}"
    fi
done

# 認証情報ファイルの所有権をApache実行ユーザーに設定し、確実に読み取れるようにする
chown "${PUID}:${PGID}" "$HTPASSWD_FILE"

# Apacheサーバーをフォアグラウンドで起動
echo "Starting Apache server..."
exec httpd -D FOREGROUND
