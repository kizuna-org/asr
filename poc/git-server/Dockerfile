# git-server/Dockerfile

FROM alpine:latest

# Apache, git, htpasswdコマンド、bashをインストール
RUN apk add --no-cache apache2 apache2-utils git bash

# Apacheの設定ファイルと起動スクリプトをコンテナにコピー
COPY git.conf /etc/apache2/conf.d/git.conf
COPY entrypoint.sh /entrypoint.sh

# 起動スクリプトに実行権限を付与
RUN chmod +x /entrypoint.sh

# コンテナの80番ポートを公開
EXPOSE 80

# コンテナ起動時にentrypoint.shを実行
ENTRYPOINT ["/entrypoint.sh"]
