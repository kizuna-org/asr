# git-server/Dockerfile (確定修正版)

FROM alpine:latest

RUN apk add --no-cache apache2 apache2-utils git bash sed

# --- ここからが確定修正 ---
# ApacheのCGIモジュールを有効化する。これが根本原因の解決策です。
# /etc/apache2/httpd.conf内の該当行のコメント(#)を外す
RUN sed -i 's/^#\(LoadModule cgi_module\)/\1/' /etc/apache2/httpd.conf
# --- ここまでが確定修正 ---


# Apacheの設定ファイルと起動スクリプトをコンテナにコピー
COPY git.conf /etc/apache2/conf.d/git.conf
COPY entrypoint.sh /entrypoint.sh

# 起動スクリプトに実行権限を付与
RUN chmod +x /entrypoint.sh

EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
