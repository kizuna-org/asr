#!/bin/bash

# リモートサーバー
REMOTE_HOST="edu-gpu"
CONTAINER_NAME="squid"
PCAP_PATH="/tmp/squid_capture.pcap"
LOCAL_PATH="./squid_capture.pcap"

# 1. リモートでtcpdumpを実行（例: 10秒間キャプチャ）
ssh "$REMOTE_HOST" "sudo docker compose exec $CONTAINER_NAME tcpdump -i any -w $PCAP_PATH -c 1000"

# 2. リモートからホストにpcapファイルをコピー
ssh "$REMOTE_HOST" "sudo docker cp $CONTAINER_NAME:$PCAP_PATH $PCAP_PATH"
scp "$REMOTE_HOST:$PCAP_PATH" "$LOCAL_PATH"

# 3. リモートの一時ファイルを削除（任意）
ssh "$REMOTE_HOST" "rm -f $PCAP_PATH"
