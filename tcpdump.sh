#!/bin/bash

set -e
set -x

# リモートサーバー
REMOTE_HOST="edu-gpu"
CONTAINER_NAME="squid"
PCAP_PATH="/var/spool/squid/squid_capture.pcap"
LOCAL_PATH="./squid_capture.pcap"

# 1. リモートでtcpdumpを実行（例: 1000パケットキャプチャ）
echo "[INFO] ssh $REMOTE_HOST sudo docker compose exec $CONTAINER_NAME tcpdump -i any -w $PCAP_PATH -c 1000"
ssh "$REMOTE_HOST" "sudo docker compose exec -T $CONTAINER_NAME tcpdump -i any -w $PCAP_PATH -c 1000"
# 2. ファイルの存在を確認
echo "[INFO] ssh $REMOTE_HOST sudo docker compose exec $CONTAINER_NAME ls -l $PCAP_PATH"
ssh "$REMOTE_HOST" "sudo docker compose exec $CONTAINER_NAME ls -l $PCAP_PATH"

# 3. リモートからホストにpcapファイルをコピー
echo "[INFO] ssh $REMOTE_HOST sudo docker cp $CONTAINER_NAME:$PCAP_PATH ~/temp/poc/squid_capture.pcap"
ssh "$REMOTE_HOST" "sudo docker cp $CONTAINER_NAME:/var/spool/squid/squid_capture.pcap ~/temp/poc/squid_capture.pcap"
echo "[INFO] scp $REMOTE_HOST:~/temp/poc/squid_capture.pcap $LOCAL_PATH"
scp "edu-gpu:~/temp/poc/squid_capture.pcap" "./squid_capture.pcap"
