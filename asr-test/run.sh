#!/bin/bash

# これは開発PCで実行し、サーバーにデプロイするためのスクリプトです。

ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/ && git pull"

ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose down"
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker build . -t asr-app"
ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose up -d"

echo "デプロイが完了しました。"

ssh -L5 8080:localhost:58080 edu-gpu &
ssh -L5 8081:localhost:58081 edu-gpu &

ssh edu-gpu "cd /home/students/r03i/r03i18/asr-test/asr/asr-test && sudo docker compose down"
