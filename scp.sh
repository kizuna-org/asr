#!/usr/bin/env bash
set -euo pipefail

ssh edu-gpu "cd ~/temp/poc && sudo docker compose down"
rsync -avz ./poc/ edu-gpu:~/temp/poc/
ssh edu-gpu "cd ~/temp/poc && sudo docker compose down && HTTP_PROXY=http://http-p.srv.cc.suzuka-ct.ac.jp:8080 HTTPS_PROXY=http://http-p.srv.cc.suzuka-ct.ac.jp:8080 sudo docker compose up --build"
