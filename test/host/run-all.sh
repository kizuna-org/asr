#!/bin/bash

APP_DIR="$HOME/whaled"

# 並列実行する
$APP_DIR/run-app-subscriber.sh &
$APP_DIR/run-build-subscriber.sh &
$APP_DIR/monitor-containers.sh &
