#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP_DIR="$SCRIPT_DIR/whaled"

(
  "$SCRIPT_DIR/infrastructure/docker_build.sh" --no-sudo -f "$APP_DIR/app/Dockerfile" "$APP_DIR" \
        --build-arg HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080" \
        --build-arg HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080" \
        -t whaled-app-subscriber && \
        docker save -o "$SCRIPT_DIR/test/host/whaled-app-subscriber.tar" whaled-app-subscriber
) &

(
  "$SCRIPT_DIR/infrastructure/docker_build.sh" --no-sudo -f "$APP_DIR/build/Dockerfile" "$APP_DIR" \
        --build-arg HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080" \
        --build-arg HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080" \
        -t whaled-build-subscriber && \
        docker save -o "$SCRIPT_DIR/test/host/whaled-build-subscriber.tar" whaled-build-subscriber
) &

wait
