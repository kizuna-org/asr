FROM docker:dind

EXPOSE 22

# docker-compose-plugin をパッケージリストに追加
RUN apk add --no-cache \
  git \
  bash \
  jq \
  python3 \
  docker-cli-compose

ARG PUB_KEY

# SSHユーザー設定 (変更なし)
RUN if [ -z "${PUB_KEY}" ]; then echo "Error: PUB_KEY build-arg is not set"; exit 1; fi && \
  mkdir -p /home/ansible_user/.ssh && \
  echo "${PUB_KEY}" > /home/ansible_user/.ssh/authorized_keys

RUN apk add --no-cache openssh sudo
RUN ssh-keygen -A
RUN adduser -D -h /home/ansible_user ansible_user
RUN passwd -u ansible_user
RUN chown ansible_user:ansible_user /etc/ssh/ssh_host_*_key

RUN chown -R ansible_user:ansible_user /home/ansible_user && \
  chmod 700 /home/ansible_user/.ssh && \
  chmod 600 /home/ansible_user/.ssh/authorized_keys

# sudo設定 (変更なし)
RUN echo "ansible_user ALL=(ALL) NOPASSWD: /usr/local/bin/docker" > /etc/sudoers.d/ansible-docker && \
  chmod 0440 /etc/sudoers.d/ansible-docker

# SSHサーバー設定 (変更なし)
RUN sed -i 's/^#?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config && \
  sed -i 's/^#?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
  echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config

# dindのイメージ自体にdind内で使うイメージをキャッシュしておくためにここでビルド
COPY --link ./for-cache/ /home/ansible_user/for-cache/
RUN chown -R ansible_user:ansible_user /home/ansible_user/for-cache
RUN chmod -R 755 /home/ansible_user/for-cache
RUN ls -al /home/ansible_user/for-cache
RUN set -eux; \
  # dockerdをバックグラウンドで起動
  dockerd-entrypoint.sh & \
  DOCKER_PID=$!; \
  # dockerdが起動するまで待機 (最大30秒)
  timeout=60; \
  while ! docker info >/dev/null 2>&1; do \
  if [ "$timeout" -le 0 ]; then \
  echo "Timed out waiting for Docker daemon to start"; \
  kill $DOCKER_PID; \
  exit 1; \
  fi; \
  sleep 1; \
  timeout=$((timeout - 1)); \
  done; \
  # ビルドスクリプトを実行 (build時はroot権限で実行)
  cd /home/ansible_user/for-cache && ./docker_pre_build.sh; \
  # 一時的に起動したdockerdを停止
  kill $DOCKER_PID; \
  wait $DOCKER_PID

# 起動コマンド (変更なし)
CMD ["sh", "-c", "/usr/sbin/sshd && exec dockerd-entrypoint.sh dockerd"]
