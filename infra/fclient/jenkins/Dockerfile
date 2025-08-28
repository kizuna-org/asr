# jenkins/Dockerfile (最終確定版)

FROM jenkins/jenkins:2.516.1-lts

USER root

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG HTTP_PROXY=http://http-p.srv.cc.suzuka-ct.ac.jp:8080
ARG HTTPS_PROXY=http://http-p.srv.cc.suzuka-ct.ac.jp:8080
ARG NO_PROXY=localhost,127.0.0.1

ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# OSレベルのパッケージインストール
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
COPY startup-trigger.groovy /usr/share/jenkins/ref/init.groovy.d/startup-trigger.groovy

RUN if ping -c 1 -W 1 http-p.srv.cc.suzuka-ct.ac.jp > /dev/null 2>&1; then \
  export http_proxy="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"; \
  export https_proxy="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"; \
  export HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"; \
  export HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"; \
  export no_proxy="localhost,127.0.0.1"; \
  export NO_PROXY="localhost,127.0.0.1"; \
  JAVA_OPTS="-Dhttps.proxyHost=http-p.srv.cc.suzuka-ct.ac.jp -Dhttps.proxyPort=8080"; \
  echo "Proxy enabled"; \
  else \
  unset http_proxy; \
  unset https_proxy; \
  unset HTTP_PROXY; \
  unset HTTPS_PROXY; \
  unset no_proxy; \
  unset NO_PROXY; \
  JAVA_OPTS=""; \
  echo "Proxy disabled"; \
  fi && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  apt-transport-https=2.6.1 \
  ca-certificates=20230311+deb12u1 \
  curl=7.88.1-10+deb12u12 \
  gnupg=2.2.40-1.1 \
  software-properties-common=0.99.30-4.1~deb12u1 \
  git=1:2.39.5-0+deb12u2 && \
  mkdir -p /etc/apt/keyrings && \
  curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
  apt-get update && \
  apt-get install -y --no-install-recommends docker-ce-cli=5:28.3.2-1~debian.12~bookworm && \
  rm -rf /var/lib/apt/lists/* && \
  $JAVA_OPTS jenkins-plugin-cli -f /usr/share/jenkins/ref/plugins.txt && \
  mkdir -p /usr/share/jenkins/ref/init.groovy.d

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 CMD curl -f http://localhost:8080/login || exit 1

USER jenkins
