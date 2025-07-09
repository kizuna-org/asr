# jenkins/Dockerfile (最終確定版)

FROM jenkins/jenkins:lts

USER root

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# OSレベルのパッケージインストール
RUN apt-get update && \
  apt-get install -y --no-install-recommends apt-transport-https ca-certificates curl gnupg2 software-properties-common git && \
  mkdir -p /etc/apt/keyrings && \
  curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
  echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
  apt-get update && \
  apt-get install -y --no-install-recommends docker-ce-cli && \
  rm -rf /var/lib/apt/lists/*

# Jenkinsプラグインのインストール
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN JAVA_OPTS="-Dhttps.proxyHost=http-p.srv.cc.suzuka-ct.ac.jp -Dhttps.proxyPort=8080" jenkins-plugin-cli -f /usr/share/jenkins/ref/plugins.txt

# Create init.groovy.d directory and add startup script
RUN mkdir -p /usr/share/jenkins/ref/init.groovy.d
COPY startup-trigger.groovy /usr/share/jenkins/ref/init.groovy.d/startup-trigger.groovy

USER jenkins
