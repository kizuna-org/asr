FROM docker:24.0.7-dind

RUN apk add --no-cache sudo shadow && \
  echo "$USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR /app

COPY . /app

RUN docker build .
RUN sudo docker ps

RUN source /app/infrastructure/gpu-server-setup.sh

CMD ["/app/test/host/run-all.sh"]
