FROM ubuntu/squid:latest

ARG PROXY_URL=http://http-p.srv.cc.suzuka-ct.ac.jp:8080/

RUN echo "Acquire::http::Proxy \"${PROXY_URL}\";" > /etc/apt/apt.conf.d/01proxy \
    && echo "Acquire::https::Proxy \"${PROXY_URL}\";" >> /etc/apt/apt.conf.d/01proxy \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    busybox=1:1.36.1-6ubuntu3.1 \
    tcpdump=4.99.4-3ubuntu4 \
    && rm -rf /var/lib/apt/lists/*
