FROM ubuntu/squid:6.6-24.04

ARG PROXY_URL=http://http-p.srv.cc.suzuka-ct.ac.jp:8080/

RUN echo "Acquire::http::Proxy \"${PROXY_URL}\";" > /etc/apt/apt.conf.d/01proxy \
    && echo "Acquire::https::Proxy \"${PROXY_URL}\";" >> /etc/apt/apt.conf.d/01proxy \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    busybox=1.30.1-4ubuntu6.5 \
    tcpdump=4.9.3-4ubuntu0.3 \
    && rm -rf /var/lib/apt/lists/*
