# kics-scan disable=f45ea400-6bbe-4501-9fc7-1c3d75c32067
# hadolint ignore=DL3007
FROM ubuntu/squid:latest@sha256:98f98aaa024e4a58433450f8adde48ef5748f1ae05133b2367d64317cc881a3a

ARG PROXY_URL=http://http-p.srv.cc.suzuka-ct.ac.jp:8080/

RUN echo "Acquire::http::Proxy \"${PROXY_URL}\";" > /etc/apt/apt.conf.d/01proxy \
    && echo "Acquire::https::Proxy \"${PROXY_URL}\";" >> /etc/apt/apt.conf.d/01proxy \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    busybox=1:1.36.1-6ubuntu3.1 \
    tcpdump=4.99.4-3ubuntu4 \
    && rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 CMD pgrep squid || exit 1
USER nobody:nogroup
