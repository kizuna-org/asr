FROM ubuntu/squid:latest

RUN echo 'Acquire::http::Proxy "http://http-p.srv.cc.suzuka-ct.ac.jp:8080/";' > /etc/apt/apt.conf.d/01proxy \
    && echo 'Acquire::https::Proxy "http://http-p.srv.cc.suzuka-ct.ac.jp:8080/";' >> /etc/apt/apt.conf.d/01proxy \
    && apt-get update && apt-get install -y busybox
