FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y cifs-utils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /mnt/samba

COPY mount.sh /mount.sh
RUN chmod +x /mount.sh

COPY file_fetcher.py /file_fetcher.py
COPY main.py /main.py

ENTRYPOINT ["sh", "-c", "/mount.sh && python3 /main.py"]
