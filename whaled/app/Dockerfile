FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Configure DNS and networking for better reliability
RUN mkdir -p /etc/docker/daemon.json && echo '{"dns": ["8.8.8.8", "8.8.4.4", "1.1.1.1"]}' > /etc/docker/daemon.json \
    && echo "nameserver 8.8.8.8" > /etc/resolv.conf \
    && echo "nameserver 8.8.4.4" >> /etc/resolv.conf \
    && echo "nameserver 1.1.1.1" >> /etc/resolv.conf \
    && echo "options timeout:2 attempts:3 rotate single-request-reopen" >> /etc/resolv.conf

# Set working directory
WORKDIR /app

# Install Python dependencies with improved network resilience
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dnsutils \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Checking network connectivity..." \
    && ping -c 1 8.8.8.8 || echo "Warning: Network connectivity to DNS server issues detected" \
    && curl -s --connect-timeout 5 https://pypi.org || echo "Warning: Network connectivity to PyPI issues detected" \
    && echo "Using alternative pip install method with multiple mirrors..." \
    && pip config set global.index-url https://pypi.org/simple/ \
    && pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/ https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir --timeout 120 --retries 15 \
    google-cloud-pubsub==2.13.0 \
    boto3==1.26.0 \
    requests==2.28.1 \
    python-dotenv==0.21.0

# Create config directory
RUN mkdir -p /app/config

# Copy subscriber script and shared logger
COPY subscriber.py .
COPY ../shared /app/shared

# Copy config directory (will contain .env and service account key)
COPY config/ /app/config/

# Make script executable
RUN chmod +x subscriber.py

# Run subscriber
CMD ["python", "subscriber.py"]
