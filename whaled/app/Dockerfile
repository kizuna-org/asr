FROM python:3.9-slim

# Accept proxy arguments
ARG HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ARG HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

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
    && pip install --no-cache-dir --timeout 120 --retries 15 \
    google-cloud-pubsub==2.13.0 \
    boto3==1.26.0 \
    requests==2.28.1 \
    python-dotenv==0.21.0

# Create config directory
RUN mkdir -p /app/config

# Copy subscriber script and shared logger
COPY ./app/subscriber.py .
COPY ./shared/ /app/shared/

# Copy config directory (will contain .env and service account key)
COPY ./config/ /app/config/

# Make script executable
RUN chmod +x subscriber.py

# Run subscriber
CMD ["python", "subscriber.py"]
