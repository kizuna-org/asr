FROM python:3.9-slim

# Accept proxy arguments
ARG HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ARG HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    google-cloud-pubsub \
    boto3 \
    requests

# Copy subscriber script
COPY subscriber.py .

# Make script executable
RUN chmod +x subscriber.py

# Run subscriber
CMD ["python", "subscriber.py"]
