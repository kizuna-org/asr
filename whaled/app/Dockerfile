FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    google-cloud-pubsub \
    boto3 \
    requests \
    python-dotenv

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