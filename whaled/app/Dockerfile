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
    requests

# Copy subscriber script
COPY subscriber.py .

# Make script executable
RUN chmod +x subscriber.py

# Run subscriber
CMD ["python", "subscriber.py"]