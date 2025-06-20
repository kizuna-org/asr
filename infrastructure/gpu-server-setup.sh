#!/bin/bash
# Setup script for GPU server - configures Docker containers with cron monitoring

set -e

echo "🖥️  Setting up GPU server for CI/CD pipeline (no sudo required)..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   You can install Docker without sudo using Docker Desktop or ask your administrator."
    exit 1
fi

# Check if user can run Docker commands
if ! docker ps &> /dev/null; then
    echo "❌ Cannot run Docker commands. Please ensure your user is in the docker group."
    echo "   Ask your administrator to run: sudo usermod -aG docker $USER"
    echo "   Then log out and log back in."
    exit 1
fi

# Create application directory in user's home
APP_DIR="$HOME/whaled"
mkdir -p $APP_DIR/build
mkdir -p $APP_DIR/app
mkdir -p $APP_DIR/logs

echo "📁 Created application directory: $APP_DIR"

# Copy subscriber scripts if they exist
if [ -f "whaled/app/subscriber.py" ]; then
    cp whaled/app/subscriber.py $APP_DIR/app/
    echo "✅ Copied app subscriber script"
else
    echo "⚠️  App subscriber script not found at whaled/app/subscriber.py"
fi

# Create environment configuration template
cat > $APP_DIR/.env <<EOF
# GCP Configuration
GCP_PROJECT_ID=your-project-id
BUILD_SUBSCRIPTION=build-triggers-sub
APP_SUBSCRIPTION=app-triggers-sub

# Cloudflare R2 Configuration
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-r2-access-key
R2_SECRET_ACCESS_KEY=your-r2-secret-key
R2_BUCKET_NAME=your-bucket-name

# Hugging Face Configuration
HF_TOKEN=your-huggingface-token

# GitHub Container Registry Authentication
# Run: echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
EOF

# Create Docker run scripts
cat > $APP_DIR/run-app-subscriber.sh <<'EOF'
#!/bin/bash
# Script to run the app subscriber container

APP_DIR="$HOME/whaled"
CONTAINER_NAME="whaled-app-subscriber"

# Load environment variables
if [ -f "$APP_DIR/.env" ]; then
    export $(grep -v '^#' "$APP_DIR/.env" | xargs)
fi

# Check if container is already running
if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container $CONTAINER_NAME is already running"
    exit 0
fi

# Remove existing container if it exists but is stopped
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Run the container
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -v "$APP_DIR/app:/app" \
    -v "$APP_DIR/logs:/logs" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -e GCP_PROJECT_ID="$GCP_PROJECT_ID" \
    -e APP_SUBSCRIPTION="$APP_SUBSCRIPTION" \
    -e R2_ENDPOINT_URL="$R2_ENDPOINT_URL" \
    -e R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    -e R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    -e R2_BUCKET_NAME="$R2_BUCKET_NAME" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json" \
    python:3.11-slim \
    bash -c "
        cd /app && \
        pip install google-cloud-pubsub boto3 requests docker && \
        python subscriber.py 2>&1 | tee /logs/app-subscriber.log
    "

echo "Started container: $CONTAINER_NAME"
EOF

# Create container monitoring script
cat > $APP_DIR/monitor-containers.sh <<'EOF'
#!/bin/bash
# Script to monitor and restart containers if they're not running

APP_DIR="$HOME/whaled"
CONTAINER_NAME="whaled-app-subscriber"

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$APP_DIR/logs/monitor.log"
}

# Check if container is running
if ! docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    log "Container $CONTAINER_NAME is not running. Starting..."
    
    # Run the start script
    if [ -f "$APP_DIR/run-app-subscriber.sh" ]; then
        bash "$APP_DIR/run-app-subscriber.sh" >> "$APP_DIR/logs/monitor.log" 2>&1
        log "Attempted to start $CONTAINER_NAME"
    else
        log "ERROR: Start script not found at $APP_DIR/run-app-subscriber.sh"
    fi
else
    log "Container $CONTAINER_NAME is running normally"
fi
EOF

# Make scripts executable
chmod +x $APP_DIR/run-app-subscriber.sh
chmod +x $APP_DIR/monitor-containers.sh

# Create cron job to monitor containers every 5 minutes
CRON_JOB="*/5 * * * * $APP_DIR/monitor-containers.sh"

# Check if cron job already exists
if ! crontab -l 2>/dev/null | grep -q "$APP_DIR/monitor-containers.sh"; then
    # Add the cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Added cron job to monitor containers every 5 minutes"
else
    echo "✅ Cron job already exists"
fi

echo "✅ GPU server setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Configure environment variables in:"
echo "   $APP_DIR/.env"
echo ""
echo "2. Set up GCP service account credentials:"
echo "   - Place your service-account-key.json in $APP_DIR/app/"
echo "   - Or set GOOGLE_APPLICATION_CREDENTIALS environment variable"
echo ""
echo "3. Login to GitHub Container Registry:"
echo "   echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
echo ""
echo "4. Start the app subscriber manually (first time):"
echo "   bash $APP_DIR/run-app-subscriber.sh"
echo ""
echo "5. Check container status:"
echo "   docker ps | grep whaled"
echo "   docker logs whaled-app-subscriber"
echo "   tail -f $APP_DIR/logs/app-subscriber.log"
echo "   tail -f $APP_DIR/logs/monitor.log"
echo ""
echo "6. The cron job will automatically restart containers if they stop"
echo "   To view cron jobs: crontab -l"
echo "   To remove cron job: crontab -e (then delete the line)"