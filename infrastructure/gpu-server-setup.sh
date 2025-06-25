#!/bin/bash
# Setup script for GPU server - configures Docker containers with cron monitoring

set -e

echo "🚀 Setting up GPU server for CI/CD pipeline..."

# Check if Docker is available
if ! command -v docker &>/dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   You can install Docker without sudo using Docker Desktop or ask your administrator."
    exit 1
fi

# Check if user can run Docker commands
if ! sudo docker ps &>/dev/null; then
    echo "❌ Cannot run Docker commands. Please ensure your user is in the docker group."
    echo "   Ask your administrator to run: sudo usermod -aG docker $USER"
    echo "   Then log out and log back in."
    exit 1
fi

# Create application directory in user's home
APP_DIR="$HOME/whaled"
mkdir -p "$APP_DIR/build"
mkdir -p "$APP_DIR/app"
mkdir -p "$APP_DIR/logs"
mkdir -p "$APP_DIR/shared"
mkdir -p "$APP_DIR/config"

echo "📁 Created application directory: $APP_DIR"

# Copy whaled files if they exist
if [ -d "whaled" ]; then
    echo "📋 Copying whaled project files..."

    # Copy requirements.txt
    if [ -f "whaled/requirements.txt" ]; then
        cp whaled/requirements.txt "$APP_DIR/"
        echo "✅ Copied requirements.txt"
    else
        echo "⚠️  Requirements.txt not found at whaled/requirements.txt"
    fi

    # Copy app subscriber
    if [ -f "whaled/app/subscriber.py" ]; then
        cp whaled/app/subscriber.py "$APP_DIR/app/"
        echo "✅ Copied app subscriber script"
    else
        echo "⚠️  App subscriber script not found at whaled/app/subscriber.py"
    fi

    # Copy build subscriber
    if [ -f "whaled/build/subscriber.py" ]; then
        cp whaled/build/subscriber.py "$APP_DIR/build/"
        echo "✅ Copied build subscriber script"
    else
        echo "⚠️  Build subscriber script not found at whaled/build/subscriber.py"
    fi

    # Copy Dockerfiles
    if [ -f "whaled/app/Dockerfile" ]; then
        cp whaled/app/Dockerfile "$APP_DIR/app/"
        echo "✅ Copied app Dockerfile"
    else
        echo "⚠️  App Dockerfile not found at whaled/app/Dockerfile"
    fi

    if [ -f "whaled/build/Dockerfile" ]; then
        cp whaled/build/Dockerfile "$APP_DIR/build/"
        echo "✅ Copied build Dockerfile"
    else
        echo "⚠️  Build Dockerfile not found at whaled/build/Dockerfile"
    fi

    # Copy shared logger if it exists
    if [ -d "shared" ]; then
        cp -r shared/* "$APP_DIR/shared/"
        echo "✅ Copied shared logger"
    else
        echo "⚠️  Shared directory not found"
    fi
else
    echo "❌ whaled directory not found. Please run this script from the project root."
fi

# docker_build.shをホームディレクトリに配置
if [ -f "$(dirname "$0")/docker_build.sh" ]; then
    cp "$(dirname "$0")/docker_build.sh" ~/docker_build.sh
    chmod +x ~/docker_build.sh
    echo "✅ docker_build.shをホームディレクトリに配置しました"
else
    echo "❌ docker_build.shが見つかりません: $(dirname "$0")/docker_build.sh"
    exit 1
fi

# Build Docker images
echo "🐳 Building Docker images..."

# Build app subscriber image
if [ -f "$APP_DIR/app/Dockerfile" ]; then
    echo "🔨 Building app subscriber image..."
    bash ~/docker_build.sh -f "$APP_DIR/app/Dockerfile" "$APP_DIR" \
        --build-arg HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080" \
        --build-arg HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080" \
        -t whaled-app-subscriber
    echo "✅ App subscriber image built successfully"
else
    echo "⚠️  App Dockerfile not found, skipping app image build"
fi

# Build build subscriber image
if [ -f "$APP_DIR/build/Dockerfile" ]; then
    echo "🔨 Building build subscriber image..."
    bash ~/docker_build.sh -f "$APP_DIR/build/Dockerfile" "$APP_DIR" \
        --build-arg HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080" \
        --build-arg HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080" \
        -t whaled-build-subscriber
    echo "✅ Build subscriber image built successfully"
else
    echo "⚠️  Build Dockerfile not found, skipping build image build"
fi

# Create Docker run scripts
echo "📝 Creating Docker run scripts..."
cat >"$APP_DIR/run-app-subscriber.sh" <<'EOF'
#!/bin/bash
# Script to run the app subscriber container

APP_DIR="$HOME/whaled"
CONTAINER_NAME="whaled-app-subscriber"

# Load environment variables
if [ -f "$APP_DIR/config/.env" ]; then
    export $(grep -v '^#' "$APP_DIR/config/.env" | xargs)
fi

# Check if container is already running
if sudo docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "🔄 Container $CONTAINER_NAME is already running"
    exit 0
fi

# Remove existing container if it exists but is stopped
sudo docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Run the container using the built image
sudo docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -v "$APP_DIR/logs:/logs" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -v "$APP_DIR/config:/app/config" \
    -v "$APP_DIR/shared:/app/shared" \
    --env-file "$APP_DIR/config/.env" \
    whaled-app-subscriber

echo "🚀 Started container: $CONTAINER_NAME"
EOF

cat >"$APP_DIR/run-build-subscriber.sh" <<'EOF'
#!/bin/bash
# Script to run the build subscriber container

APP_DIR="$HOME/whaled"
CONTAINER_NAME="whaled-build-subscriber"

# Load environment variables
if [ -f "$APP_DIR/config/.env" ]; then
    export $(grep -v '^#' "$APP_DIR/config/.env" | xargs)
fi

# Check if container is already running
if sudo docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "🔄 Container $CONTAINER_NAME is already running"
    exit 0
fi

# Remove existing container if it exists but is stopped
sudo docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Run the container using the built image
sudo docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -v "$APP_DIR/logs:/logs" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -v "/tmp:/tmp" \
    --env-file "$APP_DIR/config/.env" \
    whaled-build-subscriber

echo "🚀 Started container: $CONTAINER_NAME"
EOF

# Create container monitoring script
echo "👀 Creating container monitoring script..."
cat >"$APP_DIR/monitor-containers.sh" <<'EOF'
#!/bin/bash
# Script to monitor and restart containers if they're not running

APP_DIR="$HOME/whaled"
APP_CONTAINER="whaled-app-subscriber"
BUILD_CONTAINER="whaled-build-subscriber"

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$APP_DIR/logs/monitor.log"
}

# Check app subscriber container
if ! sudo docker ps --format "table {{.Names}}" | grep -q "^${APP_CONTAINER}$"; then
    log "⚠️  Container $APP_CONTAINER is not running. Starting..."

    if [ -f "$APP_DIR/run-app-subscriber.sh" ]; then
        bash "$APP_DIR/run-app-subscriber.sh" >> "$APP_DIR/logs/monitor.log" 2>&1
        log "🚀 Attempted to start $APP_CONTAINER"
    else
        log "❌ ERROR: Start script not found at $APP_DIR/run-app-subscriber.sh"
    fi
else
    log "✅ Container $APP_CONTAINER is running normally"
fi

# Check build subscriber container
if ! sudo docker ps --format "table {{.Names}}" | grep -q "^${BUILD_CONTAINER}$"; then
    log "⚠️  Container $BUILD_CONTAINER is not running. Starting..."

    if [ -f "$APP_DIR/run-build-subscriber.sh" ]; then
        bash "$APP_DIR/run-build-subscriber.sh" >> "$APP_DIR/logs/monitor.log" 2>&1
        log "🚀 Attempted to start $BUILD_CONTAINER"
    else
        log "❌ ERROR: Start script not found at $APP_DIR/run-build-subscriber.sh"
    fi
else
    log "✅ Container $BUILD_CONTAINER is running normally"
fi
EOF

# Make scripts executable
chmod +x "$APP_DIR/run-app-subscriber.sh"
chmod +x "$APP_DIR/run-build-subscriber.sh"
chmod +x "$APP_DIR/monitor-containers.sh"

# Create cron job to monitor containers every 5 minutes
echo "⏰ Setting up cron job for container monitoring..."
CRON_JOB="*/5 * * * * $APP_DIR/monitor-containers.sh"

# Check if cron job already exists
if ! crontab -l 2>/dev/null | grep -q "$APP_DIR/monitor-containers.sh"; then
    # Add the cron job
    (
        crontab -l 2>/dev/null
        echo "$CRON_JOB"
    ) | crontab -
    echo "✅ Added cron job to monitor containers every 5 minutes"
else
    echo "🔄 Cron job already exists"
fi

echo "🎉 GPU server setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. ⚙️  Configure environment variables in:"
echo "   $APP_DIR/config/.env"
echo ""
echo "2. 🔑 Set up GCP service account credentials:"
echo "   - Place your service-account-key.json in $APP_DIR/config/"
echo "   - The containers will automatically use it from /app/config/"
echo ""
echo "3. 🔐 Login to GitHub Container Registry:"
echo "   echo \$GITHUB_TOKEN | sudo docker login ghcr.io -u USERNAME --password-stdin"
echo ""
echo "4. 🚀 Start the containers manually (first time):"
echo "   bash $APP_DIR/run-app-subscriber.sh"
echo "   bash $APP_DIR/run-build-subscriber.sh"
echo ""
echo "5. 📊 Check container status:"
echo "   sudo docker ps | grep whaled"
echo "   sudo docker logs whaled-app-subscriber"
echo "   sudo docker logs whaled-build-subscriber"
echo "   tail -f $APP_DIR/logs/monitor.log"
echo ""
echo "6. ⏰ The cron job will automatically restart containers if they stop"
echo "   To view cron jobs: crontab -l"
echo "   To remove cron job: crontab -e (then delete the line)"
