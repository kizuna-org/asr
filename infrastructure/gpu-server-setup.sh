#!/bin/bash
# Setup script for GPU server - installs dependencies and configures subscribers

set -e

echo "🖥️  Setting up GPU server for CI/CD pipeline..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install NVIDIA Container Toolkit if not present
if ! command -v nvidia-container-runtime &> /dev/null; then
    echo "🎮 Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi

# Install Python and pip
echo "🐍 Installing Python dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv git

# Create application directory
APP_DIR="/opt/whaled"
sudo mkdir -p $APP_DIR/build
sudo mkdir -p $APP_DIR/app
sudo mkdir -p $APP_DIR/logs

# Create virtual environment
echo "🌐 Setting up Python virtual environment..."
sudo python3 -m venv $APP_DIR/venv
sudo $APP_DIR/venv/bin/pip install --upgrade pip

# Install Python dependencies
sudo $APP_DIR/venv/bin/pip install \
    google-cloud-pubsub \
    boto3 \
    requests

# Copy subscriber scripts (assuming they're in the current directory)
if [ -f "../whaled/build/subscriber.py" ]; then
    sudo cp ../whaled/build/subscriber.py $APP_DIR/build/
    sudo cp ../whaled/app/subscriber.py $APP_DIR/app/
    sudo chmod +x $APP_DIR/build/subscriber.py
    sudo chmod +x $APP_DIR/app/subscriber.py
fi

# Create systemd service files
echo "⚙️  Creating systemd services..."

# Build subscriber service
sudo tee /etc/systemd/system/whaled-build.service > /dev/null <<EOF
[Unit]
Description=Whaled Build Subscriber
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR/build
Environment=PYTHONPATH=$APP_DIR/venv/lib/python3.*/site-packages
ExecStart=$APP_DIR/venv/bin/python subscriber.py
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/logs/build-subscriber.log
StandardError=append:$APP_DIR/logs/build-subscriber.log

# Environment variables (configure these)
Environment=GCP_PROJECT_ID=your-project-id
Environment=BUILD_SUBSCRIPTION=build-triggers-sub
Environment=R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
Environment=R2_ACCESS_KEY_ID=your-r2-access-key
Environment=R2_SECRET_ACCESS_KEY=your-r2-secret-key
Environment=R2_BUCKET_NAME=your-bucket-name

[Install]
WantedBy=multi-user.target
EOF

# App subscriber service
sudo tee /etc/systemd/system/whaled-app.service > /dev/null <<EOF
[Unit]
Description=Whaled App Subscriber
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR/app
Environment=PYTHONPATH=$APP_DIR/venv/lib/python3.*/site-packages
ExecStart=$APP_DIR/venv/bin/python subscriber.py
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/logs/app-subscriber.log
StandardError=append:$APP_DIR/logs/app-subscriber.log

# Environment variables (configure these)
Environment=GCP_PROJECT_ID=your-project-id
Environment=APP_SUBSCRIPTION=app-triggers-sub
Environment=R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
Environment=R2_ACCESS_KEY_ID=your-r2-access-key
Environment=R2_SECRET_ACCESS_KEY=your-r2-secret-key
Environment=R2_BUCKET_NAME=your-bucket-name
Environment=HF_TOKEN=your-huggingface-token

[Install]
WantedBy=multi-user.target
EOF

# Create environment configuration template
sudo tee $APP_DIR/env.template > /dev/null <<EOF
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

# Set permissions
sudo chown -R root:root $APP_DIR
sudo chmod 755 $APP_DIR/build/subscriber.py
sudo chmod 755 $APP_DIR/app/subscriber.py

# Reload systemd
sudo systemctl daemon-reload

echo "✅ GPU server setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Configure environment variables in:"
echo "   - /etc/systemd/system/whaled-build.service"
echo "   - /etc/systemd/system/whaled-app.service"
echo "   Or use the template: $APP_DIR/env.template"
echo ""
echo "2. Set up GCP service account credentials:"
echo "   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json"
echo ""
echo "3. Login to GitHub Container Registry:"
echo "   echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
echo ""
echo "4. Start the services:"
echo "   sudo systemctl enable whaled-build whaled-app"
echo "   sudo systemctl start whaled-build whaled-app"
echo ""
echo "5. Check service status:"
echo "   sudo systemctl status whaled-build whaled-app"
echo "   sudo journalctl -u whaled-build -f"
echo "   sudo journalctl -u whaled-app -f"