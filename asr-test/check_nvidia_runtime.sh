#!/bin/bash

echo "=== NVIDIA Container Runtime Check ==="

# Check if nvidia-container-runtime is installed
echo "1. Checking nvidia-container-runtime installation..."
if command -v nvidia-container-runtime &> /dev/null; then
    echo "✅ nvidia-container-runtime is installed"
    nvidia-container-runtime --version
else
    echo "❌ nvidia-container-runtime is not installed"
fi

# Check if nvidia-container-toolkit is installed
echo -e "\n2. Checking nvidia-container-toolkit installation..."
if command -v nvidia-container-toolkit &> /dev/null; then
    echo "✅ nvidia-container-toolkit is installed"
    nvidia-container-toolkit --version
else
    echo "❌ nvidia-container-toolkit is not installed"
fi

# Check Docker daemon configuration
echo -e "\n3. Checking Docker daemon configuration..."
if [ -f /etc/docker/daemon.json ]; then
    echo "✅ Docker daemon.json exists"
    cat /etc/docker/daemon.json
else
    echo "❌ Docker daemon.json not found"
fi

# Check if nvidia runtime is available in Docker
echo -e "\n4. Checking available Docker runtimes..."
if command -v docker &> /dev/null; then
    echo "Available runtimes:"
    docker info | grep -i runtime || echo "No runtime information available"
else
    echo "❌ Docker not available"
fi

# Check NVIDIA driver
echo -e "\n5. Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi is available"
    nvidia-smi
else
    echo "❌ nvidia-smi not found"
fi

# Check CUDA installation
echo -e "\n6. Checking CUDA installation..."
if [ -d "/usr/local/cuda" ]; then
    echo "✅ CUDA directory exists: /usr/local/cuda"
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        echo "✅ CUDA compiler (nvcc) is available"
        /usr/local/cuda/bin/nvcc --version
    else
        echo "❌ CUDA compiler (nvcc) not found"
    fi
else
    echo "❌ CUDA directory not found"
fi

echo -e "\n=== Check completed ==="
