# GPU Server Setup Script - Dockerfile Integration

**Date**: 2025-06-25
**Type**: Infrastructure Enhancement
**Component**: gpu-server-setup.sh

## Overview

Updated the GPU server setup script to properly use the whaled Dockerfiles instead of manually installing dependencies in containers.

## Changes Made

### 1. Enhanced File Copying
- Added copying of both `whaled/app/Dockerfile` and `whaled/build/Dockerfile`
- Added copying of shared logger directory
- Added creation of config directory for environment files

### 2. Docker Image Building
- Added building of custom Docker images using the copied Dockerfiles
- **App Subscriber Image**: Built from `whaled/app/Dockerfile` with context `$APP_DIR`
- **Build Subscriber Image**: Built from `whaled/build/Dockerfile` with context `$APP_DIR/build`
- Both images built with proxy arguments for network connectivity

### 3. Container Run Scripts
- **App Subscriber**: Now uses pre-built `whaled-app-subscriber` image
- **Build Subscriber**: Now uses pre-built `whaled-build-subscriber` image
- Improved volume mounting for config and shared directories
- Uses `--env-file` for cleaner environment variable management

### 4. Monitoring Enhancement
- Updated monitoring script to handle both app and build subscriber containers
- Added separate start scripts for each container type

## Key Improvements

1. **Consistency**: Now uses the same Dockerfiles as docker-compose.yml
2. **Network Resilience**: Leverages the network improvements in whaled/app/Dockerfile
3. **Proxy Support**: Properly handles proxy configuration during image builds
4. **Better Structure**: Separates concerns between build and app subscribers
5. **Maintainability**: Changes to Dockerfiles automatically reflected in GPU server setup

## File Structure Created

```
$HOME/whaled/
├── app/
│   ├── subscriber.py
│   └── Dockerfile
├── build/
│   ├── subscriber.py
│   └── Dockerfile
├── shared/
│   └── logger.py
├── config/
│   └── .env
├── logs/
├── .env
├── run-app-subscriber.sh
├── run-build-subscriber.sh
└── monitor-containers.sh
```

## Docker Images Built

- `whaled-app-subscriber`: For handling application execution requests
- `whaled-build-subscriber`: For handling container build requests

## Usage

The script now:
1. Copies all necessary files from the project
2. Builds Docker images using the proper Dockerfiles
3. Creates run scripts that use the built images
4. Sets up monitoring for both containers

This ensures the GPU server setup is consistent with the development environment and benefits from all the network resilience and proxy support improvements.
