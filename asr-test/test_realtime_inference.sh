#!/bin/bash
# Test script for realtime inference on GPU server
# This script should be run from the local machine

set -e

SSH_HOST="edu-gpu"
REMOTE_PATH="/home/students/r03i/r03i18/asr-test/asr/asr-test"

echo "🧪 Testing realtime inference on GPU server..."
echo ""

# Test 1: Run the realtime model test
echo "📋 Test 1: Running realtime model unit tests..."
ssh ${SSH_HOST} "cd ${REMOTE_PATH} && sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec -T asr-api python test_realtime_model.py"
echo ""

# Test 2: Run the realtime demo
echo "📋 Test 2: Running realtime demo..."
ssh ${SSH_HOST} "cd ${REMOTE_PATH} && sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec -T asr-api python demo_realtime.py"
echo ""

# Test 3: Check logs
echo "📋 Test 3: Checking recent logs..."
ssh ${SSH_HOST} "cd ${REMOTE_PATH} && sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=50 asr-api"
echo ""

echo "✅ All tests completed!"
