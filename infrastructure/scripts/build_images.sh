#!/bin/bash
set -e

echo "Building Docker images..."

# Build API image
echo "Building API image..."
docker build -f infrastructure/docker/Dockerfile.api -t kaggle-api:latest .

# Build Worker image
echo "Building Worker image..."
docker build -f infrastructure/docker/Dockerfile.worker -t kaggle-worker:latest .

# Build Agent image
echo "Building Agent image..."
docker build -f infrastructure/docker/Dockerfile.agent -t kaggle-agent:latest .

echo "✓ All images built successfully"
docker images | grep kaggle

