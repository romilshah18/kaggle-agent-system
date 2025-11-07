#!/bin/bash
set -e

echo "Kaggle Agent System Deployment"
echo "==============================="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose required but not installed. Aborting." >&2; exit 1; }

# Check .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Copy .env.example to .env and fill in your credentials"
    exit 1
fi

# Validate required env vars
source .env
required_vars=("KAGGLE_USERNAME" "KAGGLE_KEY" "ANTHROPIC_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var not set in .env"
        exit 1
    fi
done

echo "✓ Prerequisites checked"

# Set STORAGE_HOST_PATH if not already set in .env
echo -e "\nConfiguring environment..."
if ! grep -q "^STORAGE_HOST_PATH=" .env 2>/dev/null; then
    PROJECT_ROOT="$(pwd)"
    echo "" >> .env
    echo "# Auto-generated: Absolute path for Docker volume mounting" >> .env
    echo "STORAGE_HOST_PATH=${PROJECT_ROOT}/storage" >> .env
    echo "✓ Added STORAGE_HOST_PATH=${PROJECT_ROOT}/storage to .env"
fi

# Build images
echo -e "\nBuilding Docker images..."
./infrastructure/scripts/build_images.sh

# Copy .env to infrastructure directory (docker-compose needs it there)
echo -e "\nCopying .env to infrastructure directory..."
cp .env infrastructure/.env

# Start services
echo -e "\nStarting services..."
cd infrastructure
docker-compose down
docker-compose up -d

# Wait for services
echo -e "\nWaiting for services to be ready..."
sleep 20

# Health check
echo -e "\nRunning health check..."
max_attempts=10
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "✓ System is healthy"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting for API... ($attempt/$max_attempts)"
    sleep 3
done

if [ $attempt -eq $max_attempts ]; then
    echo "✗ Health check failed"
    docker-compose logs api
    exit 1
fi

echo -e "\n==============================="
echo "✓ DEPLOYMENT SUCCESSFUL"
echo "==============================="
echo ""
echo "Services:"
echo "  API:    http://localhost:8000"
echo "  Docs:   http://localhost:8000/docs"
echo "  Flower: http://localhost:5555"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop system:"
echo "  docker-compose down"

