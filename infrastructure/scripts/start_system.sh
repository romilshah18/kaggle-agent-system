#!/bin/bash
set -e

echo "Starting Kaggle Agent System..."

# Build images if needed
if ! docker images | grep -q kaggle-api; then
    echo "Building images..."
    ./infrastructure/scripts/build_images.sh
fi

# Start services
cd infrastructure
docker-compose up -d

echo "Waiting for services to be healthy..."
sleep 15

# Check health
curl -f http://localhost:8000/health || (echo "Health check failed" && exit 1)

echo "✓ System started successfully"
echo ""
echo "Services:"
echo "  API:    http://localhost:8000"
echo "  Flower: http://localhost:5555"
echo ""
echo "View logs: docker-compose -f infrastructure/docker-compose.yml logs -f"

