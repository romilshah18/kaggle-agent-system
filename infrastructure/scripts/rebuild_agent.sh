#!/bin/bash
# Quick rebuild of just the agent image
set -e

cd "$(dirname "$0")/../.."

echo "🔨 Rebuilding agent image..."
docker build --no-cache \
  -f infrastructure/docker/Dockerfile.agent \
  -t kaggle-agent:latest \
  .

echo "✅ Agent image rebuilt successfully!"
echo ""
echo "Next steps:"
echo "  1. Restart workers: cd infrastructure && docker-compose restart worker"
echo "  2. Test: curl -X POST http://localhost:8000/run -H 'Content-Type: application/json' -d '{\"kaggle_url\":\"https://www.kaggle.com/competitions/titanic\"}'"

