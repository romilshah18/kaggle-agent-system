# Setup Guide

## Prerequisites

1. **Docker & Docker Compose**
   ```bash
   docker --version
   docker-compose --version
   ```

2. **Python 3.11+**
   ```bash
   python3 --version
   ```

3. **API Keys Required**:
   - Anthropic API Key (for Claude)
   - Kaggle API credentials (username + key)

## Step-by-Step Setup

### 1. Configure Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
POSTGRES_HOST=aws-1-ap-south-1.pooler.supabase.com
POSTGRES_PORT=5432
POSTGRES_DB=postgres
POSTGRES_USER=postgres.mfpzlcayybipldovxmcg
POSTGRES_PASSWORD=JyHlUBSV4iUjrYnY

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
CELERY_WORKER_CONCURRENCY=10
CELERY_TASK_TIME_LIMIT=7200
CELERY_TASK_SOFT_TIME_LIMIT=6900

# Docker
DOCKER_HOST=unix:///var/run/docker.sock
CONTAINER_CPU_LIMIT=4
CONTAINER_MEMORY_LIMIT=8g
CONTAINER_TIMEOUT=7200

# LLM API Keys - REQUIRED
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# Kaggle Credentials - REQUIRED
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_api_key

# Storage
STORAGE_PATH=/app/storage
SUBMISSIONS_PATH=/app/storage/submissions
LOGS_PATH=/app/storage/logs

# Rate Limiting
RATE_LIMIT_PER_MINUTE=10
MAX_CONCURRENT_JOBS=50
```

### 2. Get API Keys

**Anthropic API Key**:
1. Go to https://console.anthropic.com/
2. Sign up / Log in
3. Go to API Keys section
4. Create a new key
5. Copy to `.env` as `ANTHROPIC_API_KEY`

**Kaggle API Credentials**:
1. Go to https://www.kaggle.com/
2. Log in to your account
3. Go to Account → API → Create New API Token
4. Download `kaggle.json`
5. Extract `username` and `key` from the file
6. Add to `.env` as `KAGGLE_USERNAME` and `KAGGLE_KEY`

### 3. Build and Start the System

```bash
# Make scripts executable
chmod +x infrastructure/scripts/*.sh

# Option 1: Full deployment (recommended)
./infrastructure/scripts/deploy.sh

# Option 2: Manual steps
# Build images
./infrastructure/scripts/build_images.sh

# Start services
cd infrastructure
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### 4. Verify System is Running

```bash
# Check all containers are up
docker-compose -f infrastructure/docker-compose.yml ps

# Should show:
# - kaggle-postgres (healthy)
# - kaggle-redis (healthy)
# - kaggle-api (running)
# - kaggle-worker (running)
# - kaggle-flower (running)

# Check API health
curl http://localhost:8000/health

# Expected output:
# {
#   "status": "healthy",
#   "timestamp": "...",
#   "services": {
#     "api": "healthy",
#     "redis": "healthy",
#     "database": "healthy"
#   },
#   "queue_length": 0
# }
```

### 5. Test with a Sample Job

```bash
# Submit a job
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/competitions/titanic"}'

# Response:
# {
#   "job_id": "abc123...",
#   "status": "queued",
#   "created_at": "...",
#   "message": "Job created successfully..."
# }

# Check status (replace with your job_id)
curl "http://localhost:8000/status/abc123..."

# Monitor logs
docker-compose -f infrastructure/docker-compose.yml logs -f worker

# Wait ~30-60 minutes for completion
# Then download submission
curl "http://localhost:8000/result/abc123.../submission.csv" -o submission.csv
```

### 6. Access Monitoring Dashboards

- **API Documentation**: http://localhost:8000/docs
- **Flower (Celery Monitor)**: http://localhost:5555
- **API Health**: http://localhost:8000/health

## Running Tests

### Integration Test
```bash
# Install test dependencies
pip install aiohttp

# Run end-to-end test
python tests/integration/test_end_to_end.py
```

### Load Test
```bash
# Run concurrency test (10, 25, 50 concurrent)
python tests/load/test_concurrency.py
```

## Troubleshooting

### Issue: Docker daemon not accessible
```bash
# Check Docker is running
docker ps

# Fix permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: Port already in use
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process or change API_PORT in .env
```

### Issue: Container fails to start
```bash
# Check logs
docker-compose -f infrastructure/docker-compose.yml logs api
docker-compose -f infrastructure/docker-compose.yml logs worker

# Restart services
docker-compose -f infrastructure/docker-compose.yml restart
```

### Issue: Job stuck in "queued"
```bash
# Check worker is running
docker-compose -f infrastructure/docker-compose.yml ps worker

# Check worker logs
docker-compose -f infrastructure/docker-compose.yml logs worker

# Restart worker
docker-compose -f infrastructure/docker-compose.yml restart worker
```

### Issue: Missing API keys error
```bash
# Verify .env file exists and has valid keys
cat .env | grep -E 'ANTHROPIC_API_KEY|KAGGLE_USERNAME|KAGGLE_KEY'

# Restart services after updating .env
docker-compose -f infrastructure/docker-compose.yml restart
```

## Stopping the System

```bash
# Stop all services
cd infrastructure
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

## Development Mode

For local development without Docker:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis locally or use Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=changeme123 postgres:16-alpine
docker run -d -p 6379:6379 redis:7-alpine

# Update .env to use localhost
# POSTGRES_HOST=localhost
# REDIS_HOST=localhost

# Run API
python -m uvicorn api.main:app --reload --port 8000

# Run worker (in another terminal)
celery -A worker.celery_app worker --loglevel=info
```

## Next Steps

- Read `README.md` for architecture overview
- Read `docs/ARCHITECTURE.md` for deep dive
- Read `docs/API.md` for API reference
- Experiment with different Kaggle competitions
- Monitor performance with Flower dashboard

