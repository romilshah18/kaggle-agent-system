# Quick Reference Card

## 🚀 Common Commands

### Start System
```bash
./infrastructure/scripts/deploy.sh
```

### Stop System
```bash
cd infrastructure
docker-compose down
```

### View Logs
```bash
# All services
docker-compose -f infrastructure/docker-compose.yml logs -f

# Specific service
docker-compose -f infrastructure/docker-compose.yml logs -f api
docker-compose -f infrastructure/docker-compose.yml logs -f worker
```

### Check Status
```bash
# Health check
curl http://localhost:8000/health

# List all containers
docker-compose -f infrastructure/docker-compose.yml ps
```

---

## 📡 API Endpoints

### Submit Job
```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/competitions/titanic"}'
```

### Check Status
```bash
curl "http://localhost:8000/status/{job_id}"
```

### Download Result
```bash
curl "http://localhost:8000/result/{job_id}/submission.csv" -o submission.csv
```

### View Logs
```bash
curl "http://localhost:8000/logs/{job_id}"
```

### List All Jobs
```bash
curl "http://localhost:8000/jobs?limit=10"
```

---

## 🧪 Testing

### Integration Test
```bash
python tests/integration/test_end_to_end.py
```

### Load Test
```bash
python tests/load/test_concurrency.py
```

---

## 🔍 Monitoring

- **API Docs**: http://localhost:8000/docs
- **Flower (Celery)**: http://localhost:5555
- **Health Check**: http://localhost:8000/health

---

## 🐛 Troubleshooting

### Restart Services
```bash
cd infrastructure
docker-compose restart api
docker-compose restart worker
```

### View Worker Status
```bash
docker-compose -f infrastructure/docker-compose.yml exec worker celery -A worker.celery_app status
```

### Clean Restart
```bash
cd infrastructure
docker-compose down -v
docker-compose up -d
```

### Check Database
```bash
docker-compose -f infrastructure/docker-compose.yml exec postgres psql -U kaggle_user -d kaggle_agent
```

---

## 📂 Important Files

- **Configuration**: `.env`
- **API Code**: `api/main.py`
- **Worker Code**: `worker/tasks/competition_task.py`
- **Agent Code**: `agent/main.py`
- **Docker Compose**: `infrastructure/docker-compose.yml`
- **Documentation**: `README.md`, `docs/ARCHITECTURE.md`

---

## 🔑 Environment Variables

Required in `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_api_key
```

---

## 🎯 Key Features

- ✅ Handles 50+ concurrent requests
- ✅ Automatic retry (max 2 attempts)
- ✅ 2-hour timeout per job
- ✅ Docker isolation (4 CPU, 8GB RAM)
- ✅ LLM-powered strategy (Claude Sonnet 4)
- ✅ Fallback templates

---

## 📊 Architecture

```
Client → FastAPI → Redis Queue → Celery Workers → Docker Containers → submission.csv
          ↓
     PostgreSQL (job state)
```

---

## 🔧 Development Mode

```bash
# Start only infrastructure
cd infrastructure
docker-compose up -d postgres redis

# Run API locally
python -m uvicorn api.main:app --reload --port 8000

# Run worker locally
celery -A worker.celery_app worker --loglevel=info
```

---

## 📈 Expected Performance

- API Response: < 200ms
- Job Duration: 30-60 minutes
- Success Rate: > 85%
- Concurrent Capacity: 50+ jobs

---

## 🆘 Support Resources

1. **Setup Issues**: See `SETUP.md`
2. **Architecture Questions**: See `docs/ARCHITECTURE.md`
3. **API Reference**: See `docs/API.md`
4. **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`

