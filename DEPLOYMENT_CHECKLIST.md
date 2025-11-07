# Deployment Checklist

Use this checklist to ensure successful deployment of the Kaggle Agent System.

---

## ✅ Pre-Deployment Checklist

### System Requirements
- [ ] Docker installed and running
  ```bash
  docker --version  # Should be 20.10+
  docker ps         # Should work without sudo
  ```
- [ ] Docker Compose installed
  ```bash
  docker-compose --version  # Should be 1.29+
  ```
- [ ] Python 3.11+ installed (for local testing)
  ```bash
  python3 --version
  ```

### API Keys
- [ ] Anthropic API key obtained
  - Visit: https://console.anthropic.com/
  - Create key and save securely
  
- [ ] Kaggle credentials obtained
  - Visit: https://www.kaggle.com/account
  - Download API token (kaggle.json)
  - Extract username and key

### Configuration
- [ ] Create `.env` file from template
  ```bash
  # Create if not exists
  touch .env
  ```
  
- [ ] Add Anthropic API key to `.env`
  ```bash
  ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxx
  ```
  
- [ ] Add Kaggle credentials to `.env`
  ```bash
  KAGGLE_USERNAME=your_username
  KAGGLE_KEY=your_key_here
  ```
  
- [ ] Verify all required variables are set
  ```bash
  grep -E 'ANTHROPIC_API_KEY|KAGGLE_USERNAME|KAGGLE_KEY' .env
  ```

---

## 🚀 Deployment Steps

### Step 1: Initial Setup
- [ ] Navigate to project directory
  ```bash
  cd /Users/romil/Documents/kaggle-agent-system
  ```

- [ ] Make scripts executable
  ```bash
  chmod +x infrastructure/scripts/*.sh
  ```

### Step 2: Build Images
- [ ] Build Docker images
  ```bash
  ./infrastructure/scripts/build_images.sh
  ```
  
- [ ] Verify images created
  ```bash
  docker images | grep kaggle
  # Should show: kaggle-api, kaggle-worker, kaggle-agent
  ```

### Step 3: Start Services
- [ ] Start all services
  ```bash
  cd infrastructure
  docker-compose up -d
  ```
  
- [ ] Wait for services to be ready (15-20 seconds)
  ```bash
  sleep 20
  ```

### Step 4: Verify Deployment
- [ ] Check all containers are running
  ```bash
  docker-compose ps
  # Should show 5 services: postgres, redis, api, worker, flower
  ```
  
- [ ] Verify PostgreSQL is healthy
  ```bash
  docker-compose logs postgres | grep "database system is ready"
  ```
  
- [ ] Verify Redis is healthy
  ```bash
  docker-compose logs redis | grep "Ready to accept connections"
  ```
  
- [ ] Test API health endpoint
  ```bash
  curl http://localhost:8000/health
  # Should return: {"status": "healthy", ...}
  ```

---

## 🧪 Post-Deployment Testing

### Basic Functionality
- [ ] Access API documentation
  ```bash
  open http://localhost:8000/docs
  # Should show Swagger UI
  ```
  
- [ ] Access Flower dashboard
  ```bash
  open http://localhost:5555
  # Should show Celery monitoring
  ```

### Submit Test Job
- [ ] Create a test job
  ```bash
  curl -X POST "http://localhost:8000/run" \
    -H "Content-Type: application/json" \
    -d '{"kaggle_url": "https://www.kaggle.com/competitions/titanic"}'
  ```
  
- [ ] Save job_id from response
  
- [ ] Check job status
  ```bash
  curl "http://localhost:8000/status/{job_id}"
  # Should show: "status": "queued" or "running"
  ```
  
- [ ] Monitor worker logs
  ```bash
  docker-compose logs -f worker
  # Should see: "Starting job..." messages
  ```

### Integration Test
- [ ] Run integration test
  ```bash
  cd /Users/romil/Documents/kaggle-agent-system
  python tests/integration/test_end_to_end.py
  ```
  
- [ ] Verify test passes
  - Health check ✓
  - Job creation ✓
  - Status query ✓
  - (Optional) Wait for completion and download

### Load Test
- [ ] Install test dependencies
  ```bash
  pip install aiohttp
  ```
  
- [ ] Run load test
  ```bash
  python tests/load/test_concurrency.py
  ```
  
- [ ] Verify results
  - All 50 requests accepted ✓
  - Success rate 100% ✓
  - Avg response time < 1s ✓

---

## 📊 Monitoring Setup

### Dashboards
- [ ] Bookmark monitoring URLs
  - API: http://localhost:8000/health
  - Docs: http://localhost:8000/docs
  - Flower: http://localhost:5555

### Log Access
- [ ] Test log viewing
  ```bash
  # All logs
  docker-compose -f infrastructure/docker-compose.yml logs -f
  
  # Specific service
  docker-compose -f infrastructure/docker-compose.yml logs -f api
  ```

---

## 🔧 Troubleshooting Checklist

### If API doesn't start:
- [ ] Check API logs
  ```bash
  docker-compose logs api
  ```
- [ ] Verify port 8000 is free
  ```bash
  lsof -i :8000
  ```
- [ ] Check .env file exists and has correct values

### If Worker doesn't process jobs:
- [ ] Check worker logs
  ```bash
  docker-compose logs worker
  ```
- [ ] Verify Docker socket is accessible
  ```bash
  docker ps
  ```
- [ ] Check Redis is running
  ```bash
  docker-compose logs redis
  ```

### If Database connection fails:
- [ ] Check PostgreSQL logs
  ```bash
  docker-compose logs postgres
  ```
- [ ] Verify database credentials in .env
- [ ] Try restarting database
  ```bash
  docker-compose restart postgres
  ```

---

## 🛑 Rollback Procedure

If deployment fails:

1. [ ] Stop all services
   ```bash
   cd infrastructure
   docker-compose down
   ```

2. [ ] Remove volumes (if needed)
   ```bash
   docker-compose down -v
   ```

3. [ ] Check logs for errors
   ```bash
   docker-compose logs > deployment_error.log
   ```

4. [ ] Fix issues and retry

---

## ✅ Success Criteria

Deployment is successful when:

- [x] All 5 containers running (postgres, redis, api, worker, flower)
- [x] Health endpoint returns "healthy"
- [x] Can create a job via API
- [x] Worker picks up and processes jobs
- [x] Flower dashboard accessible
- [x] Integration test passes
- [x] Load test shows 100% success rate

---

## 📝 Production Readiness Checklist

For production deployment, additionally verify:

### Security
- [ ] Change default database password
- [ ] Add API authentication
- [ ] Enable HTTPS (reverse proxy)
- [ ] Set up firewall rules
- [ ] Rotate API keys regularly

### Monitoring
- [ ] Set up Prometheus/Grafana
- [ ] Configure alerting
- [ ] Set up log aggregation (ELK/Loki)
- [ ] Monitor disk space
- [ ] Monitor container health

### Backup
- [ ] Set up database backups
- [ ] Back up .env file (securely)
- [ ] Document recovery procedures

### Scaling
- [ ] Determine worker capacity needs
- [ ] Plan horizontal scaling strategy
- [ ] Set up load balancer (if needed)

---

## 📞 Support

If issues persist:

1. Check `SETUP.md` for detailed instructions
2. Review logs in `storage/logs/`
3. Check `TROUBLESHOOTING.md` (if available)
4. Review `docs/ARCHITECTURE.md` for system design

---

## 🎉 Deployment Complete!

Once all items are checked, your Kaggle Agent System is ready to use.

**Next Steps**:
1. Submit real Kaggle competition jobs
2. Monitor performance via Flower
3. Tune worker concurrency as needed
4. Explore extension scenarios in README.md

