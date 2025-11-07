# Testing Guide

This guide covers different testing approaches for the Kaggle Agent System.

## 🚀 Quick Start: Standalone Agent Testing

**Use this for fast iteration during development!** No Docker required.

### Setup

```bash
# 1. Install dependencies locally
pip install -r requirements.txt

# 2. Make sure your .env has these variables
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_key
# ANTHROPIC_API_KEY=your_key
```

### Run Standalone Test

```bash
# Option 1: Use the quick test script
./tests/unit/test_agent_quick.sh

# Option 2: Run directly with Python
source .env  # Load environment variables
python tests/unit/test_agent_standalone.py
```

This will:
- ✅ Test all 4 agent stages (Analyze, Plan, Generate, Execute)
- ✅ Use Titanic competition as test case
- ✅ Run locally without Docker (much faster!)
- ✅ Output results to `storage/submissions/test-standalone/`

### What to Check

After running the test:

```bash
# View generated training code
cat storage/submissions/test-standalone/generated_model.py

# View submission file
head storage/submissions/test-standalone/submission.csv
```

## 🐳 Full Integration Testing

Once the agent works standalone, test the full Docker stack:

### 1. Deploy Full System

```bash
./infrastructure/scripts/deploy.sh
```

### 2. Submit Test Job

```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/competitions/titanic"}'
```

### 3. Monitor Progress

```bash
# Get job status
curl http://localhost:8000/status/YOUR_JOB_ID

# Watch logs
docker-compose -f infrastructure/docker-compose.yml logs -f worker

# View Flower UI
open http://localhost:5555
```

### 4. Get Results

```bash
# Download submission
curl http://localhost:8000/result/YOUR_JOB_ID/submission.csv -o submission.csv

# View logs
curl http://localhost:8000/logs/YOUR_JOB_ID
```

## 🧪 Running Tests

### Unit Tests

```bash
# Test agent components individually
python -m pytest tests/unit/ -v

# Test specific component
python -m pytest tests/unit/test_agent_standalone.py -v
```

### Integration Tests

```bash
# Full end-to-end test (requires Docker running)
python tests/integration/test_end_to_end.py
```

### Load Tests

```bash
# Test concurrent requests
python tests/load/test_concurrency.py --requests 50 --workers 10
```

## 🔍 Debugging

### Agent Failures

```bash
# 1. Run standalone test to see detailed logs
./tests/unit/test_agent_quick.sh

# 2. Check generated code
cat storage/submissions/test-standalone/generated_model.py

# 3. Run generated code manually
cd storage/submissions/test-standalone
python generated_model.py
```

### Docker Issues

```bash
# Check container logs
docker-compose -f infrastructure/docker-compose.yml logs -f worker
docker-compose -f infrastructure/docker-compose.yml logs -f api

# Inspect specific job container
docker ps -a | grep agent
docker logs <container_id>

# Rebuild without cache
cd infrastructure
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Test Environments

| Environment | Speed | Isolation | Use Case |
|-------------|-------|-----------|----------|
| **Standalone** | ⚡ Fast | None | Agent logic development |
| **Local Docker** | 🐢 Slow | Full | Integration testing |
| **Production** | 🐢 Slow | Full | Final validation |

## 💡 Development Workflow

**Recommended iteration cycle:**

1. **Develop** → Edit agent code (analyzer, planner, generator, executor)
2. **Test Standalone** → Run `./tests/unit/test_agent_quick.sh`
3. **Fix Issues** → Debug with actual generated code
4. **Repeat** until standalone test passes
5. **Integration Test** → Build Docker and test full stack
6. **Deploy** → Use `deploy.sh` for production

This workflow is **10-20x faster** than rebuilding Docker each time! 🚀

## 📝 Common Issues

### "No module named 'agent'"

- **Standalone**: Make sure you're in project root
- **Docker**: Check `PYTHONPATH` is set in Dockerfile

### "Kaggle API credentials not found"

```bash
# Check .env has these set
grep KAGGLE .env

# Make sure to source it
source .env
```

### "ANTHROPIC_API_KEY not set"

```bash
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

### Generated code fails

1. Check `storage/submissions/test-standalone/generated_model.py`
2. Look for obvious errors (missing imports, wrong paths)
3. Test manually: `python storage/submissions/test-standalone/generated_model.py`
4. Improve the prompt in `agent/generator/code_generator.py`

## 🎯 Next Steps

Once standalone tests pass:

1. ✅ Build agent Docker image: `docker build -f infrastructure/docker/Dockerfile.agent -t kaggle-agent .`
2. ✅ Deploy full stack: `./infrastructure/scripts/deploy.sh`
3. ✅ Test with real competitions
4. ✅ Run load tests to verify 50+ concurrent handling

