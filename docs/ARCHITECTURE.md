# Architecture Deep Dive

## System Components

### 1. API Layer (FastAPI)

**Responsibilities**:
- HTTP request handling
- Job creation and tracking
- Status queries
- File serving (submission.csv)

**Design Principles**:
- **Stateless**: No session state, all state in PostgreSQL/Redis
- **Idempotent**: Same URL can create multiple jobs
- **Async**: Non-blocking I/O for high concurrency

**Scaling**:
- Horizontal: Multiple API instances behind load balancer
- Each instance: 4 Uvicorn workers (1 per CPU core)
- Total capacity: N instances × 4 workers × 100 connections = 400N concurrent requests

### 2. Message Queue (Redis)

**Purpose**:
- Job queue (Celery broker)
- Result storage (Celery backend)
- Rate limiting counters

**Why Redis over RabbitMQ**:
- Simpler setup (single binary)
- Serves dual purpose (broker + cache)
- Fast (in-memory)
- Built-in persistence (AOF)

**Configuration**:
```
Persistence: appendonly yes
Memory Policy: noeviction
Max Memory: 2GB
```

### 3. Database (PostgreSQL)

**Purpose**:
- Job metadata persistence
- Historical records
- Status tracking

**Schema Design**:
```sql
CREATE TABLE jobs (
    job_id UUID PRIMARY KEY,
    kaggle_url TEXT NOT NULL,
    competition_name VARCHAR(255),
    status VARCHAR(20),  -- Indexed
    created_at TIMESTAMP,  -- Indexed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    celery_task_id VARCHAR(36),
    submission_path TEXT,
    error_message TEXT,
    metadata JSONB  -- Flexible storage
);

-- Composite index for common queries
CREATE INDEX idx_status_created ON jobs(status, created_at DESC);
```

**Why PostgreSQL**:
- ACID compliance
- JSONB for flexible metadata
- Strong indexing
- Better for analytics vs NoSQL

### 4. Worker Pool (Celery)

**Architecture**:
```
Worker Process
├─ Task 1: Job ABC (spawns Docker container)
├─ Task 2: Job DEF (spawns Docker container)
└─ Task N: Job XYZ (spawns Docker container)
```

**Configuration**:
- Concurrency: 10 (10 parallel tasks per worker)
- Prefetch: 1 (don't hog tasks)
- Max tasks per child: 1 (fresh process per task, prevents memory leaks)

**Retry Strategy**:
```python
max_retries=2
retry_delay=60s
exponential_backoff=True

Failure → Wait 60s → Retry 1
Failure → Wait 120s → Retry 2
Failure → Mark as failed
```

### 5. Docker Executor

**Isolation Model**:
```
Host Machine
├─ Worker Process (persistent)
│   └─ Docker Container (ephemeral)
│       └─ Agent Code
│           ├─ Downloads data
│           ├─ Trains model
│           └─ Generates submission.csv
```

**Resource Limits**:
```yaml
CPU: 4 cores (cpu_quota=400000)
Memory: 8GB (mem_limit=8g)
Network: bridge mode (isolated)
Filesystem: Read-only except /output
Timeout: 2 hours (7200s)
```

**Security**:
- No privileged mode
- Drop unnecessary capabilities
- User namespace remapping
- Seccomp profile (default)

### 6. Agent (Inside Container)

**Pipeline**:
```
Stage 1: Competition Analysis (5-10 min)
├─ Download data via Kaggle API
├─ Parse CSV files
├─ Identify task type (classification/regression)
└─ Extract metadata

Stage 2: Strategy Planning (2-3 min)
├─ Query Claude for approach
├─ Select models (LightGBM/XGBoost)
├─ Plan feature engineering
└─ Define validation strategy

Stage 3: Code Generation (3-5 min)
├─ Generate training script (Claude)
├─ Fallback to templates if LLM fails
└─ Save to generated_solution.py

Stage 4: Model Training (20-40 min)
├─ Execute generated script
├─ Train model with cross-validation
├─ Generate predictions
└─ Save submission.csv

Total: ~30-60 minutes
```

**LLM Usage**:
- Model: Claude Sonnet 4.5
- Caching: Competition context cached for 5 min
- Fallback: Template-based generation if API fails
- Cost per job: ~$0.50-1.00

---

## Data Flow

### Job Creation Flow
```
1. Client → POST /run
2. API validates URL
3. API creates Job record (status=queued)
4. API enqueues Celery task
5. API returns job_id immediately
6. Client polls /status/{job_id}
```

### Job Execution Flow
```
1. Worker pulls task from queue
2. Worker updates Job (status=running)
3. Worker spawns Docker container
4. Container downloads competition data
5. Container analyzes data
6. Container generates strategy (LLM)
7. Container generates code (LLM)
8. Container trains model
9. Container saves submission.csv to /output
10. Worker extracts submission.csv
11. Worker updates Job (status=success, submission_path=...)
12. Worker cleans up container
```

---

## Scaling Strategy

### Vertical Scaling (Single Machine)
```
Current: 10 workers, 4 concurrent containers
Upgrade: 20 workers, 8 concurrent containers

Requirements:
- CPU: 32 cores (8 containers × 4 cores)
- RAM: 64GB (8 containers × 8GB)
- Storage: 500GB SSD
```

### Horizontal Scaling (Multi-Machine)
```
2× Worker Machines
├─ Machine 1: 10 workers
└─ Machine 2: 10 workers

Shared:
├─ PostgreSQL (single instance)
├─ Redis (single instance)
└─ NFS/S3 (shared storage)

Total Capacity: 20 workers, 8 concurrent containers
```

---

## Failure Handling

### Worker Crash
**Scenario**: Worker process crashes mid-task

**Recovery**:
1. Celery marks task as "lost"
2. Task requeued automatically (acks_late=True)
3. Another worker picks up task
4. Job retried from beginning

### Container Crash
**Scenario**: Docker container exits with error

**Recovery**:
1. Worker detects non-zero exit code
2. Worker updates Job (status=failed)
3. Celery retry mechanism kicks in
4. New container spawned (max 2 retries)

### Database Unavailable
**Scenario**: PostgreSQL connection fails

**Recovery**:
1. SQLAlchemy connection pool retries
2. If persistent failure, task fails
3. Job marked as failed in Redis (Celery backend)
4. Manual intervention required

---

## Security Considerations

### Container Escape Prevention
- No privileged mode
- AppArmor/SELinux profiles
- Seccomp filters
- User namespace isolation

### API Security
- Rate limiting (10 req/min per IP)
- Input validation (URL whitelist)
- CORS restrictions
- Optional API key authentication

### Secrets Management
- Environment variables (not in code)
- Docker secrets (production)
- Rotate credentials monthly
- Separate keys per environment

---

## Trade-offs & Decisions

### Why Celery over Custom Queue?
**Decision**: Use Celery

**Rationale**:
- Battle-tested (10+ years)
- Built-in retry logic
- Monitoring tools (Flower)
- Don't reinvent the wheel

**Trade-off**: Learning curve, but worth it

### Why Docker over Kubernetes?
**Decision**: Use Docker (for now)

**Rationale**:
- Simpler setup (3 days vs 1 week)
- Sufficient for 50 concurrent jobs
- Easy local development
- K8s migration path clear

**Trade-off**: Manual scaling, but acceptable

### Why PostgreSQL over MongoDB?
**Decision**: Use PostgreSQL

**Rationale**:
- Strong consistency (ACID)
- Better for time-series queries
- JSONB for flexibility
- Mature ecosystem

**Trade-off**: Slightly slower writes, but negligible

### Why LLM over Rule-Based?
**Decision**: Use LLM (Claude)

**Rationale**:
- Adapts to new competition types
- Better code quality
- Less maintenance
- Competitive advantage

**Trade-off**: API costs ($0.80/job), but worth it

---

## Cost Analysis

### Per-Job Cost Breakdown
```
LLM API Calls (Claude):
  - Strategy planning: ~$0.30
  - Code generation: ~$0.50
  Total: ~$0.80

Compute (AWS t3.xlarge):
  - 60 min × $0.166/hr = $0.166

Storage:
  - Data: ~100MB
  - Submission: ~1MB
  Total: $0.001

Total per job: ~$0.97
```

### Monthly Cost (1000 jobs)
```
Infrastructure:
  - API servers (2× t3.medium): $60
  - Worker servers (2× t3.xlarge): $240
  - PostgreSQL (db.t3.small): $25
  - Redis (cache.t3.small): $15
  - Storage (500GB EBS): $50
  Subtotal: $390

Per-Job Costs:
  - 1000 jobs × $0.97 = $970

Total: $1,360/month
```

### Cost Optimization
1. **Spot Instances**: -70% on worker costs → Save $168/month
2. **LLM Caching**: -30% on API calls → Save $240/month
3. **Reserved Instances**: -40% on fixed infrastructure → Save $156/month

**Total Savings**: ~$564/month (41%)

