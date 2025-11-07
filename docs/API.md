# API Reference

Base URL: `http://localhost:8000`

## Authentication
Currently no authentication required. For production, add API key:
```bash
curl -H "X-API-Key: your-key" http://localhost:8000/run
```

## Endpoints

### POST /run
Create new competition job

**Request**:
```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/competitions/titanic"}'
```

**Response** (201 Created):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2024-01-01T12:00:00Z",
  "message": "Job created successfully. Check status at /status/{job_id}"
}
```

**Error** (400 Bad Request):
```json
{
  "detail": "URL must be a Kaggle competition URL"
}
```

### GET /status/{job_id}
Check job status

**Request**:
```bash
curl "http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000"
```

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "kaggle_url": "https://www.kaggle.com/competitions/titanic",
  "competition_name": "titanic",
  "status": "running",
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:05Z",
  "completed_at": null,
  "progress": "Training model (Stage 4/4)",
  "metadata": {
    "progress": "Training model with LightGBM",
    "logs_preview": "CV Accuracy: 0.8234 (+/- 0.0156)..."
  }
}
```

**Status Values**:
- `queued`: Waiting in queue
- `running`: Currently executing
- `success`: Completed successfully
- `failed`: Execution failed
- `timeout`: Exceeded time limit

### GET /result/{job_id}/submission.csv
Download submission file

**Request**:
```bash
curl "http://localhost:8000/result/550e8400-e29b-41d4-a716-446655440000/submission.csv" \
  -o submission.csv
```

**Response** (200 OK):
```
Content-Type: text/csv
Content-Disposition: attachment; filename="submission.csv"

[CSV content]
```

**Error** (400 Bad Request):
```json
{
  "detail": "Job is not complete. Current status: running"
}
```

### GET /logs/{job_id}
View execution logs

**Request**:
```bash
curl "http://localhost:8000/logs/550e8400-e29b-41d4-a716-446655440000"
```

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "logs": "2024-01-01 12:00:05 - INFO - KAGGLE AGENT STARTED\n..."
}
```

### GET /jobs
List all jobs

**Request**:
```bash
# All jobs
curl "http://localhost:8000/jobs"

# Filter by status
curl "http://localhost:8000/jobs?status_filter=success&limit=10"
```

**Response** (200 OK):
```json
{
  "total": 15,
  "jobs": [
    {
      "job_id": "...",
      "kaggle_url": "...",
      "competition_name": "titanic",
      "status": "success",
      "created_at": "2024-01-01T12:00:00Z",
      "started_at": "2024-01-01T12:00:05Z",
      "completed_at": "2024-01-01T12:35:22Z",
      "progress": "Completed successfully",
      "metadata": {}
    }
  ]
}
```

### GET /health
System health check

**Request**:
```bash
curl "http://localhost:8000/health"
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "api": "healthy",
    "redis": "healthy",
    "database": "healthy"
  },
  "queue_length": 5
}
```

## Rate Limiting
- Default: 10 requests per minute per IP
- Exceeded: 429 Too Many Requests

## Error Responses

### 400 Bad Request
Invalid input

### 404 Not Found
Resource not found

### 429 Too Many Requests
Rate limit exceeded

### 500 Internal Server Error
Server error

### 503 Service Unavailable
System overloaded

