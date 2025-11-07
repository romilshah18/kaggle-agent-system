from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import os
from pathlib import Path

from api.models.database import get_db, init_db
from api.models.schemas import (
    JobCreate, JobResponse, JobStatusResponse, 
    JobDetailResponse, HealthResponse, JobStatus
)
from api.services.job_service import JobService
from worker.celery_app import celery_app
from worker.tasks.competition_task import process_competition

# Initialize FastAPI
app = FastAPI(
    title="Kaggle Competition Agent API",
    description="Autonomous agent for Kaggle competitions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("✓ Database initialized")
    
    # Ensure storage directories exist
    os.makedirs("/app/storage/submissions", exist_ok=True)
    os.makedirs("/app/storage/logs", exist_ok=True)
    print("✓ Storage directories ready")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_status = "healthy" if celery_app.control.inspect().active() is not None else "unhealthy"
    except:
        redis_status = "unhealthy"
    
    # Get queue length
    try:
        queue_length = celery_app.control.inspect().reserved()
        queue_length = sum(len(tasks) for tasks in queue_length.values()) if queue_length else 0
    except:
        queue_length = -1
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        services={
            "api": "healthy",
            "redis": redis_status,
            "database": "healthy"
        },
        queue_length=queue_length
    )


@app.post("/run", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    job_create: JobCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new Kaggle competition job
    
    - **kaggle_url**: Full URL to Kaggle competition
    """
    # Validate URL is Kaggle competition
    url_str = str(job_create.kaggle_url)
    if "kaggle.com/competitions/" not in url_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must be a Kaggle competition URL"
        )
    
    # Create job in database
    job = JobService.create_job(db, url_str)
    
    # Submit to Celery
    task = process_competition.apply_async(
        args=[job.job_id, url_str],
        task_id=job.job_id  # Use job_id as task_id for easier tracking
    )
    
    # Update job with celery task ID
    JobService.set_celery_task_id(db, job.job_id, task.id)
    
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        message=f"Job created successfully. Check status at /status/{job.job_id}"
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Get job status"""
    job = JobService.get_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Get progress from metadata
    progress = job.job_metadata.get("progress", "No progress information")
    
    return JobStatusResponse(
        job_id=job.job_id,
        kaggle_url=job.kaggle_url,
        competition_name=job.competition_name,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=progress,
        metadata=job.job_metadata
    )


@app.get("/result/{job_id}/submission.csv")
async def get_submission(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Download submission.csv for completed job"""
    job = JobService.get_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    if job.status != JobStatus.SUCCESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not complete. Current status: {job.status}"
        )
    
    if not job.submission_path or not Path(job.submission_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission file not found"
        )
    
    return FileResponse(
        path=job.submission_path,
        filename="submission.csv",
        media_type="text/csv"
    )


@app.get("/logs/{job_id}")
async def get_job_logs(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Get job execution logs"""
    job = JobService.get_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    log_path = f"/app/storage/logs/{job_id}.log"
    
    if not Path(log_path).exists():
        return JSONResponse(
            content={"job_id": job_id, "logs": "No logs available yet"}
        )
    
    with open(log_path, 'r') as f:
        logs = f.read()
    
    return JSONResponse(
        content={"job_id": job_id, "logs": logs}
    )


@app.get("/jobs")
async def list_jobs(
    status_filter: Optional[JobStatus] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all jobs with optional status filter"""
    if status_filter:
        jobs = JobService.get_jobs_by_status(db, status_filter)
    else:
        jobs = JobService.get_recent_jobs(db, limit)
    
    return {
        "total": len(jobs),
        "jobs": [
            JobStatusResponse(
                job_id=job.job_id,
                kaggle_url=job.kaggle_url,
                competition_name=job.competition_name,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                progress=job.job_metadata.get("progress"),
                metadata=job.job_metadata
            )
            for job in jobs
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

