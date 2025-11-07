from pydantic import BaseModel, HttpUrl, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class JobCreate(BaseModel):
    kaggle_url: HttpUrl = Field(..., description="Kaggle competition URL")


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    message: str = "Job created successfully"
    
    model_config = ConfigDict(from_attributes=True)


class JobStatusResponse(BaseModel):
    job_id: str
    kaggle_url: str
    competition_name: Optional[str] = None
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    model_config = ConfigDict(from_attributes=True)


class JobDetailResponse(JobStatusResponse):
    submission_path: Optional[str] = None
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    queue_length: int

