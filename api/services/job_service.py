from sqlalchemy.orm import Session
from api.models.database import Job
from api.models.schemas import JobStatus
from typing import Optional, List
from datetime import datetime
import uuid


class JobService:
    @staticmethod
    def create_job(db: Session, kaggle_url: str) -> Job:
        job_id = str(uuid.uuid4())
        
        # Extract competition name from URL
        competition_name = kaggle_url.rstrip('/').split('/')[-1]
        
        job = Job(
            job_id=job_id,
            kaggle_url=kaggle_url,
            competition_name=competition_name,
            status=JobStatus.QUEUED,
            created_at=datetime.utcnow(),
            job_metadata={"progress": "Job queued"}
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        return job
    
    @staticmethod
    def get_job(db: Session, job_id: str) -> Optional[Job]:
        return db.query(Job).filter(Job.job_id == job_id).first()
    
    @staticmethod
    def update_job_status(
        db: Session, 
        job_id: str, 
        status: JobStatus,
        error_message: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Optional[Job]:
        job = JobService.get_job(db, job_id)
        if not job:
            return None
        
        job.status = status
        
        if status == JobStatus.RUNNING and not job.started_at:
            job.started_at = datetime.utcnow()
        
        if status in [JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.TIMEOUT]:
            job.completed_at = datetime.utcnow()
        
        if error_message:
            job.error_message = error_message
        
        if metadata:
            job.job_metadata.update(metadata)
        
        db.commit()
        db.refresh(job)
        
        return job
    
    @staticmethod
    def set_celery_task_id(db: Session, job_id: str, task_id: str):
        job = JobService.get_job(db, job_id)
        if job:
            job.celery_task_id = task_id
            db.commit()
    
    @staticmethod
    def set_submission_path(db: Session, job_id: str, path: str):
        job = JobService.get_job(db, job_id)
        if job:
            job.submission_path = path
            db.commit()
    
    @staticmethod
    def get_recent_jobs(db: Session, limit: int = 100) -> List[Job]:
        return db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_jobs_by_status(db: Session, status: JobStatus) -> List[Job]:
        return db.query(Job).filter(Job.status == status).all()

