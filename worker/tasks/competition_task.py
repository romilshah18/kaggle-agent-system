from celery import Task
from worker.celery_app import celery_app
from worker.executors.docker_executor import DockerExecutor
from api.models.database import SessionLocal
from api.services.job_service import JobService
from api.models.schemas import JobStatus
import logging
import os
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class CompetitionTask(Task):
    """Base task with database session management"""
    
    def __init__(self):
        super().__init__()
        self._docker_executor = None
    
    @property
    def docker_executor(self):
        if self._docker_executor is None:
            self._docker_executor = DockerExecutor()
        return self._docker_executor


@celery_app.task(
    base=CompetitionTask,
    bind=True,
    name='worker.tasks.process_competition',
    max_retries=2,
    default_retry_delay=60
)
def process_competition(self, job_id: str, kaggle_url: str):
    """
    Main task to process a Kaggle competition
    
    Steps:
    1. Update job status to RUNNING
    2. Spawn Docker container with agent
    3. Monitor execution
    4. Extract results
    5. Update job status with results
    """
    db = SessionLocal()
    
    try:
        logger.info(f"Starting job {job_id} for competition: {kaggle_url}")
        
        # Update status to RUNNING
        JobService.update_job_status(
            db, 
            job_id, 
            JobStatus.RUNNING,
            metadata={"progress": "Initializing agent container"}
        )
        
        # Get credentials from environment
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not all([kaggle_username, kaggle_key, anthropic_api_key]):
            raise ValueError("Missing required API credentials")
        
        # Update progress
        JobService.update_job_status(
            db,
            job_id,
            JobStatus.RUNNING,
            metadata={"progress": "Running agent in Docker container"}
        )
        
        # Execute agent in Docker
        result = self.docker_executor.run_agent(
            job_id=job_id,
            kaggle_url=kaggle_url,
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key,
            anthropic_api_key=anthropic_api_key,
            timeout=int(os.getenv('CONTAINER_TIMEOUT', 7200))
        )
        
        # Process results
        if result['success']:
            logger.info(f"✓ Job {job_id} completed successfully")
            
            # Update job with success
            JobService.update_job_status(
                db,
                job_id,
                JobStatus.SUCCESS,
                metadata={
                    "progress": "Completed successfully",
                    "exit_code": result['exit_code'],
                    "logs_preview": result['logs'][-500:]  # Last 500 chars
                }
            )
            
            # Set submission path
            if result['submission_path']:
                JobService.set_submission_path(db, job_id, result['submission_path'])
            
        else:
            logger.error(f"✗ Job {job_id} failed with exit code {result['exit_code']}")
            
            # Determine if timeout or failure
            if "timeout" in result['logs'].lower() or "exceeded timeout" in result['logs'].lower():
                status = JobStatus.TIMEOUT
            else:
                status = JobStatus.FAILED
            
            JobService.update_job_status(
                db,
                job_id,
                status,
                error_message=f"Exit code: {result['exit_code']}",
                metadata={
                    "progress": f"Failed with exit code {result['exit_code']}",
                    "exit_code": result['exit_code'],
                    "logs_preview": result['logs'][-500:]
                }
            )
        
        return {
            "job_id": job_id,
            "success": result['success'],
            "exit_code": result['exit_code']
        }
        
    except Exception as e:
        logger.error(f"Exception in job {job_id}: {str(e)}", exc_info=True)
        
        # Update job status to FAILED
        JobService.update_job_status(
            db,
            job_id,
            JobStatus.FAILED,
            error_message=str(e),
            metadata={"progress": f"Exception: {str(e)}"}
        )
        
        # Retry on transient errors
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 1})")
            raise self.retry(exc=e)
        
        raise
        
    finally:
        db.close()

