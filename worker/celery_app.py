from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Celery
celery_app = Celery(
    'kaggle_agent',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1'),
    include=['worker.tasks.competition_task']
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Worker configuration
    worker_prefetch_multiplier=1,  # Don't hog tasks
    worker_max_tasks_per_child=1,  # Fresh worker per task (memory safety)
    
    # Task configuration
    task_time_limit=int(os.getenv('CELERY_TASK_TIME_LIMIT', 7200)),  # 2 hours hard limit
    task_soft_time_limit=int(os.getenv('CELERY_TASK_SOFT_TIME_LIMIT', 6900)),  # 115 min warning
    task_acks_late=True,  # Acknowledge after completion
    task_reject_on_worker_lost=True,
    
    # Result backend
    result_expires=86400,  # Results expire after 24 hours
    
    # Retry configuration
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=2,
)

if __name__ == '__main__':
    celery_app.start()

