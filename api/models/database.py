from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"
    
    job_id = Column(String(36), primary_key=True, index=True)
    kaggle_url = Column(Text, nullable=False)
    competition_name = Column(String(255), index=True)
    status = Column(String(20), nullable=False, index=True, default="queued")
    # Status: queued, running, success, failed, timeout
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    celery_task_id = Column(String(36), index=True, nullable=True)
    
    submission_path = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    
    job_metadata = Column(JSON, default=dict)
    # Stores: logs, metrics, progress updates, etc.
    
    def __repr__(self):
        return f"<Job {self.job_id} - {self.status}>"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)

