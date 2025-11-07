#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

from api.models.database import init_db, engine
from sqlalchemy import text

def main():
    print("Initializing database...")
    
    # Create tables
    init_db()
    print("✓ Tables created")
    
    # Create indexes
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created 
            ON jobs(status, created_at DESC);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_jobs_competition 
            ON jobs(competition_name);
        """))
        conn.commit()
    print("✓ Indexes created")
    
    print("Database initialization complete!")

if __name__ == "__main__":
    main()

