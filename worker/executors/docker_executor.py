import docker
from docker.errors import ContainerError, ImageNotFound, APIError
import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DockerExecutor:
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.info("✓ Docker client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def build_agent_image(self) -> bool:
        """Build agent Docker image if not exists"""
        try:
            self.client.images.get("kaggle-agent:latest")
            logger.info("✓ Agent image already exists")
            return True
        except ImageNotFound:
            logger.info("Building agent image...")
            
            dockerfile_path = Path(__file__).parent.parent.parent / "infrastructure/docker"
            
            try:
                image, build_logs = self.client.images.build(
                    path=str(dockerfile_path.parent.parent),
                    dockerfile=str(dockerfile_path / "Dockerfile.agent"),
                    tag="kaggle-agent:latest",
                    rm=True
                )
                
                for log in build_logs:
                    if 'stream' in log:
                        logger.info(log['stream'].strip())
                
                logger.info("✓ Agent image built successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to build agent image: {e}")
                return False
    
    def run_agent(
        self,
        job_id: str,
        kaggle_url: str,
        kaggle_username: str,
        kaggle_key: str,
        anthropic_api_key: str,
        timeout: int = 7200
    ) -> Dict[str, Any]:
        """
        Run agent in isolated Docker container
        
        Returns:
            Dict with keys: success (bool), exit_code (int), logs (str), submission_path (str)
        """
        # Ensure agent image exists
        if not self.build_agent_image():
            return {
                "success": False,
                "exit_code": -1,
                "logs": "Failed to build agent image",
                "submission_path": None
            }
        
        # Prepare output directory
        # The worker is inside a container with ../storage mounted at /app/storage
        # We need to create the directory in /app/storage (which is accessible)
        internal_output_dir = f"/app/storage/submissions/{job_id}"
        internal_log_dir = f"/app/storage/logs"
        os.makedirs(internal_output_dir, exist_ok=True)
        os.makedirs(internal_log_dir, exist_ok=True)
        
        # Get the actual host path for the storage directory
        # This is passed as an environment variable from docker-compose
        host_storage_path = os.getenv('STORAGE_HOST_PATH')
        
        # If not set, try to determine it (fallback)
        if not host_storage_path:
            # Default to the project root + storage
            host_storage_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../storage'))
            logger.warning(f"STORAGE_HOST_PATH not set, using: {host_storage_path}")
        
        # Container configuration
        container_config = {
            "image": "kaggle-agent:latest",
            "command": [
                "--job-id", job_id,
                "--url", kaggle_url
            ],
            "environment": {
                "KAGGLE_USERNAME": kaggle_username,
                "KAGGLE_KEY": kaggle_key,
                "ANTHROPIC_API_KEY": anthropic_api_key,
                "PYTHONUNBUFFERED": "1"
            },
            "volumes": {
                host_storage_path: {"bind": "/app/storage", "mode": "rw"}
            },
            "mem_limit": os.getenv("CONTAINER_MEMORY_LIMIT", "8g"),
            "cpu_quota": int(os.getenv("CONTAINER_CPU_LIMIT", "4")) * 100000,
            "network_mode": "bridge",
            "detach": True,
            "remove": False,  # Keep for log extraction
            "name": f"kaggle-agent-{job_id}"
        }
        
        container = None
        logs = ""
        
        try:
            logger.info(f"Starting container for job {job_id}")
            container = self.client.containers.run(**container_config)
            
            # Stream logs with timeout
            start_time = time.time()
            log_lines = []
            
            for line in container.logs(stream=True, follow=True):
                if time.time() - start_time > timeout:
                    logger.warning(f"Job {job_id} exceeded timeout {timeout}s")
                    container.kill()
                    break
                
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                log_lines.append(decoded_line)
                logger.info(f"[{job_id}] {decoded_line}")
            
            logs = "\n".join(log_lines)
            
            # Wait for container to finish
            result = container.wait(timeout=10)
            exit_code = result.get('StatusCode', -1)
            
            # Save logs to file
            log_path = f"{internal_log_dir}/{job_id}.log"
            with open(log_path, 'w') as f:
                f.write(logs)
            
            # Check for submission file
            # The agent writes to /output/submission.csv but we mounted /app/storage
            # So it actually writes to /app/storage/submissions/{job_id}/submission.csv
            submission_path = f"{internal_output_dir}/submission.csv"
            submission_exists = os.path.exists(submission_path)
            
            success = exit_code == 0 and submission_exists
            
            if not submission_exists and exit_code == 0:
                logs += "\nWARNING: Container exited successfully but no submission.csv found"
            
            return {
                "success": success,
                "exit_code": exit_code,
                "logs": logs,
                "submission_path": submission_path if submission_exists else None
            }
            
        except ContainerError as e:
            logger.error(f"Container error for job {job_id}: {e}")
            return {
                "success": False,
                "exit_code": e.exit_status,
                "logs": f"Container error: {str(e)}\n{logs}",
                "submission_path": None
            }
            
        except APIError as e:
            logger.error(f"Docker API error for job {job_id}: {e}")
            return {
                "success": False,
                "exit_code": -1,
                "logs": f"Docker API error: {str(e)}\n{logs}",
                "submission_path": None
            }
            
        except Exception as e:
            logger.error(f"Unexpected error for job {job_id}: {e}")
            return {
                "success": False,
                "exit_code": -1,
                "logs": f"Unexpected error: {str(e)}\n{logs}",
                "submission_path": None
            }
            
        finally:
            # Cleanup container
            if container:
                try:
                    container.remove(force=True)
                    logger.info(f"✓ Cleaned up container for job {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove container for job {job_id}: {e}")
    
    def cleanup_old_containers(self, max_age_hours: int = 24):
        """Remove old containers to prevent accumulation"""
        try:
            containers = self.client.containers.list(
                all=True,
                filters={"name": "kaggle-agent-"}
            )
            
            cleaned = 0
            for container in containers:
                try:
                    container.remove(force=True)
                    cleaned += 1
                except:
                    pass
            
            if cleaned > 0:
                logger.info(f"✓ Cleaned up {cleaned} old containers")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old containers: {e}")

