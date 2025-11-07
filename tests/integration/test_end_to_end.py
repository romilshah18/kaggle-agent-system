import requests
import time
import sys

API_BASE = "http://localhost:8000"
TEST_URL = "https://www.kaggle.com/competitions/titanic"


def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    response = requests.get(f"{API_BASE}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print("✓ Health check passed")


def test_create_job():
    """Test job creation"""
    print(f"\nTesting /run with URL: {TEST_URL}")
    response = requests.post(
        f"{API_BASE}/run",
        json={"kaggle_url": TEST_URL}
    )
    assert response.status_code == 201
    data = response.json()
    assert 'job_id' in data
    job_id = data['job_id']
    print(f"✓ Job created: {job_id}")
    return job_id


def test_job_status(job_id):
    """Test status endpoint"""
    print(f"\nTesting /status/{job_id}")
    response = requests.get(f"{API_BASE}/status/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data['job_id'] == job_id
    print(f"✓ Status: {data['status']}, Progress: {data.get('progress')}")
    return data['status']


def test_wait_for_completion(job_id, timeout=3600):
    """Wait for job to complete"""
    print(f"\nWaiting for job {job_id} to complete...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = test_job_status(job_id)
        
        if status == 'success':
            print("✓ Job completed successfully!")
            return True
        elif status in ['failed', 'timeout']:
            print(f"✗ Job {status}")
            return False
        
        time.sleep(30)  # Check every 30 seconds
    
    print("✗ Timeout waiting for job completion")
    return False


def test_download_submission(job_id):
    """Test submission download"""
    print(f"\nTesting /result/{job_id}/submission.csv")
    response = requests.get(f"{API_BASE}/result/{job_id}/submission.csv")
    
    if response.status_code == 200:
        print(f"✓ Submission downloaded ({len(response.content)} bytes)")
        # Save locally for inspection
        with open('submission_test.csv', 'wb') as f:
            f.write(response.content)
        print("✓ Saved to submission_test.csv")
        return True
    else:
        print(f"✗ Download failed: {response.status_code}")
        return False


def main():
    print("="*60)
    print("END-TO-END INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test sequence
        test_health()
        job_id = test_create_job()
        
        # Wait for completion
        success = test_wait_for_completion(job_id, timeout=3600)
        
        if success:
            test_download_submission(job_id)
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("✗ TEST FAILED")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n✗ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

