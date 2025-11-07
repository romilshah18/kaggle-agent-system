import asyncio
import aiohttp
import time
from datetime import datetime
import json

API_BASE = "http://localhost:8000"
TEST_URL = "https://www.kaggle.com/competitions/titanic"


class LoadTester:
    def __init__(self, num_concurrent: int = 50):
        self.num_concurrent = num_concurrent
        self.results = []
        
    async def submit_job(self, session, job_num):
        """Submit a single job"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{API_BASE}/run",
                json={"kaggle_url": TEST_URL},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                
                if response.status == 201:
                    data = await response.json()
                    return {
                        "job_num": job_num,
                        "success": True,
                        "job_id": data.get('job_id'),
                        "response_time": end_time - start_time,
                        "status_code": response.status
                    }
                else:
                    return {
                        "job_num": job_num,
                        "success": False,
                        "response_time": end_time - start_time,
                        "status_code": response.status,
                        "error": await response.text()
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                "job_num": job_num,
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    async def run_load_test(self):
        """Run load test with concurrent requests"""
        print(f"\n{'='*60}")
        print(f"LOAD TEST: {self.num_concurrent} CONCURRENT REQUESTS")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.submit_job(session, i+1)
                for i in range(self.num_concurrent)
            ]
            
            # Execute all concurrently
            self.results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        self.print_results(total_time)
        
        return self.results
    
    def print_results(self, total_time):
        """Print load test results"""
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        response_times = [r['response_time'] for r in self.results]
        avg_response = sum(response_times) / len(response_times)
        min_response = min(response_times)
        max_response = max(response_times)
        
        print(f"\n{'='*60}")
        print(f"LOAD TEST RESULTS")
        print(f"{'='*60}")
        print(f"\nTotal Requests:     {self.num_concurrent}")
        print(f"Successful:         {len(successful)} ({len(successful)/self.num_concurrent*100:.1f}%)")
        print(f"Failed:             {len(failed)} ({len(failed)/self.num_concurrent*100:.1f}%)")
        print(f"\nTotal Time:         {total_time:.2f}s")
        print(f"Requests/sec:       {self.num_concurrent/total_time:.2f}")
        print(f"\nResponse Times:")
        print(f"  Average:          {avg_response:.3f}s")
        print(f"  Min:              {min_response:.3f}s")
        print(f"  Max:              {max_response:.3f}s")
        
        if failed:
            print(f"\nFailed Requests:")
            for r in failed[:5]:  # Show first 5 failures
                print(f"  Job {r['job_num']}: {r.get('error', 'Unknown error')}")
        
        # Check queue status
        print(f"\n{'='*60}")
        
        # Save results
        with open('load_test_results.json', 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "num_concurrent": self.num_concurrent,
                "total_time": total_time,
                "successful": len(successful),
                "failed": len(failed),
                "avg_response_time": avg_response,
                "results": self.results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to load_test_results.json")


async def check_queue_status():
    """Check queue status after load test"""
    print(f"\nChecking queue status...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    queue_length = data.get('queue_length', 'unknown')
                    print(f"Queue Length: {queue_length}")
                    return queue_length
    except Exception as e:
        print(f"Failed to check queue: {e}")
        return None


async def main():
    # Test different concurrency levels
    concurrency_levels = [10, 25, 50]
    
    for level in concurrency_levels:
        tester = LoadTester(num_concurrent=level)
        await tester.run_load_test()
        
        # Check queue
        await check_queue_status()
        
        # Wait between tests
        if level != concurrency_levels[-1]:
            print(f"\nWaiting 10 seconds before next test...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())

