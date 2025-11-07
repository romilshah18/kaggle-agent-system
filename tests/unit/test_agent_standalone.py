#!/usr/bin/env python3
"""
Standalone Agent Test - Run agent logic locally without Docker
Usage: python tests/unit/test_agent_standalone.py
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.analyzer.competition_analyzer import CompetitionAnalyzer
from agent.planner.strategy_planner import StrategyPlanner
from agent.generator.code_generator import CodeGenerator
from agent.executor.model_executor import ModelExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_agent_workflow():
    """Test the complete agent workflow for Titanic competition"""
    
    # Configuration
    job_id = "test-standalone"
    kaggle_url = "https://www.kaggle.com/competitions/titanic"
    output_dir = project_root / "storage" / "submissions" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("STANDALONE AGENT TEST")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Competition: {kaggle_url}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)
    
    # Check environment variables
    required_vars = ["KAGGLE_USERNAME", "KAGGLE_KEY", "ANTHROPIC_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"❌ Missing environment variables: {missing}")
        logger.error("Please set them in your .env file and source it:")
        logger.error("  export KAGGLE_USERNAME=your_username")
        logger.error("  export KAGGLE_KEY=your_key")
        logger.error("  export ANTHROPIC_API_KEY=your_key")
        return False
    
    try:
        # Stage 1: Analyze Competition
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: ANALYZING COMPETITION")
        logger.info("="*60)
        analyzer = CompetitionAnalyzer()
        competition_info = analyzer.analyze(kaggle_url, str(output_dir))
        
        if not competition_info:
            logger.error("❌ Failed to analyze competition")
            return False
        
        logger.info(f"✓ Competition: {competition_info.get('name', 'Unknown')}")
        logger.info(f"✓ Task Type: {competition_info.get('task_type', 'Unknown')}")
        logger.info(f"✓ Metric: {competition_info.get('metric', 'Unknown')}")
        
        # Stage 2: Plan Strategy
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: PLANNING STRATEGY")
        logger.info("="*60)
        planner = StrategyPlanner()
        strategy = planner.plan(competition_info)
        
        if not strategy:
            logger.error("❌ Failed to plan strategy")
            return False
        
        logger.info(f"✓ Approach: {strategy.get('approach', 'N/A')[:100]}...")
        logger.info(f"✓ Models: {strategy.get('models', 'N/A')[:100]}...")
        
        # Stage 3: Generate Code
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: GENERATING CODE")
        logger.info("="*60)
        generator = CodeGenerator()
        code = generator.generate(competition_info, strategy, str(output_dir))
        
        if not code:
            logger.error("❌ Failed to generate code")
            return False
        
        # Save generated code for inspection
        code_path = output_dir / "generated_model.py"
        with open(code_path, 'w') as f:
            f.write(code)
        logger.info(f"✓ Code saved to: {code_path}")
        logger.info(f"✓ Code length: {len(code)} characters")
        
        # Stage 4: Execute Training
        logger.info("\n" + "="*60)
        logger.info("STAGE 4: EXECUTING TRAINING")
        logger.info("="*60)
        executor = ModelExecutor()
        success = executor.execute(code, str(output_dir))
        
        if not success:
            logger.error("❌ Training failed")
            return False
        
        # Check submission file
        submission_path = output_dir / "submission.csv"
        if submission_path.exists():
            logger.info(f"✓ Submission created: {submission_path}")
            with open(submission_path, 'r') as f:
                lines = f.readlines()
                logger.info(f"✓ Submission has {len(lines)} lines")
                logger.info(f"✓ First few lines:\n{''.join(lines[:5])}")
        else:
            logger.error("❌ submission.csv not found")
            return False
        
        # Success!
        logger.info("\n" + "="*60)
        logger.info("🎉 AGENT TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Results in: {output_dir}")
        logger.info(f"- generated_model.py (training code)")
        logger.info(f"- submission.csv (predictions)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Load .env if available
    try:
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"✓ Loaded environment from {env_path}")
    except ImportError:
        logger.warning("python-dotenv not installed, make sure env vars are set manually")
    
    # Run test
    success = test_agent_workflow()
    sys.exit(0 if success else 1)

