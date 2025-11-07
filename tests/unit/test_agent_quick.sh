#!/bin/bash
# Quick test script for agent development
# This runs the agent locally without Docker for faster iteration

set -e

cd "$(dirname "$0")/../.."

echo "🧪 Standalone Agent Test"
echo "========================"
echo ""

# Check .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found"
    echo "Please create .env with:"
    echo "  KAGGLE_USERNAME=your_username"
    echo "  KAGGLE_KEY=your_key"
    echo "  ANTHROPIC_API_KEY=your_key"
    exit 1
fi

# Load .env
export $(cat .env | grep -v '^#' | xargs)

# Check Python environment
echo "📦 Checking Python environment..."
if ! python3 -c "import pandas, numpy, sklearn, anthropic, kaggle" 2>/dev/null; then
    echo "⚠️  Missing dependencies. Installing..."
    pip3 install -q pandas numpy scikit-learn xgboost lightgbm anthropic requests beautifulsoup4 lxml kaggle python-dotenv
fi

# Create storage directory
mkdir -p storage/submissions/test-standalone

# Run test
echo ""
echo "🚀 Running agent test..."
echo ""
python3 tests/unit/test_agent_standalone.py

echo ""
echo "✅ Test complete! Check storage/submissions/test-standalone/ for results"

