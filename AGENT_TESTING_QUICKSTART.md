# 🚀 Agent Testing Quick Start

**Fast iteration without Docker!** Test agent logic locally in seconds instead of minutes.

## One-Command Test

```bash
./tests/unit/test_agent_quick.sh
```

That's it! 🎉

## What It Does

Tests the complete agent workflow on Titanic competition:

1. **Analyze** → Downloads data, infers task type, scrapes competition details
2. **Plan** → Uses Claude to plan strategy (models, features, validation)
3. **Generate** → Creates Python training code
4. **Execute** → Runs the code and produces `submission.csv`

Results saved to: `storage/submissions/test-standalone/`

## Prerequisites

```bash
# 1. Install dependencies (one time)
pip install -r requirements.txt

# 2. Set up .env (one time)
cat >> .env << EOF
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
ANTHROPIC_API_KEY=your_claude_key
EOF
```

## Inspect Results

```bash
# View generated training code
cat storage/submissions/test-standalone/generated_model.py

# View predictions
head storage/submissions/test-standalone/submission.csv

# Run manually if needed
cd storage/submissions/test-standalone
python generated_model.py
```

## Development Loop

**Old way (with Docker):**
1. Edit code → 2 min
2. Build Docker → 5 min
3. Deploy → 2 min
4. Test → 3 min
5. Debug → repeat
**Total: ~12 min per iteration** 😴

**New way (standalone):**
1. Edit code → 2 min
2. Run test → 30 sec
3. Debug → repeat
**Total: ~2.5 min per iteration** ⚡

## When to Use What

| Test Type | When | Command |
|-----------|------|---------|
| **Standalone** | Developing agent logic | `./tests/unit/test_agent_quick.sh` |
| **Docker** | Testing full integration | `./infrastructure/scripts/deploy.sh` |
| **Production** | Real deployment | Deploy to cloud |

## Common Fixes

### Fix 1: Module Import Errors

```python
# In your code, use absolute imports:
from agent.analyzer.competition_analyzer import CompetitionAnalyzer

# NOT relative imports like:
from analyzer.competition_analyzer import CompetitionAnalyzer
```

### Fix 2: Path Issues

```python
# Use Path objects and resolve paths properly
from pathlib import Path
output_dir = Path("/app/storage/submissions") / job_id
output_dir.mkdir(parents=True, exist_ok=True)
```

### Fix 3: Environment Variables

```bash
# Load .env before running
source .env
./tests/unit/test_agent_quick.sh

# Or use python-dotenv in your code
from dotenv import load_dotenv
load_dotenv()
```

## Next Steps

1. ✅ Run standalone test and verify it works
2. ✅ Fix any issues (errors will be in console output)
3. ✅ Once working, test in Docker: `./infrastructure/scripts/deploy.sh`
4. ✅ Submit test job via API

## Pro Tips

- 💡 Keep a terminal watching the test output
- 💡 Modify agent code and rerun test immediately
- 💡 Check generated code when training fails
- 💡 Use `pdb` for debugging: add `import pdb; pdb.set_trace()` in agent code
- 💡 Only rebuild Docker after standalone tests pass

---

**Full docs:** See `TESTING.md` for comprehensive testing guide

