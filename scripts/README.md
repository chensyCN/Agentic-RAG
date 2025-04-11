# Agentic-RAG Scripts

This directory contains scripts for running and evaluating the Agentic-RAG system on various datasets.

## Available Scripts

### run.sh

Runs evaluations on all available datasets in the `dataset` directory.

```bash
# Run with default parameters
./scripts/run.sh

# You can also modify the script to change:
# - MAX_ROUNDS: Number of agent rounds (default: 3)
# - TOP_K: Number of top contexts to retrieve (default: 5)
# - EVAL_TOP_KS: List of k values for evaluation (default: "5 10 20")
# - LIMIT: Number of questions to evaluate per dataset (default: 20)
```

#### What it does:

1. Runs evaluations on HotpotQA, MuSiQue, and 2WikiMultihopQA datasets
2. Saves results to separate files in the `result` directory
3. Shows progress and completion messages

#### Output:

The script will generate the following output files:
- `result/hotpotqa_evaluation.json`
- `result/musique_evaluation.json`
- `result/2wikimultihopqa_evaluation.json`

## Adding New Scripts

To add a new script:
1. Create your script in this directory
2. Make it executable with `chmod +x scripts/your_script.sh`
3. Add documentation in this README.md file 