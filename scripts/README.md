# Agentic-RAG Scripts

This directory contains scripts for running and evaluating the Agentic-RAG system on various datasets.

## Available Scripts

### run.sh

Runs evaluations on all available datasets in the `dataset` directory.

```bash
# Run with default parameters (standard AgenticRAG)
./scripts/run.sh

# Run with LightAgenticRAG
./scripts/run.sh --rag-type light

# Run both algorithms for comparison
./scripts/run.sh --rag-type both

# You can also modify other parameters:
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

The script will generate output files like:
- `result/hotpotqa_evaluation_standard.json` (for standard AgenticRAG)
- `result/hotpotqa_evaluation_light.json` (for LightAgenticRAG)

### run_light_agentic.sh

Runs evaluations using only the LightAgenticRAG algorithm on all available datasets.

```bash
# Run with default parameters
./scripts/run_light_agentic.sh
```

#### What it does:

1. Runs LightAgenticRAG evaluations on all datasets
2. Saves results to the `result/light_agentic` directory
3. Shows progress and completion messages

#### Output:

The script will generate the following output files:
- `result/light_agentic/hotpotqa_light_evaluation.json`
- `result/light_agentic/musique_light_evaluation.json`
- `result/light_agentic/2wikimultihopqa_light_evaluation.json`

### test_light_agentic_rag.py

Tests and compares LightAgenticRAG with standard AgenticRAG using a single question.

```bash
# Test only LightAgenticRAG
python scripts/test_light_agentic_rag.py --question "Your question here"

# Compare LightAgenticRAG with standard AgenticRAG
python scripts/test_light_agentic_rag.py --question "Your question here" --compare
```

Additional options:
- `--corpus`: Path to corpus file (default: dataset/hotpotqa_corpus.json)
- `--max-rounds`: Maximum number of rounds (default: 3)
- `--top-k`: Number of contexts to retrieve (default: 5)

#### Output:

Results are saved to `result/light_agentic_test_results.json`

## Adding New Scripts

To add a new script:
1. Create your script in this directory
2. Make it executable with `chmod +x scripts/your_script.sh`
3. Add documentation in this README.md file 