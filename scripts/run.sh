#!/bin/bash

# Agentic-RAG evaluation script
# This script runs evaluations on all available datasets with configurable RAG model

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration parameters
MAX_ROUNDS=5
TOP_K=3
EVAL_TOP_KS="5 10"
LIMIT=50 # Number of questions to evaluate per dataset
MODEL="light" # Default model: vanilla, agentic, light

# Output directory for results
RESULTS_DIR="result"
mkdir -p $RESULTS_DIR

# Show usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help               Show this help message"
    echo "  -m, --max-rounds NUM     Set maximum rounds (default: $MAX_ROUNDS)"
    echo "  -k, --top-k NUM          Set top-k contexts (default: $TOP_K)"
    echo "  -l, --limit NUM          Set number of questions per dataset (default: $LIMIT)"
    echo "  -r, --model MODEL        Set RAG model: vanilla, agentic, light (default: $MODEL)"
    echo ""
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -m|--max-rounds)
            MAX_ROUNDS="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -r|--model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate RAG model
if [[ "$MODEL" != "vanilla" && "$MODEL" != "agentic" && "$MODEL" != "light" ]]; then
    echo "Invalid RAG model: $MODEL. Must be 'vanilla', 'agentic', or 'light'."
    usage
fi

echo -e "${GREEN}Starting Agentic-RAG evaluations...${NC}"
echo -e "Model: $MODEL, Max rounds: $MAX_ROUNDS, Top-K: $TOP_K, Eval Top-Ks: $EVAL_TOP_KS, Questions per dataset: $LIMIT"
echo ""

# Function to run evaluation on a dataset
run_evaluation() {
    local dataset=$1
    local corpus=$2
    local output=$3
    local dataset_name=$(basename "$dataset" .json)
    
    echo -e "${BLUE}Evaluating ${dataset_name} with ${MODEL} model${NC}"
    echo "Dataset: $dataset"
    echo "Corpus: $corpus"
    echo "Output: $output"
    
    python run.py \
        --dataset "$dataset" \
        --corpus "$corpus" \
        --model "$MODEL" \
        --max-rounds $MAX_ROUNDS \
        --top-k $TOP_K \
        --eval-top-ks $EVAL_TOP_KS \
        --limit $LIMIT \
        --output "$output"
    
    echo -e "${GREEN}Evaluation complete for $dataset_name${NC}"
    echo ""
}

# HotpotQA dataset
echo -e "${YELLOW}=== HotpotQA Dataset ===${NC}"
run_evaluation \
    "dataset/hotpotqa.json" \
    "dataset/hotpotqa_corpus.json" \
    "${MODEL}_hotpotqa_evaluation.json"

# MuSiQue dataset
echo -e "${YELLOW}=== MuSiQue Dataset ===${NC}"
run_evaluation \
    "dataset/musique.json" \
    "dataset/musique_corpus.json" \
    "${MODEL}_musique_evaluation.json"

# 2WikiMultihopQA dataset
echo -e "${YELLOW}=== 2WikiMultihopQA Dataset ===${NC}"
run_evaluation \
    "dataset/2wikimultihopqa.json" \
    "dataset/2wikimultihopqa_corpus.json" \
    "${MODEL}_2wikimultihopqa_evaluation.json"

echo -e "${GREEN}All evaluations complete!${NC}"
echo -e "Results saved with prefix '${MODEL}_' in the $RESULTS_DIR directory."