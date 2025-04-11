#!/bin/bash

# Agentic-RAG evaluation script
# This script runs evaluations on all available datasets

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration parameters
MAX_ROUNDS=3
TOP_K=5
EVAL_TOP_KS="5 10 20"
LIMIT=20 # Number of questions to evaluate per dataset

# Output directory for results
RESULTS_DIR="result"
mkdir -p $RESULTS_DIR

echo -e "${GREEN}Starting Agentic-RAG evaluations...${NC}"
echo -e "Max rounds: $MAX_ROUNDS, Top-K: $TOP_K, Eval Top-Ks: $EVAL_TOP_KS, Questions per dataset: $LIMIT"
echo ""

# Function to run evaluation on a dataset
run_evaluation() {
    local dataset=$1
    local corpus=$2
    local output=$3
    
    echo -e "${YELLOW}Evaluating dataset: $(basename $dataset)${NC}"
    echo "Dataset: $dataset"
    echo "Corpus: $corpus"
    echo "Output: $output"
    
    python run.py \
        --dataset "$dataset" \
        --corpus "$corpus" \
        --max-rounds $MAX_ROUNDS \
        --top-k $TOP_K \
        --eval-top-ks $EVAL_TOP_KS \
        --limit $LIMIT \
        --output "$output"
    
    echo -e "${GREEN}Evaluation complete for $(basename $dataset)${NC}"
    echo ""
}

# HotpotQA dataset
echo -e "${YELLOW}=== HotpotQA Dataset ===${NC}"
run_evaluation \
    "dataset/hotpotqa.json" \
    "dataset/hotpotqa_corpus.json" \
    "$RESULTS_DIR/hotpotqa_evaluation.json"

# MuSiQue dataset
echo -e "${YELLOW}=== MuSiQue Dataset ===${NC}"
run_evaluation \
    "dataset/musique.json" \
    "dataset/musique_corpus.json" \
    "$RESULTS_DIR/musique_evaluation.json"

# 2WikiMultihopQA dataset
echo -e "${YELLOW}=== 2WikiMultihopQA Dataset ===${NC}"
run_evaluation \
    "dataset/2wikimultihopqa.json" \
    "dataset/2wikimultihopqa_corpus.json" \
    "$RESULTS_DIR/2wikimultihopqa_evaluation.json"

echo -e "${GREEN}All evaluations complete!${NC}"
echo "Results can be found in the $RESULTS_DIR directory." 