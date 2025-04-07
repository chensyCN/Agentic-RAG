#!/usr/bin/env python
import argparse
import json
import logging
from typing import Dict, List
from src.evaluation.evaluation import RAGEvaluator
from src.models.agentic_rag import AgenticRAG
from src.models.vanilla_rag import VanillaRAG

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run AgenticRAG evaluation')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='dataset/hotpotqa.json',
                      help='Path to the dataset file')
    parser.add_argument('--corpus', type=str, default='dataset/hotpotqa_corpus.json',
                      help='Path to the corpus file')
    parser.add_argument('--limit', type=int, default=20,
                      help='Number of questions to evaluate (default: 20)')
    
    # RAG configuration
    parser.add_argument('--max-rounds', type=int, default=3, 
                      help='Maximum number of agent rounds')
    parser.add_argument('--top-k', type=int, default=5, 
                      help='Number of top contexts to retrieve')
    
    # Evaluation options
    parser.add_argument('--output', type=str, default='agent_vs_vanilla_comparison.json',
                      help='Output file name')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'agentic', 'vanilla'], 
                      default='evaluate',
                      help='Mode of operation: evaluate compares both approaches, other options run a single approach')
    parser.add_argument('--question', type=str,
                      help='Single question to answer (used with --mode=agentic or --mode=vanilla)')
    
    return parser.parse_args()

def load_evaluation_data(dataset_path: str, limit: int) -> List[Dict]:
    """Load and limit the evaluation dataset."""
    try:
        with open(dataset_path, 'r') as f:
            eval_data = json.load(f)
        
        # Limit the number of questions if needed
        if limit and limit > 0:
            eval_data = eval_data[:limit]
            
        return eval_data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return []

def run_single_question(question: str, mode: str, corpus_path: str, max_rounds: int, top_k: int):
    """Run a single question through either AgenticRAG or VanillaRAG."""
    if mode == "agentic":
        rag = AgenticRAG(corpus_path)
        rag.set_max_rounds(max_rounds)
        rag.set_top_k(top_k)
        answer, contexts, rounds = rag.answer_question(question)
        
        logger.info(f"\nQuestion: {question}")
        logger.info(f"\nAgenticRAG Answer: {answer}")
        logger.info(f"Retrieved in {rounds} rounds")
        logger.info("\nContexts used:")
        for i, ctx in enumerate(contexts):
            logger.info(f"{i+1}. {ctx[:100]}...")
            
    elif mode == "vanilla":
        rag = VanillaRAG(corpus_path)
        rag.set_top_k(top_k)
        answer, contexts = rag.answer_question(question)
        
        logger.info(f"\nQuestion: {question}")
        logger.info(f"\nVanillaRAG Answer: {answer}")
        logger.info("\nContexts used:")
        for i, ctx in enumerate(contexts):
            logger.info(f"{i+1}. {ctx[:100]}...")

def main():
    """Main function to run the evaluation."""
    args = parse_arguments()
    
    # For single question mode
    if args.mode in ["agentic", "vanilla"] and args.question:
        run_single_question(
            question=args.question,
            mode=args.mode,
            corpus_path=args.corpus,
            max_rounds=args.max_rounds,
            top_k=args.top_k
        )
        return
    
    # For evaluation mode
    logger.info("Starting AgenticRAG vs VanillaRAG evaluation")
    logger.info(f"Max rounds: {args.max_rounds}, Top-k: {args.top_k}")
    
    # Load evaluation data
    eval_data = load_evaluation_data(args.dataset, args.limit)
    if not eval_data:
        logger.error("No evaluation data available. Exiting.")
        return
    
    logger.info(f"Loaded {len(eval_data)} questions for evaluation")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        corpus_path=args.corpus,
        max_rounds=args.max_rounds,
        top_k=args.top_k
    )
    
    # Run evaluation
    evaluation_summary = evaluator.run_evaluation(
        eval_data=eval_data,
        output_file=args.output
    )
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main() 