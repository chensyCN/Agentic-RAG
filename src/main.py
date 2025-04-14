#!/usr/bin/env python
import argparse
import json
import logging
import importlib
from typing import Dict, List, Type

from src.evaluation.evaluation import RAGEvaluator
from src.models.base_rag import BaseRAG
from src.models.vanilla_rag import VanillaRAG
from src.models.agentic_rag import AgenticRAG
from src.models.light_agentic_rag import LightAgenticRAG

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dictionary of available RAG models
RAG_MODELS = {
    "vanilla": VanillaRAG,
    "agentic": AgenticRAG,
    "light": LightAgenticRAG
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run RAG models')
    
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
    parser.add_argument('--eval-top-ks', type=int, nargs='+', default=[5, 10],
                      help='List of k values for top-k accuracy evaluation (default: [5, 10])')
    
    # Single question (optional)
    parser.add_argument('--question', type=str,
                      help='Optional: Single question to answer')
    
    # RAG model selection
    parser.add_argument('--model', type=str, choices=list(RAG_MODELS.keys()), 
                      default='vanilla',
                      help='Which RAG model to use')
    
    # Evaluation options
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                      help='Output file name')
    
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

def create_rag_model(model_name: str, corpus_path: str, max_rounds: int = 3, top_k: int = 5) -> BaseRAG:
    """Create and configure a RAG model instance.
    
    Args:
        model_name: Name of the RAG model to create
        corpus_path: Path to the corpus file
        max_rounds: Maximum number of rounds for agentic models
        top_k: Number of contexts to retrieve
        
    Returns:
        A configured RAG model instance
    """
    # Check if model exists
    if model_name not in RAG_MODELS:
        raise ValueError(f"Unknown RAG model: {model_name}")
    
    # Create model instance
    model_class = RAG_MODELS[model_name]
    model = model_class(corpus_path)
    
    # Configure model
    model.set_top_k(top_k)
    
    # Set max rounds for agentic models
    if hasattr(model, 'set_max_rounds'):
        model.set_max_rounds(max_rounds)
        
    return model

def run_single_question(model_name: str, question: str, corpus_path: str, max_rounds: int, top_k: int):
    """Run a single question through the specified RAG model."""
    # Create and configure the model
    model = create_rag_model(model_name, corpus_path, max_rounds, top_k)
    
    # Get answer
    logger.info(f"\nQuestion: {question}")
    logger.info(f"Using {model_name} RAG model")
    
    # Handle different return signatures
    if model_name in ["agentic", "light"]:
        answer, contexts, rounds = model.answer_question(question)
        logger.info(f"\nAnswer: {answer}")
        logger.info(f"Retrieved in {rounds} rounds")
    else:
        answer, contexts = model.answer_question(question)
        logger.info(f"\nAnswer: {answer}")
    
    # Log contexts
    logger.info("\nContexts used:")
    for i, ctx in enumerate(contexts):
        logger.info(f"{i+1}. {ctx[:100]}...")
    
    return answer, contexts

def main():
    """Main function to run the RAG model."""
    args = parse_arguments()
    
    # If a question is provided, run in single question mode
    if args.question:
        run_single_question(
            model_name=args.model,
            question=args.question,
            corpus_path=args.corpus,
            max_rounds=args.max_rounds,
            top_k=args.top_k
        )
        return
    
    # Otherwise run in evaluation mode
    logger.info(f"Starting evaluation of {args.model} RAG model")
    logger.info(f"Max rounds: {args.max_rounds}, Top-k: {args.top_k}")
    logger.info(f"Evaluating top-k accuracy for k values: {args.eval_top_ks}")
    
    # Load evaluation data
    eval_data = load_evaluation_data(args.dataset, args.limit)
    if not eval_data:
        logger.error("No evaluation data available. Exiting.")
        return
    
    logger.info(f"Loaded {len(eval_data)} questions for evaluation")
    
    # Initialize evaluator for single model
    evaluator = RAGEvaluator(
        model_name=args.model,
        corpus_path=args.corpus,
        max_rounds=args.max_rounds,
        top_k=args.top_k,
        eval_top_ks=args.eval_top_ks
    )
    
    # Run evaluation
    evaluation_summary = evaluator.run_single_model_evaluation(
        eval_data=eval_data,
        output_file=args.output
    )
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main() 