import logging
import time
from typing import Dict, List, Tuple, Any
import json
import os
from tqdm import tqdm

from src.utils.utils import (
    normalize_answer, 
    evaluate_with_llm, 
    string_based_evaluation,
    save_results
)
from src.models.vanilla_rag import VanillaRAG
from src.models.agentic_rag import AgenticRAG
from src.models.light_agentic_rag import LightAgenticRAG
from config.config import RESULT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary of available RAG models
RAG_MODELS = {
    "vanilla": VanillaRAG,
    "agentic": AgenticRAG,
    "light": LightAgenticRAG
}

class RAGEvaluator:
    """Evaluator for RAG models."""
    
    def __init__(self, model_name: str, corpus_path: str, max_rounds: int = 3, top_k: int = 5, 
                eval_top_ks: List[int] = [5, 10]):
        """Initialize the evaluator with corpus path and parameters.
        
        Args:
            model_name: Name of the RAG model to evaluate
            corpus_path: Path to the corpus file
            max_rounds: Maximum number of rounds for agentic RAG
            top_k: Number of contexts to retrieve
            eval_top_ks: List of k values for top-k accuracy evaluation
        """
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.eval_top_ks = sorted(eval_top_ks)  # Sort to ensure consistent processing
        
        # Create result directory if it doesn't exist
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        # Initialize the RAG model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified RAG model."""
        if self.model_name not in RAG_MODELS:
            raise ValueError(f"Unknown RAG model: {self.model_name}")
        
        # Create model instance
        model_class = RAG_MODELS[self.model_name]
        self.model = model_class(self.corpus_path)
        
        # Configure model
        self.model.set_top_k(self.top_k)
        
        # Set max rounds for agentic models
        if hasattr(self.model, 'set_max_rounds'):
            self.model.set_max_rounds(self.max_rounds)
        
        logger.info(f"Initialized {self.model_name} model")
    
    def evaluate_question(self, question: str, gold_answer: str) -> Dict:
        """Evaluate the model on a single question."""
        # Run the model on the question
        start_time = time.time()
        
        # Handle different return signatures
        if self.model_name in ["agentic", "light"]:
            answer, contexts, rounds = self.model.answer_question(question)
            elapsed_time = time.time() - start_time
            
            # Evaluate answer with LLM
            is_correct = evaluate_with_llm(answer, gold_answer)
            
            return {
                "question": question,
                "gold_answer": gold_answer,
                "answer": answer,
                "contexts": contexts,
                "time": elapsed_time,
                "rounds": rounds,
                "is_correct": is_correct
            }
        else:
            answer, contexts = self.model.answer_question(question)
            elapsed_time = time.time() - start_time
            
            # Evaluate answer with LLM
            is_correct = evaluate_with_llm(answer, gold_answer)
            
            return {
                "question": question,
                "gold_answer": gold_answer,
                "answer": answer,
                "contexts": contexts,
                "time": elapsed_time,
                "is_correct": is_correct
            }
        
    def calculate_retrieval_metrics(self, retrieved_contexts: List[List[str]], answers: List[str]) -> Dict[str, float]:
        """Calculate retrieval-based metrics."""
        total = len(answers)
        found_in_context = 0
        
        # Initialize answer_in_top_k counters for each k in eval_top_ks
        answer_in_top_k = {k: 0 for k in self.eval_top_ks}
        
        for contexts, answer in zip(retrieved_contexts, answers):
            normalized_answer = normalize_answer(answer)
            
            # Check if answer is in any context
            for i, context in enumerate(contexts):
                if normalized_answer in normalize_answer(context):
                    found_in_context += 1
                    # Update counters for each k value
                    for k in self.eval_top_ks:
                        if i < k:
                            answer_in_top_k[k] += 1
                    break
        
        # Prepare result dictionary
        result = {
            "answer_found_in_context": found_in_context / total,
            "total_questions": total
        }
        
        # Add top-k metrics to result
        for k in self.eval_top_ks:
            result[f"answer_in_top{k}"] = answer_in_top_k[k] / total
            
        return result
    
    def run_single_model_evaluation(self, eval_data: List[Dict], output_file: str = "evaluation_results.json"):
        """Run evaluation of a single model on the given evaluation data."""
        results = []
        
        # Evaluation metrics
        total_questions = len(eval_data)
        
        # Initialize metrics dictionary with dynamic top-k keys
        metrics = {
            "total_time": 0,
            "answer_coverage": 0,
            "answer_accuracy": 0,
            "string_accuracy": 0,
            "string_precision": 0,
            "string_recall": 0
        }
        
        # Add top-k hits for each k in eval_top_ks
        for k in self.eval_top_ks:
            metrics[f"top{k}_hits"] = 0
        
        # Add rounds tracking for agentic models
        if self.model_name in ["agentic", "light"]:
            metrics["total_rounds"] = 0
        
        for item in tqdm(eval_data, desc=f"Evaluating {self.model_name}"):
            question = item['question']
            gold_answer = item['answer']
            
            # Evaluate the model on this question
            result = self.evaluate_question(
                question=question,
                gold_answer=gold_answer
            )
            results.append(result)
            
            # Update metrics
            metrics["total_time"] += result["time"]
            normalized_gold = normalize_answer(gold_answer)
            
            # String-based evaluation
            string_metrics = string_based_evaluation(
                result["answer"], 
                gold_answer
            )
            metrics["string_accuracy"] += string_metrics["accuracy"]
            metrics["string_precision"] += string_metrics["precision"]
            metrics["string_recall"] += string_metrics["recall"]
            
            # Check retrieval coverage
            for i, ctx in enumerate(result["contexts"]):
                if normalized_gold in normalize_answer(ctx):
                    metrics["answer_coverage"] += 1
                    # Update counters for each k value
                    for k in self.eval_top_ks:
                        if i < k:
                            metrics[f"top{k}_hits"] += 1
                    break
            
            # Update rounds for agentic models
            if self.model_name in ["agentic", "light"] and "rounds" in result:
                metrics["total_rounds"] += result["rounds"]
            
            # Evaluate answer using LLM
            if result["is_correct"]:
                metrics["answer_accuracy"] += 1
        
        # Calculate average metrics
        avg_metrics = {
            "avg_time": metrics["total_time"] / total_questions,
            "answer_coverage": metrics["answer_coverage"] / total_questions * 100,
            "answer_accuracy": metrics["answer_accuracy"] / total_questions * 100,
            "string_accuracy": metrics["string_accuracy"] / total_questions * 100,
            "string_precision": metrics["string_precision"] / total_questions * 100,
            "string_recall": metrics["string_recall"] / total_questions * 100
        }
        
        # Add top-k coverage (renamed from accuracy) for each k in eval_top_ks
        for k in self.eval_top_ks:
            avg_metrics[f"top{k}_coverage"] = metrics[f"top{k}_hits"] / total_questions * 100
        
        # Add average rounds for agentic models
        if self.model_name in ["agentic", "light"]:
            avg_metrics["avg_rounds"] = metrics["total_rounds"] / total_questions
        
        # Organize metrics by category
        organized_metrics = {
            "performance": {
                "avg_time": avg_metrics["avg_time"]
            },
            "string_based": {
                "accuracy": avg_metrics["string_accuracy"],
                "precision": avg_metrics["string_precision"],
                "recall": avg_metrics["string_recall"]
            },
            "llm_evaluated": {
                "answer_accuracy": avg_metrics["answer_accuracy"]
            },
            "retrieval": {
                "answer_coverage": avg_metrics["answer_coverage"]
            }
        }
        
        # Add rounds for agentic models
        if self.model_name in ["agentic", "light"]:
            organized_metrics["performance"]["avg_rounds"] = avg_metrics["avg_rounds"]
        
        # Add top-k coverage metrics
        for k in self.eval_top_ks:
            organized_metrics["retrieval"][f"top{k}_coverage"] = avg_metrics[f"top{k}_coverage"]
        
        # Add raw metrics for backwards compatibility
        organized_metrics["raw"] = metrics
        
        # Prepare final evaluation summary
        evaluation_summary = {
            "model": self.model_name,
            "metrics": organized_metrics,
            "results": results
        }
        
        # Save results
        save_results(
            results=evaluation_summary,
            output_file=output_file,
            results_dir=RESULT_DIR
        )
        
        # Log results in three sections
        logger.info(f"\nEvaluation Summary for {self.model_name}:")
        
        # Performance metrics
        if self.model_name in ["agentic", "light"]:
            logger.info(f"Average time per question: {avg_metrics['avg_time']:.2f} seconds")
            logger.info(f"Average rounds per question: {avg_metrics['avg_rounds']:.2f}")
        else:
            logger.info(f"Average time per question: {avg_metrics['avg_time']:.2f} seconds")
        
        # 1. String-based metrics
        logger.info("\n1. String-based Metrics:")
        logger.info(f"  • Accuracy: {avg_metrics['string_accuracy']:.2f}%")
        logger.info(f"  • Precision: {avg_metrics['string_precision']:.2f}%")
        logger.info(f"  • Recall: {avg_metrics['string_recall']:.2f}%")
        
        # 2. LLM evaluated metrics
        logger.info("\n2. LLM Evaluated Metrics:")
        logger.info(f"  • Answer Accuracy: {avg_metrics['answer_accuracy']:.2f}%")
        
        # 3. Retrieval performance
        logger.info("\n3. Retrieval Performance:")
        logger.info(f"  • Answer Coverage: {avg_metrics['answer_coverage']:.2f}%")
        
        # Log top-k coverage metrics
        for k in self.eval_top_ks:
            logger.info(f"  • Top-{k} Coverage: {avg_metrics[f'top{k}_coverage']:.2f}%")
        
        return evaluation_summary 