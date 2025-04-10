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
from config.config import RESULT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluator for comparing AgenticRAG and VanillaRAG approaches."""
    
    def __init__(self, corpus_path: str, max_rounds: int = 3, top_k: int = 5, eval_top_ks: List[int] = [5, 10]):
        """Initialize the evaluator with corpus path and parameters.
        
        Args:
            corpus_path: Path to the corpus file
            max_rounds: Maximum number of rounds for agentic RAG
            top_k: Number of contexts to retrieve
            eval_top_ks: List of k values for top-k accuracy evaluation
        """
        self.corpus_path = corpus_path
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.eval_top_ks = sorted(eval_top_ks)  # Sort to ensure consistent processing
        
        # Create result directory if it doesn't exist
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        # Initialize both RAG systems
        self.agentic_rag = AgenticRAG(corpus_path)
        self.agentic_rag.set_max_rounds(max_rounds)
        self.agentic_rag.set_top_k(top_k)
        
        self.vanilla_rag = VanillaRAG(corpus_path)
        self.vanilla_rag.set_top_k(top_k)
        
    def compare_approaches(self, question: str, gold_answer: str) -> Dict:
        """Compare agent-based and vanilla retrieval approaches for a single question."""
        # Vanilla RAG
        start_time = time.time()
        vanilla_answer, vanilla_contexts = self.vanilla_rag.answer_question(question)
        vanilla_time = time.time() - start_time
        
        # Agent RAG
        start_time = time.time()
        agent_answer, agent_contexts, rounds = self.agentic_rag.answer_question(question)
        agent_time = time.time() - start_time
        
        # Evaluate vanilla answer with LLM
        vanilla_is_correct = evaluate_with_llm(vanilla_answer, gold_answer)
        
        # Evaluate agent answer with LLM
        agent_is_correct = evaluate_with_llm(agent_answer, gold_answer)
        
        return {
            "question": question,
            "gold_answer": gold_answer,
            "vanilla": {
                "answer": vanilla_answer,
                "contexts": vanilla_contexts,
                "time": vanilla_time,
                "is_correct": vanilla_is_correct
            },
            "agent": {
                "answer": agent_answer,
                "contexts": agent_contexts,
                "time": agent_time,
                "rounds": rounds,
                "is_correct": agent_is_correct
            }
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
        
    def run_evaluation(self, eval_data: List[Dict], output_file: str = "agent_vs_vanilla_comparison.json"):
        """Run evaluation on the given evaluation data."""
        results = []
        
        # Evaluation metrics
        total_questions = len(eval_data)
        
        # Initialize metrics dictionaries with dynamic top-k keys
        vanilla_metrics = {
            "total_time": 0,
            "answer_coverage": 0,
            "answer_accuracy": 0,
            "string_accuracy": 0,
            "string_precision": 0,
            "string_recall": 0
        }
        
        # Add top-k hits for each k in eval_top_ks
        for k in self.eval_top_ks:
            vanilla_metrics[f"top{k}_hits"] = 0
        
        agent_metrics = {
            "total_time": 0,
            "answer_coverage": 0,
            "total_rounds": 0,
            "answer_accuracy": 0,
            "string_accuracy": 0,
            "string_precision": 0,
            "string_recall": 0
        }
        
        # Add top-k hits for each k in eval_top_ks
        for k in self.eval_top_ks:
            agent_metrics[f"top{k}_hits"] = 0
        
        for item in tqdm(eval_data, desc="Evaluating"):
            question = item['question']
            gold_answer = item['answer']
            
            # Compare agent-based and vanilla retrieval
            comparison_result = self.compare_approaches(
                question=question,
                gold_answer=gold_answer
            )
            results.append(comparison_result)
            
            # Update vanilla metrics
            vanilla_metrics["total_time"] += comparison_result["vanilla"]["time"]
            normalized_gold = normalize_answer(gold_answer)
            
            # String-based evaluation for vanilla
            vanilla_string_metrics = string_based_evaluation(
                comparison_result["vanilla"]["answer"], 
                gold_answer
            )
            vanilla_metrics["string_accuracy"] += vanilla_string_metrics["accuracy"]
            vanilla_metrics["string_precision"] += vanilla_string_metrics["precision"]
            vanilla_metrics["string_recall"] += vanilla_string_metrics["recall"]
            
            # Check vanilla retrieval coverage
            for i, ctx in enumerate(comparison_result["vanilla"]["contexts"]):
                if normalized_gold in normalize_answer(ctx):
                    vanilla_metrics["answer_coverage"] += 1
                    # Update counters for each k value
                    for k in self.eval_top_ks:
                        if i < k:
                            vanilla_metrics[f"top{k}_hits"] += 1
                    break
            
            # Update agent metrics
            agent_metrics["total_time"] += comparison_result["agent"]["time"]
            agent_metrics["total_rounds"] += comparison_result["agent"]["rounds"]
            
            # String-based evaluation for agent
            agent_string_metrics = string_based_evaluation(
                comparison_result["agent"]["answer"], 
                gold_answer
            )
            agent_metrics["string_accuracy"] += agent_string_metrics["accuracy"]
            agent_metrics["string_precision"] += agent_string_metrics["precision"]
            agent_metrics["string_recall"] += agent_string_metrics["recall"]
            
            # Check agent retrieval coverage
            for i, ctx in enumerate(comparison_result["agent"]["contexts"]):
                if normalized_gold in normalize_answer(ctx):
                    agent_metrics["answer_coverage"] += 1
                    # Update counters for each k value
                    for k in self.eval_top_ks:
                        if i < k:
                            agent_metrics[f"top{k}_hits"] += 1
                    break
            
            # Evaluate answer using LLM
            if comparison_result["agent"]["is_correct"]:
                agent_metrics["answer_accuracy"] += 1
            if comparison_result["vanilla"]["is_correct"]:
                vanilla_metrics["answer_accuracy"] += 1
        
        # Calculate average metrics
        avg_vanilla_metrics = {
            "avg_time": vanilla_metrics["total_time"] / total_questions,
            "answer_coverage": vanilla_metrics["answer_coverage"] / total_questions * 100,
            "answer_accuracy": vanilla_metrics["answer_accuracy"] / total_questions * 100,
            "string_accuracy": vanilla_metrics["string_accuracy"] / total_questions * 100,
            "string_precision": vanilla_metrics["string_precision"] / total_questions * 100,
            "string_recall": vanilla_metrics["string_recall"] / total_questions * 100
        }
        
        # Add top-k accuracy for each k in eval_top_ks
        for k in self.eval_top_ks:
            avg_vanilla_metrics[f"top{k}_accuracy"] = vanilla_metrics[f"top{k}_hits"] / total_questions * 100
        
        # Update vanilla metrics with averages
        vanilla_metrics.update(avg_vanilla_metrics)
        
        avg_agent_metrics = {
            "avg_time": agent_metrics["total_time"] / total_questions,
            "avg_rounds": agent_metrics["total_rounds"] / total_questions,
            "answer_coverage": agent_metrics["answer_coverage"] / total_questions * 100,
            "answer_accuracy": agent_metrics["answer_accuracy"] / total_questions * 100,
            "string_accuracy": agent_metrics["string_accuracy"] / total_questions * 100,
            "string_precision": agent_metrics["string_precision"] / total_questions * 100,
            "string_recall": agent_metrics["string_recall"] / total_questions * 100
        }
        
        # Add top-k accuracy for each k in eval_top_ks
        for k in self.eval_top_ks:
            avg_agent_metrics[f"top{k}_accuracy"] = agent_metrics[f"top{k}_hits"] / total_questions * 100
        
        # Update agent metrics with averages
        agent_metrics.update(avg_agent_metrics)
        
        # Prepare evaluation summary
        evaluation_summary = {
            "total_questions": total_questions,
            "vanilla_retrieval": vanilla_metrics,
            "agent_retrieval": agent_metrics,
            "detailed_results": results
        }
        
        # Save detailed results and evaluation summary
        output_path = os.path.join(RESULT_DIR, output_file)
        with open(output_path, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        # Log evaluation summary
        logger.info("\nEvaluation Summary:")
        logger.info("\nVanilla Retrieval Metrics:")
        logger.info(f"Average Time: {vanilla_metrics['avg_time']:.3f}s")
        logger.info(f"Answer Coverage: {vanilla_metrics['answer_coverage']:.2f}%")
        
        # Log top-k accuracy for each k in eval_top_ks
        for k in self.eval_top_ks:
            logger.info(f"Top-{k} Accuracy: {vanilla_metrics[f'top{k}_accuracy']:.2f}%")
            
        logger.info(f"Answer Accuracy (LLM Evaluated): {vanilla_metrics['answer_accuracy']:.2f}%")
        logger.info(f"String-based Metrics:")
        logger.info(f"  - Accuracy: {vanilla_metrics['string_accuracy']:.2f}%")
        logger.info(f"  - Precision: {vanilla_metrics['string_precision']:.2f}%")
        logger.info(f"  - Recall: {vanilla_metrics['string_recall']:.2f}%")
        
        logger.info("\nAgent Retrieval Metrics:")
        logger.info(f"Average Time: {agent_metrics['avg_time']:.3f}s")
        logger.info(f"Average Rounds: {agent_metrics['avg_rounds']:.2f}")
        logger.info(f"Answer Coverage: {agent_metrics['answer_coverage']:.2f}%")
        
        # Log top-k accuracy for each k in eval_top_ks
        for k in self.eval_top_ks:
            logger.info(f"Top-{k} Accuracy: {agent_metrics[f'top{k}_accuracy']:.2f}%")
            
        logger.info(f"Answer Accuracy (LLM Evaluated): {agent_metrics['answer_accuracy']:.2f}%")
        logger.info(f"String-based Metrics:")
        logger.info(f"  - Accuracy: {agent_metrics['string_accuracy']:.2f}%")
        logger.info(f"  - Precision: {agent_metrics['string_precision']:.2f}%")
        logger.info(f"  - Recall: {agent_metrics['string_recall']:.2f}%")
        
        logger.info(f"\nDetailed results saved to {output_path}")
        
        return evaluation_summary 