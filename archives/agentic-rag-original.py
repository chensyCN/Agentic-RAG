import os
import gc
import pdb
import pickle
import logging
import numpy as np
import torch
from typing import List, Optional, Dict, Tuple, Set, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import backoff
import argparse
from ratelimit import limits, sleep_and_retry
import re
from config import (
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    CALLS_PER_MINUTE,
    PERIOD,
    MAX_RETRIES,
    RETRY_DELAY,
    CACHE_DIR,
    RESULT_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE
)
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REFLECTION_PROMPT = """Based on the question and the retrieved context, analyze:
1. Can you confidently answer the question with the given context and your knowledge?
2. If not, what specific information is missing?
3. Generate a focused search query to find the missing information.

Format your response as:
{
    "can_answer": true/false,
    "missing_info": "description of what information is missing",
    "subquery": "specific search query for missing information",
    "current_understanding": "brief summary of current understanding"
}
"""

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=MAX_RETRIES,
    max_time=300
)
def get_response_with_retry(prompt: str, temperature: float = 0.0) -> str:
    """Get response from OpenAI API with retry logic."""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in get_response_with_retry: {str(e)}")
        return ""

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace hyphen with space
    text = text.replace('-', ' ')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

class AgentRAG:
    def __init__(self, corpus_path: str = None, cache_dir: str = CACHE_DIR):
        """Initialize the AgentRAG system."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.corpus = {}
        self.corpus_embeddings = None
        self.embeddings = None  # For compatibility with vanilla_retrieve
        self.sentences = None   # For compatibility with vanilla_retrieve
        self.retrieval_cache = {}
        self.top_k = 5  # Default retrieval count
        self.max_rounds = 3  # Default max rounds
        
        if corpus_path:
            self.load_corpus(corpus_path)
    
    def load_corpus(self, corpus_path: str):
        """Load and process the document corpus."""
        logger.info("Loading corpus...")
        with open(corpus_path, 'r') as f:
            documents = json.load(f)
        
        # Process documents into chunks
        self.corpus = {
            i: f"title: {doc['title']} content: {doc['text']}"
            for i, doc in enumerate(documents)
        }
        
        # Store sentences for vanilla retrieval
        self.sentences = list(self.corpus.values())
        
        # Try to load cached embeddings
        cache_file = os.path.join(self.cache_dir, f'embeddings_{len(self.corpus)}.pt')
        
        if os.path.exists(cache_file):
            logger.info("Loading cached embeddings...")
            self.corpus_embeddings = torch.load(cache_file)
            self.embeddings = self.corpus_embeddings  # For compatibility with vanilla_retrieve
        else:
            logger.info("Computing embeddings...")
            texts = list(self.corpus.values())
            self.corpus_embeddings = self.encode_sentences_batch(texts)
            self.embeddings = self.corpus_embeddings  # For compatibility with vanilla_retrieve
            torch.save(self.corpus_embeddings, cache_file)

    def encode_batch(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Encode texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings)

    def encode_sentences_batch(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode sentences in batches with memory management."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
            batch = sentences[i:i + batch_size]
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch, 
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        
        final_embeddings = torch.cat(all_embeddings, dim=0)
        del all_embeddings
        gc.collect()
        
        return final_embeddings

    def build_index(self, sentences: List[str], batch_size: int = 32):
        """Build the embedding index for the sentences."""
        self.sentences = sentences
        
        # Try to load existing embeddings
        embedding_file = f'cache/embeddings_{len(sentences)}.pkl'
        if os.path.exists(embedding_file):
            try:
                with open(embedding_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Embeddings loaded from {embedding_file}")
                return
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")

        # Build new embeddings
        self.embeddings = self.encode_sentences_batch(sentences, batch_size)
        
        # Save embeddings
        try:
            os.makedirs('cache', exist_ok=True)
            with open(embedding_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def retrieve(self, query: str) -> List[str]:
        """Retrieve similar sentences using query embedding."""
        # Check cache first
        if query in self.retrieval_cache:
            return self.retrieval_cache[query]

        if self.corpus_embeddings is None or not self.corpus:
            return []

        try:
            # Encode query
            with torch.no_grad():
                query_embedding = self.model.encode([query], convert_to_tensor=True)[0]
                query_embedding = query_embedding.cpu()

            # Calculate similarities
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.corpus_embeddings
            )
            
            # Convert indices to list before using them
            top_k_scores, top_k_indices = similarities.topk(self.top_k)
            indices = top_k_indices.tolist()
            
            # Get results using integer indices
            results = [self.corpus[idx] for idx in indices]
            
            # Cache results
            self.retrieval_cache[query] = results
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return []

    def analyze_completeness(self, question: str, context: List[str]) -> Dict:
        """Analyze if the retrieved context is sufficient to answer the question."""
        try:
            context_text = "\n".join(context)
            prompt = f"""Question: {question}

Retrieved Context:
{context_text}

{REFLECTION_PROMPT}"""
            
            try:
                response = get_response_with_retry(prompt)
                
                # Clean up response to ensure it's valid JSON
                response = response.strip()
                
                # Remove any markdown code block markers
                response = response.replace('```json', '').replace('```', '')
                
                # Try to find JSON-like content within the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group()
                
                # Parse the cleaned response
                result = json.loads(response)
                
                # Validate required fields
                required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
                if not all(field in result for field in required_fields):
                    raise ValueError("Missing required fields")
                
                # Ensure boolean type for can_answer
                result["can_answer"] = bool(result["can_answer"])
                
                # Ensure non-empty subquery
                if not result["subquery"]:
                    result["subquery"] = question
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw response: {response}")
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response."
                }
                
        except Exception as e:
            logger.error(f"Error in analyze_completeness: {e}")
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}"
            }

    def generate_answer(self, question: str, context: List[str], 
                       current_understanding: str = "") -> str:
        """Generate final answer based on all retrieved context."""
        try:
            context_text = "\n".join(context)
            current_understanding_text = f"\nCurrent Understanding: {current_understanding}" if current_understanding else ""
            
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}{current_understanding_text}

Context:
{context_text}

Remember: Be as concise as vanilla RAG - give ONLY the essential answer, nothing more.
Ans: """
            
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return ""

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:
        """Answer question with iterative retrieval and reflection."""
        all_contexts = []
        round_count = 0
        current_query = question
        retrieval_history = []
        
        while round_count < self.max_rounds:
            round_count += 1
            logger.info(f"Retrieval round {round_count}")
            
            # Retrieve relevant contexts
            new_contexts = self.retrieve(current_query)
            all_contexts.extend(new_contexts)
            
            # Remove duplicates while preserving order
            seen = set()
            all_contexts = [x for x in all_contexts if not (x in seen or seen.add(x))]
            
            # Record retrieval history
            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts
            })
            
            # Analyze completeness
            analysis = self.analyze_completeness(question, all_contexts)
            
            if analysis["can_answer"]:
                # Generate and return final answer
                answer = self.generate_answer(
                    question, 
                    all_contexts,
                    analysis["current_understanding"]
                )
                return answer, all_contexts, round_count
            
            # Update query for next round
            current_query = analysis["subquery"]
            logger.info(f"Generated subquery: {current_query}")
        
        # If max rounds reached, generate best possible answer
        answer = self.generate_answer(
            question,
            all_contexts,
            "Note: Maximum retrieval rounds reached. Providing best possible answer."
        )
        return answer, all_contexts, round_count

    def vanilla_retrieve(self, query: str) -> List[str]:
        """Basic retrieval without agent iteration - uses same implementation as retrieve."""
        return self.retrieve(query)

    def compare_with_vanilla(self, question: str, gold_answer: str, max_rounds: int, top_k: int) -> Dict:
        """Compare agent-based and vanilla retrieval approaches."""
        self.max_rounds = max_rounds
        self.top_k = top_k
        
        # Vanilla RAG
        start_time = time.time()
        vanilla_contexts = self.vanilla_retrieve(question)
        vanilla_prompt = f"Only give the answer. Question: {question}\nHint: {' '.join(vanilla_contexts)}\nAns: "
        vanilla_answer = get_response_with_retry(vanilla_prompt)
        vanilla_time = time.time() - start_time
        
        # Agent RAG
        start_time = time.time()
        agent_answer, agent_contexts, rounds = self.answer_question(question)
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
        answer_in_top1 = 0
        answer_in_top3 = 0
        
        for contexts, answer in zip(retrieved_contexts, answers):
            normalized_answer = normalize_answer(answer)
            
            # Check if answer is in any context
            for i, context in enumerate(contexts):
                if normalized_answer in normalize_answer(context):
                    found_in_context += 1
                    if i == 0:  # Top-1
                        answer_in_top1 += 1
                    if i < 3:   # Top-3
                        answer_in_top3 += 1
                    break
        
        return {
            "answer_found_in_context": found_in_context / total,
            "answer_in_top1": answer_in_top1 / total,
            "answer_in_top3": answer_in_top3 / total,
            "total_questions": total
        }

def save_results(results: Dict, filename: str):
    """Save evaluation results to file."""
    os.makedirs('result', exist_ok=True)
    with open(f"result/{filename}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def evaluate_with_llm(generated: str, gold: str) -> bool:
    """Use LLM to evaluate if the generated answer correctly answers the question."""
    if not isinstance(generated, str) or not isinstance(gold, str):
        return False
        
    prompt = f"""You are an expert evaluator. Please evaluate if the generated answer is correct by comparing it with the gold answer.

Generated answer: {generated}
Gold answer: {gold}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""

    try:
        response = get_response_with_retry(prompt, temperature=0.0)
        return response.strip().lower() == "correct"
    except Exception as e:
        logger.error(f"Error in LLM evaluation: {e}")
        return False

def string_based_evaluation(generated: str, gold: str) -> dict:
    """Evaluate string similarity between generated and gold answers.
    
    Args:
        generated: Generated answer string
        gold: Gold/ground truth answer string
        
    Returns:
        Dictionary containing accuracy, precision, recall metrics
    """
    # Normalize answers
    normalized_prediction = normalize_answer(generated)
    normalized_ground_truth = normalize_answer(gold)
    
    # Calculate accuracy
    accuracy = 1 if normalized_ground_truth in normalized_prediction else 0
    
    # Calculate precision and recall
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    # Handle yes/no/noanswer cases
    if (normalized_prediction in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth) or \
       (normalized_ground_truth in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return {
            "accuracy": accuracy,
            "precision": 0,
            "recall": 0
        }
    
    # Calculate token overlap
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    # Calculate precision and recall
    precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

def main():
    parser = argparse.ArgumentParser(description='Run AgentRAG evaluation')
    parser.add_argument('--max-rounds', type=int, default=3, help='Maximum number of agent rounds')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top contexts to retrieve')
    args = parser.parse_args()

    # Load evaluation data
    with open('dataset/hotpotqa.json', 'r') as f:
        eval_data = json.load(f)
    
    # Initialize RAG system
    rag = AgentRAG()
    rag.load_corpus('dataset/hotpotqa_corpus.json')
    
    # Run evaluation on first 20 questions
    eval_data = eval_data[:20]
    results = []
    
    # Evaluation metrics
    total_questions = len(eval_data)
    vanilla_metrics = {
        "total_time": 0,
        "answer_coverage": 0,
        "top1_hits": 0,
        "top3_hits": 0,
        "top5_hits": 0,
        "answer_accuracy": 0,
        "string_accuracy": 0,
        "string_precision": 0,
        "string_recall": 0
    }
    
    agent_metrics = {
        "total_time": 0,
        "answer_coverage": 0,
        "top1_hits": 0,
        "top3_hits": 0,
        "top5_hits": 0,
        "total_rounds": 0,
        "answer_accuracy": 0,
        "string_accuracy": 0,
        "string_precision": 0,
        "string_recall": 0
    }
    
    for item in tqdm(eval_data, desc="Evaluating"):
        question = item['question']
        gold_answer = item['answer']
        
        # Compare agent-based and vanilla retrieval
        comparison_result = rag.compare_with_vanilla(
            question=question,
            gold_answer=gold_answer,
            max_rounds=args.max_rounds,
            top_k=args.top_k
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
                if i == 0:
                    vanilla_metrics["top1_hits"] += 1
                if i < 3:
                    vanilla_metrics["top3_hits"] += 1
                if i < 5:
                    vanilla_metrics["top5_hits"] += 1
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
                if i == 0:
                    agent_metrics["top1_hits"] += 1
                if i < 3:
                    agent_metrics["top3_hits"] += 1
                if i < 5:
                    agent_metrics["top5_hits"] += 1
                break
        
        # Evaluate answer using LLM
        if comparison_result["agent"]["is_correct"]:
            agent_metrics["answer_accuracy"] += 1
        if comparison_result["vanilla"]["is_correct"]:
            vanilla_metrics["answer_accuracy"] += 1
    
    # Calculate average metrics
    vanilla_metrics.update({
        "avg_time": vanilla_metrics["total_time"] / total_questions,
        "answer_coverage": vanilla_metrics["answer_coverage"] / total_questions * 100,
        "top1_accuracy": vanilla_metrics["top1_hits"] / total_questions * 100,
        "top3_accuracy": vanilla_metrics["top3_hits"] / total_questions * 100,
        "top5_accuracy": vanilla_metrics["top5_hits"] / total_questions * 100,
        "answer_accuracy": vanilla_metrics["answer_accuracy"] / total_questions * 100,
        "string_accuracy": vanilla_metrics["string_accuracy"] / total_questions * 100,
        "string_precision": vanilla_metrics["string_precision"] / total_questions * 100,
        "string_recall": vanilla_metrics["string_recall"] / total_questions * 100
    })
    
    agent_metrics.update({
        "avg_time": agent_metrics["total_time"] / total_questions,
        "avg_rounds": agent_metrics["total_rounds"] / total_questions,
        "answer_coverage": agent_metrics["answer_coverage"] / total_questions * 100,
        "top1_accuracy": agent_metrics["top1_hits"] / total_questions * 100,
        "top3_accuracy": agent_metrics["top3_hits"] / total_questions * 100,
        "top5_accuracy": agent_metrics["top5_hits"] / total_questions * 100,
        "answer_accuracy": agent_metrics["answer_accuracy"] / total_questions * 100,
        "string_accuracy": agent_metrics["string_accuracy"] / total_questions * 100,
        "string_precision": agent_metrics["string_precision"] / total_questions * 100,
        "string_recall": agent_metrics["string_recall"] / total_questions * 100
    })
    
    # Prepare evaluation summary
    evaluation_summary = {
        "total_questions": total_questions,
        "vanilla_retrieval": vanilla_metrics,
        "agent_retrieval": agent_metrics,
        "detailed_results": results
    }
    
    # Save detailed results and evaluation summary
    output_path = os.path.join(RESULT_DIR, 'agent_vs_vanilla_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # Log evaluation summary
    logger.info("\nEvaluation Summary:")
    logger.info("\nVanilla Retrieval Metrics:")
    logger.info(f"Average Time: {vanilla_metrics['avg_time']:.3f}s")
    logger.info(f"Answer Coverage: {vanilla_metrics['answer_coverage']:.2f}%")
    logger.info(f"Top-1 Accuracy: {vanilla_metrics['top1_accuracy']:.2f}%")
    logger.info(f"Top-3 Accuracy: {vanilla_metrics['top3_accuracy']:.2f}%")
    logger.info(f"Top-5 Accuracy: {vanilla_metrics['top5_accuracy']:.2f}%")
    logger.info(f"Answer Accuracy (LLM Evaluated): {vanilla_metrics['answer_accuracy']:.2f}%")
    logger.info(f"String-based Metrics:")
    logger.info(f"  - Accuracy: {vanilla_metrics['string_accuracy']:.2f}%")
    logger.info(f"  - Precision: {vanilla_metrics['string_precision']:.2f}%")
    logger.info(f"  - Recall: {vanilla_metrics['string_recall']:.2f}%")
    
    logger.info("\nAgent Retrieval Metrics:")
    logger.info(f"Average Time: {agent_metrics['avg_time']:.3f}s")
    logger.info(f"Average Rounds: {agent_metrics['avg_rounds']:.2f}")
    logger.info(f"Answer Coverage: {agent_metrics['answer_coverage']:.2f}%")
    logger.info(f"Top-1 Accuracy: {agent_metrics['top1_accuracy']:.2f}%")
    logger.info(f"Top-3 Accuracy: {agent_metrics['top3_accuracy']:.2f}%")
    logger.info(f"Top-5 Accuracy: {agent_metrics['top5_accuracy']:.2f}%")
    logger.info(f"Answer Accuracy (LLM Evaluated): {agent_metrics['answer_accuracy']:.2f}%")
    logger.info(f"String-based Metrics:")
    logger.info(f"  - Accuracy: {agent_metrics['string_accuracy']:.2f}%")
    logger.info(f"  - Precision: {agent_metrics['string_precision']:.2f}%")
    logger.info(f"  - Recall: {agent_metrics['string_recall']:.2f}%")
    
    logger.info(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main() 