import json
import logging
import time
from typing import List, Dict, Tuple, Any
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightAgenticRAG(BaseRAG):
    """
    LightAgenticRAG implements a memory-efficient agentic approach to retrieval-augmented generation.
    Instead of accumulating all retrieved contexts across iterations, it maintains a concise
    information summary that is continually refined with new retrieved information.
    """
    
    def __init__(self, corpus_path: str = None, cache_dir: str = "./cache"):
        """Initialize the LightAgenticRAG system."""
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3  # Default max rounds for iterative retrieval
    
    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of retrieval rounds."""
        self.max_rounds = max_rounds
    
    def generate_or_refine_summary(self, question: str, new_contexts: List[str], 
                                  current_summary: str = "") -> str:
        """
        Generate a new summary or refine an existing one based on newly retrieved contexts.
        
        Args:
            question: The original question
            new_contexts: Newly retrieved context chunks
            current_summary: Current information summary (if any)
            
        Returns:
            A concise summary of all relevant information so far
        """
        try:
            context_text = "\n".join(new_contexts)
            
            if not current_summary:
                # Generate initial summary
                prompt = f"""Please create a concise summary of the following information as it relates to answering this question:

Question: {question}

Information:
{context_text}

Your summary should:
1. Include all relevant facts that might help answer the question
2. Exclude irrelevant information
3. Be clear and concise
4. Preserve specific details, dates, numbers, and names that may be relevant

Summary:"""
            else:
                # Refine existing summary with new information
                prompt = f"""Please refine the following information summary using newly retrieved information.

Question: {question}

Current summary:
{current_summary}

New information:
{context_text}

Your refined summary should:
1. Integrate new relevant facts with the existing summary
2. Remove redundancies
3. Remain concise while preserving all important information
4. Prioritize information that helps answer the question
5. Maintain specific details, dates, numbers, and names that may be relevant

Refined summary:"""
            
            summary = get_response_with_retry(prompt)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating/refining summary: {e}")
            # If error occurs, concatenate current summary with new contexts as fallback
            if current_summary:
                return f"{current_summary}\n\nNew information:\n{context_text}"
            return context_text
    
    def analyze_completeness(self, question: str, info_summary: str) -> Dict:
        """
        Analyze if the current information summary is sufficient to answer the question.
        
        Args:
            question: The original question
            info_summary: Current information summary
            
        Returns:
            Dictionary with analysis results
        """
        try:
            prompt = f"""Question: {question}

Available Information:
{info_summary}

Based on the information provided, please analyze:
1. Can the question be answered completely with this information? (Yes/No)
2. What specific information is missing, if any?
3. What specific question should we ask to find the missing information?
4. Summarize our current understanding based on available information.

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "missing_info": string
- "subquery": string
- "current_understanding": string"""
            
            try:
                response = get_response_with_retry(prompt)
                
                # Clean up response to ensure it's valid JSON
                response = response.strip()
                
                # Remove any markdown code block markers
                response = response.replace('```json', '').replace('```', '')
                
                # Try to find JSON-like content within the response
                import re
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

    def generate_answer(self, question: str, info_summary: str) -> str:
        """Generate final answer based on the information summary."""
        try:
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Information Summary:
{info_summary}

Remember: Be concise - give ONLY the essential answer, nothing more.
Ans: """
            
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return ""

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:
        """
        Answer question with iterative retrieval and information summary refinement.
        
        Returns:
            Tuple of (answer, last_retrieved_contexts, round_count)
        """
        info_summary = ""  # Start with empty summary
        round_count = 0
        current_query = question
        retrieval_history = []
        last_contexts = []  # Store only the last retrieved contexts
        
        logger.info(f"LightAgenticRAG answering: {question}")
        
        while round_count < self.max_rounds:
            round_count += 1
            logger.info(f"Retrieval round {round_count}")
            
            # Retrieve relevant contexts for the current query
            new_contexts = self.retrieve(current_query)
            last_contexts = new_contexts  # Save current contexts
            
            # Record retrieval history
            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts
            })
            
            # Generate or refine information summary with new contexts
            info_summary = self.generate_or_refine_summary(
                question, 
                new_contexts, 
                info_summary
            )
            
            logger.info(f"Information summary after round {round_count} (length: {len(info_summary)})")
            
            # Analyze if we can answer the question with current summary
            analysis = self.analyze_completeness(question, info_summary)
            
            if analysis["can_answer"]:
                # Generate and return final answer
                answer = self.generate_answer(question, info_summary)
                # We return the last retrieved contexts for evaluation purposes
                return answer, last_contexts, round_count
            
            # Update query for next round
            current_query = analysis["subquery"]
            logger.info(f"Generated subquery: {current_query}")
        
        # If max rounds reached, generate best possible answer
        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer = self.generate_answer(question, info_summary)
        return answer, last_contexts, round_count 