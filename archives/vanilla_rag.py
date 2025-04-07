import logging
from typing import List, Dict, Tuple
from base_rag import BaseRAG
from utils import get_response_with_retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VanillaRAG(BaseRAG):
    """
    VanillaRAG performs basic retrieval-augmented generation without iterative refinement.
    It inherits basic retrieval and embedding functionality from BaseRAG.
    """
    
    def __init__(self, corpus_path: str = None, cache_dir: str = None):
        """Initialize the VanillaRAG system."""
        super().__init__(corpus_path, cache_dir)
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve documents for the given query using vector similarity."""
        return super().retrieve(query)
    
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        """
        Answer a question using vanilla RAG approach:
        1. Retrieve relevant contexts
        2. Pass the question and contexts to the LLM to generate an answer
        """
        # Retrieve relevant contexts
        contexts = self.retrieve(question)
        
        # Generate answer using retrieved contexts
        answer = self.generate_answer(question, contexts)
        
        return answer, contexts
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate an answer based on the retrieved contexts."""
        try:
            context_text = "\n".join(contexts)
            
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Context:
{context_text}

Remember: Be concise - give ONLY the essential answer, nothing more.
Ans: """
            
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "" 