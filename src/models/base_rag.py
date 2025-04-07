import os
import gc
import json
import pickle
import torch
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from config.config import (
    CACHE_DIR,
    RESULT_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRAG:
    def __init__(self, corpus_path: str = None, cache_dir: str = CACHE_DIR):
        """Initialize the BaseRAG system."""
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
            
    def set_top_k(self, top_k: int):
        """Set the number of top contexts to retrieve."""
        self.top_k = top_k 