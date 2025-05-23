"""
Configuration file for API keys and other settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# API Rate Limiting Configuration
CALLS_PER_MINUTE = 20
PERIOD = 60
MAX_RETRIES = 3
RETRY_DELAY = 120

# Model Configuration
DEFAULT_MODEL = "gpt-4o-mini"  # please specify your preferred LLM model
DEFAULT_MAX_TOKENS = 150

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # please specify your preferred embedding model
EMBEDDING_BATCH_SIZE = 32

# Cache Configuration
CACHE_DIR = "cache"
RESULT_DIR = "result" 