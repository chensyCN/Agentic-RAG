# Agentic-RAG

Agentic-RAG enhances traditional RAG with iterative, agent-based retrieval. While vanilla RAG performs a single retrieval pass, this approach adds reflection capabilities to identify information gaps and retrieve additional context when needed.

## Core Components

1. **Embedding System**: SentenceTransformer with batched encoding, memory management, and disk caching.

2. **Retrieval System**: Semantic search with cosine similarity, configurable top-k, and caching.

3. **Reflection Module**: Analyzes context completeness and generates targeted subqueries.

4. **Iterative Process**: Manages multiple retrieval rounds with context accumulation and deduplication.

5. **Answer Generation**: Concise responses based on accumulated context.

## Workflow

1. Initial retrieval based on query
2. Reflection analysis determines if information is sufficient
3. If incomplete, generate subquery and retrieve additional context
4. Repeat until complete or max rounds reached
5. Generate final answer from accumulated context

## Key Features

1. **Iterative Retrieval**: Multiple rounds of retrieval instead of a single pass
2. **Self-reflection**: Determines when more information is needed
3. **Targeted Subqueries**: Focused queries for missing information
4. **Context Accumulation**: Builds comprehensive context across rounds
5. **Dynamic Stopping**: Stops when sufficient information is available

## Usage

```python
rag = AgentRAG()
rag.load_corpus('path/to/corpus.json')
rag.top_k = 5        # Documents per round
rag.max_rounds = 3   # Maximum rounds

# Get answer
answer, contexts, rounds = rag.answer_question("Your question here")

# Compare with vanilla RAG
comparison = rag.compare_with_vanilla(
    question="Your question",
    gold_answer="Known correct answer",
    max_rounds=3,
    top_k=5
)
```

## Requirements

Python 3.7+, PyTorch, SentenceTransformer, OpenAI API, tqdm, numpy, backoff, ratelimit

## Performance Metrics

- **Retrieval**: Answer coverage and position accuracy
- **Quality**: LLM-based evaluation and string similarity metrics
- **Efficiency**: Response time and round counts vs. vanilla RAG 