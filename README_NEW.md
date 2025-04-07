# Agentic RAG System

This repository contains an implementation of an Agentic Retrieval-Augmented Generation (RAG) system that uses iterative refinement to improve answer quality compared to traditional RAG approaches.

## File Structure

The codebase has been decoupled into the following modules:

- `base_rag.py` - Base RAG class with common functionality like corpus loading and embedding
- `vanilla_rag.py` - Implementation of traditional RAG approach
- `agentic_rag.py` - Implementation of the agentic RAG approach with iterative retrieval
- `evaluation.py` - Evaluation utilities for comparing both approaches
- `utils.py` - Shared utility functions
- `main.py` - Command-line interface and entry point
- `config.py` - Configuration settings

## Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Evaluation

To compare the AgenticRAG and VanillaRAG approaches on a dataset:

```bash
python main.py --dataset path/to/dataset.json --corpus path/to/corpus.json
```

Options:
- `--max-rounds`: Maximum number of agent retrieval rounds (default: 3)
- `--top-k`: Number of top contexts to retrieve (default: 5)
- `--limit`: Number of questions to evaluate (default: 20)
- `--output`: Output file name for results (default: agent_vs_vanilla_comparison.json)

### Running a Single Query

To run a single question through the AgenticRAG system:

```bash
python main.py --mode agentic --question "Your question here" --corpus path/to/corpus.json
```

To run a single question through the VanillaRAG system:

```bash
python main.py --mode vanilla --question "Your question here" --corpus path/to/corpus.json
```

## Configuration

Edit `config.py` to set your OpenAI API key, model choices, and other configuration options.

## Components

### BaseRAG

The base RAG class provides shared functionality:
- Loading and processing document corpus
- Computing and caching document embeddings
- Basic retrieval functionality

### VanillaRAG

Traditional RAG approach:
- Single retrieval step for relevant contexts
- Direct answer generation from retrieved contexts

### AgenticRAG

Enhanced RAG with agentic capabilities:
- Multiple retrieval rounds with iterative refinement
- Reflection on retrieved information to identify missing details
- Generation of focused sub-queries for additional retrieval
- Final answer generation from comprehensive context

### Evaluation

The evaluation module compares both approaches on:
- Answer accuracy
- Retrieval metrics
- Performance efficiency
- String-based evaluation metrics

## Example

```python
from agentic_rag import AgenticRAG

# Initialize RAG system
rag = AgenticRAG('path/to/corpus.json')
rag.set_max_rounds(3)
rag.set_top_k(5)

# Ask a question
answer, contexts, rounds = rag.answer_question("What is the capital of France?")
print(f"Answer: {answer}")
print(f"Retrieved in {rounds} rounds")
``` 