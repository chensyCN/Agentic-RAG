# Agentic RAG System

This repository contains a modularized implementation of an Agentic Retrieval-Augmented Generation (RAG) system. Offering
- [x] minimal viable prototype of an Agentic RAG pipeline;
- [x] modularized implementation that is easy to hack;
- [x] comprehensive evaluation framework for comparing traditional and agentic RAG approaches;

## File Structure

```
Agentic-RAG/
├── run.py                  # Script for running the application
├── setup.py                # Setup script for package installation
├── requirements.txt        # Package dependencies
├── config/
│   └── config.py           # Configuration settings
├── src/
│   ├── main.py             # Command-line interface and entry point
│   ├── models/
│   │   ├── base_rag.py     # Base RAG class with common functionality
│   │   ├── vanilla_rag.py  # Implementation of traditional RAG approach
│   │   └── agentic_rag.py  # Implementation of agentic RAG with iterative retrieval
│   ├── evaluation/
│   │   └── evaluation.py   # Evaluation utilities for comparing approaches
│   └── utils/
│       └── utils.py        # Shared utility functions
├── dataset/                # Dataset storage
├── cache/                  # Cache storage
└── result/                 # Result storage
```


## Installation

- Python 3.8+
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set your OpenAI API key, model choices, and other configuration options.


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

## Components

| Component | Features/Description |
|-----------|---------------------|
| **BaseRAG** | • Loading and processing document corpus<br>• Computing and caching document embeddings<br>• Basic retrieval functionality |
| **VanillaRAG** | • Single retrieval step for relevant contexts<br>• Direct answer generation from retrieved contexts |
| **AgenticRAG** | • Multiple retrieval rounds with iterative refinement<br>• Reflection on retrieved information to identify missing details<br>• Generation of focused sub-queries for additional retrieval<br>• Final answer generation from comprehensive context |
| **Evaluation** | • Answer accuracy<br>• Retrieval metrics<br>• Performance efficiency<br>• String-based evaluation metrics |

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