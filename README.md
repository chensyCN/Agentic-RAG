# LogicRAG: Structured RAG Guided by Query Logic Dependency Graph

<div align="center">
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-green.svg"/></a>
      <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/last-commit/chensyCN/Agentic-RAG?color=blue"/></a>
      <a href="https://arxiv.org/abs/2508.06105"><img src="https://img.shields.io/badge/paper-available-brightgreen"/></a>
</div>

LogicRAG enables structured retrieval without building knowledge graphs on corpora. By constructing query logic dependency graphs to guide structured retrieval adaptively, it enables test-time scaling of graphRAG on large/dynamic knowledge bases. This work has been accepted to **AAAI'26**, with title [You Don't Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures](https://openreview.net/forum?id=ov1bwU35Mf). An updated version also available on [Arxiv](https://arxiv.org/abs/2508.06105).

![System Architecture](figs/framework.png)

## 🌟 Key Features

- **❶ Logic Dependency Analysis**: Convert complex questions into logical dependency graphs for planning multi-step retrieval.
- **❷ Graph Reasoning Linearization**: Linearize complex graph reasoning into sequential subproblem solution while maintaining logic-coherence.
- **❸ Efficiency**: Efficient scheduling via graph pruning, and context-length optimization via rolling memory.
- **❹ Interpretable Results**: Provides clear reasoning paths and dependency analysis for better explainability.

## 🚀 Quick Start

### Installation and Configuration

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Set your OpenAI API key:

```bash
# Create a .env file in the root directory with:
OPENAI_API_KEY=your_api_key_here
```

- Other configuration options can be modified in `config/config.py`

### Running Evaluation on a Dataset

```bash
python run.py --model logic-rag --dataset path/to/dataset.json --corpus path/to/corpus.json --max-rounds 5 --top-k 3
```

Options:

- `--max-rounds`: Maximum number of reasoning rounds (default: 3)
- `--top-k`: Number of top contexts to retrieve (default: 5)
- `--limit`: Number of questions to evaluate (default: 20)
  - Set to `0` to process all questions in the dataset

### Running a Single Question

```bash
python run.py --model logic-rag --question "Your question here" --corpus path/to/corpus.json --max-rounds 5 --top-k 3
```

### Example Usage

```python
from src.models.logic_rag import LogicRAG

# Initialize RAG system
rag = LogicRAG('path/to/corpus.json')
rag.set_max_rounds(5)
rag.set_top_k(3)

# Ask a question
answer, contexts, rounds = rag.answer_question("What is the capital of France?")
print(f"Answer: {answer}")
print(f"Retrieved in {rounds} rounds")
```

## 🍀 Citation

If you find this work helpful, please cite our paper:

```
@inproceedings{logicrag,
title={You Don't Need Pre-built Graphs for {RAG}: Retrieval Augmented Generation with Adaptive Reasoning Structures},
author={Shengyuan Chen and Chuang Zhou and Zheng Yuan and Qinggang Zhang and Zeyang Cui and Hao Chen and Yilin
Xiao and Jiannong Cao and Xiao Huang},
booktitle={The Fortieth AAAI Conference on Artificial Intelligence},
year={2026}
}
```
