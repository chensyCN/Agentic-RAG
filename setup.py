from setuptools import setup, find_packages

setup(
    name="agentic-rag",
    version="0.1.0",
    description="Agentic Retrieval-Augmented Generation with iterative retrieval",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch",
        "sentence-transformers",
        "openai",
        "tqdm",
        "numpy",
        "backoff",
        "ratelimit",
    ],
    python_requires=">=3.7",
) 