# DocuRAG — Document Question Answering with RAG

A document-grounded question-answering system that lets users ask questions about uploaded PDFs and receive answers generated using retrieved document context.

**[🚀 Try the Live Demo](https://tinyurl.com/4b248k9n)** · **[View Source Code](https://github.com/Dhruvindave/AI-Knowledge-System-RAG/)**

---

## Overview

DocuRAG implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline, covering document ingestion, structural processing, semantic chunking, embedding generation, vector retrieval, and LLM-based answer generation.

The project also includes a retrieval evaluation pipeline to measure how effectively relevant document chunks are retrieved for a given question.

## Key Features

* **Document processing:** Convert PDFs into Markdown while preserving document structure.
* **Semantic chunking:** Split documents into meaningful chunks for retrieval.
* **Vector search:** Generate embeddings and retrieve relevant chunks using FAISS.
* **Context-grounded generation:** Use retrieved document context to generate answers with an LLM.
* **Retrieval evaluation:** Benchmark retrieval performance using Hit Rate and Mean Reciprocal Rank (MRR).
* **Interactive interface:** Upload documents and query their contents through a Streamlit application.

## Retrieval Evaluation

Retrieval quality was evaluated using a benchmark dataset of 100 questions.

| Metric   | Top 1 | Top 3 | Top 5 | Top 10 |
| -------- | ----: | ----: | ----: | -----: |
| Hit Rate |  0.64 |  0.90 |  0.95 |   0.97 |
| MRR      | 0.640 | 0.760 | 0.771 |  0.774 |

**Key observations**

* The correct relevant result appeared in the top 3 retrieved chunks for 90% of benchmark queries.
* Top-5 retrieval achieved a 95% hit rate.
* Increasing the retrieval depth to 10 raised the hit rate to 97%, with a smaller improvement in MRR.

These results measure retrieval performance, not the factual accuracy of generated answers. Answer quality requires separate evaluation.

## Architecture

### 1. Document Ingestion

PDF → Markdown conversion → Document structure processing → Semantic chunking → Embeddings → FAISS index

### 2. Question Answering

User question → Query embedding → Similarity search → Relevant chunks → Context construction → LLM response → Answer displayed in the UI

## Technical Stack

* **Language:** Python
* **LLM orchestration:** LangChain
* **Language model:** Google Gemini
* **Document processing:** PyMuPDF4LLM
* **Vector retrieval:** FAISS
* **Interface:** Streamlit
* **Deployment tooling:** Docker

## Run Locally

### Prerequisites

* Python
* Git
* A Google Gemini API key

### Setup

```bash
git clone https://github.com/Dhruvindave/AI-Knowledge-System-RAG.git
cd AI-Knowledge-System-RAG

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows, activate the environment using:

```powershell
.venv\Scripts\Activate.ps1
```

Configure the Gemini API key using the environment variable or configuration mechanism expected by the application. Keep credentials out of source control.

Launch the Streamlit interface using the repository's configured entry point. If using the provided Docker configuration, follow the deployment settings in `Dockerfile` and `docker-compose.yaml`.

## Project Structure

```text
AI-Knowledge-System-RAG/
├── src/                    # Core RAG implementation
├── experiments/             # Experiments and evaluation work
├── ui.py                    # Streamlit interface
├── main.ipynb               # Development notebook
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yaml      # Docker Compose configuration
└── README.md
```

## What This Project Demonstrates

* Building an end-to-end RAG pipeline beyond a basic LLM wrapper.
* Designing document-processing and semantic chunking workflows.
* Implementing vector-based retrieval with FAISS.
* Evaluating retrieval effectiveness using a benchmark dataset.
* Integrating LLM inference with a usable application interface.

## Limitations

* Retrieval metrics do not independently establish answer correctness or citation accuracy.
* Results depend on the document collection, chunking strategy, embeddings, and retrieval configuration.
* Generated answers may still be incomplete or incorrect when retrieved context is insufficient.

## Try It

**[Open DocuRAG](https://tinyurl.com/4b248k9n)** to explore document-based question answering.

**[Explore the code on GitHub](https://github.com/Dhruvindave/AI-Knowledge-System-RAG/)** to inspect the implementation.
