# AI-Knowledge-System-RAG-
A user uploads documents and asks questions. The system answers using the uploaded documents instead of general LLM knowledge.

---

## Objective

Build a system where users upload a PDF and ask questions. The system answers using the uploaded document rather than relying only on the LLM's general knowledge.

---

## High Level Architecture

### Document Ingestion Pipeline

User Upload PDF

↓

PDF Text Extraction

↓

Chunking

↓

Embedding Generation

↓

Store Embeddings in Vector Database

---

### Question Answering Pipeline

User Question

↓

Question Embedding

↓

Similarity Search in Vector Database

↓

Relevant Chunks Retrieved

↓

Context + User Question sent to LLM

↓

LLM Generates Final Response

---

## Components

### 1. PDF Processor

Purpose:

Extract text from uploaded PDF documents.

Input:

PDF File

Output:

Raw Text

---

### 2. Chunking Module

Purpose:

Split large documents into smaller meaningful sections.

Input:

Raw Text

Output:

Text Chunks

Reason:

LLMs and embedding models work better on smaller sections.

---

### 3. Embedding Module

Purpose:

Convert text chunks into numerical vectors.

Input:

Text Chunk

Output:

Embedding Vector

Reason:

Allows semantic search based on meaning rather than exact words.

---

### 4. Vector Database

Purpose:

Store embeddings and their corresponding chunks.

Input:

Embedding Vectors

Output:

Relevant Chunks during retrieval

Examples:

ChromaDB

FAISS

Pinecone

(Choose one during implementation)

---

### 5. Retriever

Purpose:

Find document chunks most relevant to the user's question.

Input:

Question Embedding

Output:

Top Relevant Chunks

---

### 6. LLM Module

Purpose:

Generate answers using retrieved context.

Input:

Retrieved Chunks + User Question

Output:

Final Answer

---

## End-to-End Flow

PDF Upload

↓

Extract Text

↓

Create Chunks

↓

Generate Embeddings

↓

Store in Vector Database

↓

User asks Question

↓

Generate Question Embedding

↓

Retrieve Relevant Chunks

↓

Send Context + Question to LLM

↓

Generate Answer

↓

Display Response
