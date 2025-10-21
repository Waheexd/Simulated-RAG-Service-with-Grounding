# Retrieval-Augmented Generation (RAG) Service

## Overview

A lightweight Retrieval-Augmented Generation (RAG) system that **crawls websites, extracts text, indexes it with embeddings, and answers questions strictly based on crawled content**.
The system provides `/crawl`, `/index`, and `/ask` endpoints via a FastAPI API.

---

## Architecture

**Pipeline:** Crawl → Clean → Chunk → Embed → Store → Retrieve → Generate Answer

* **Crawl:** Collects in-domain pages while respecting `robots.txt` and crawl limits.
* **Clean:** Extracts readable text and removes HTML boilerplate.
* **Chunk:** Splits text into ~800-character segments with 200-character overlap.
* **Embed:** Generates text embeddings using `all-MiniLM-L6-v2`.
* **Index:** Stores embeddings in FAISS (or scikit-learn) for similarity search.
* **Ask:** Retrieves top-k relevant chunks and generates answers **only using that context**, with refusal for unanswerable questions.

---

## Setup

### 1. Create Environment

```bash
python -m venv venv
# Activate
source venv/bin/activate        # mac/linux
venv\Scripts\activate           # windows
```

### 2. Install Dependencies

Create `requirements.txt`:

```text
fastapi
uvicorn[standard]
requests
beautifulsoup4
tldextract
sentence-transformers
numpy
scikit-learn
faiss-cpu
transformers
torch
python-multipart
pydantic
pytest
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Run the API

```bash
uvicorn rag_service:app --reload --port 8000
```

Access the interactive documentation at:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints

### POST /crawl

Crawl a website within the same domain.

**Request JSON Example:**

```json
{
  "start_url": "https://en.wikipedia.org/wiki/OpenAI",
  "max_pages": 5,
  "max_depth": 2,
  "crawl_delay_ms": 200
}
```

**Response Example:**

```json
{
  "page_count": 5,
  "skipped_count": 0,
  "urls": [
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/Artificial_intelligence"
  ],
  "ms": 1500
}
```

---

### POST /index

Embeds and indexes crawled pages.

**Request JSON Example:**

```json
{ "chunk_size": 800, "chunk_overlap": 200 }
```

**Response Example:**

```json
{
  "vector_count": 125,
  "index_time_ms": 3500
}
```

---

### POST /ask

Retrieve relevant content and generate answers.

**Answerable Question Example:**

```json
{ "question": "Who founded OpenAI?", "top_k": 3, "refusal_threshold": 0.25 }
```

**Response Example:**

```json
{
  "answer": "OpenAI was founded by Elon Musk, Sam Altman, Greg Brockman, Ilya Sutskever, John Schulman, and Wojciech Zaremba.",
  "sources": [
    { "url": "https://en.wikipedia.org/wiki/OpenAI", "snippet": "OpenAI is an AI research laboratory consisting of..." }
  ],
  "timings": { "retrieval_ms": 12, "generation_ms": 8, "total_ms": 20 }
}
```

**Unanswerable Question Example:**

```json
{ "question": "Who invented a time machine in 1800?", "top_k": 3, "refusal_threshold": 0.25 }
```

**Response Example:**

```json
{
  "answer": "not found in crawled content",
  "sources": [
    { "url": "https://en.wikipedia.org/wiki/OpenAI", "snippet": "OpenAI is an AI research laboratory consisting of..." }
  ],
  "timings": { "retrieval_ms": 10, "generation_ms": 0, "total_ms": 10 }
}
```

---

## Evaluation

### Run Example Evaluations

Keep the API running and execute:

```bash
python eval_run.py
```

It runs **two evaluations**:

* `answerable.json` → verifies correct answers are grounded in content
* `unanswerable.json` → verifies refusal behavior when content is missing

**Sample `answerable.json`:**

```json
[
  { "question": "Who founded OpenAI?", "expected_answer": "Elon Musk, Sam Altman, Greg Brockman, Ilya Sutskever, John Schulman, Wojciech Zaremba" }
]
```

**Sample `unanswerable.json`:**

```json
[
  { "question": "Who invented a time machine in 1800?", "expected_answer": "not found in crawled content" }
]
```

### Run Sanity Tests

```bash
pytest -q
```

---

## Design & Trade-offs

* **Chunking:** 800-character chunks with 200-character overlap balance recall and precision.
* **Embeddings:** `all-MiniLM-L6-v2` chosen for speed and open-source quality.
* **Vector Index:** FAISS for fast similarity search, sklearn fallback for portability.
* **Safety:** Answers only from retrieved content; ignores page instructions.
* **Refusals:** Returns `"not found in crawled content"` for unanswerable queries.
* **Observability:** Logs retrieval, generation, and total latency.

---

## Folder Structure

```
sde_intern_rag/
├── rag_service.py
├── requirements.txt
├── README.md
├── eval_run.py
├── test_crawler.py
└── evals/
    ├── answerable.json
    └── unanswerable.json
```
## Key Features

* Retrieval-grounded and hallucination-free answers
* Fast vector search using FAISS
* Compact and efficient MiniLM embeddings
* Simple REST API with FastAPI
* Fully testable through evaluation and unit tests


