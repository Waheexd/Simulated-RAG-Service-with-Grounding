# Simulated-RAG-Service-with-Grounding
🧠 Lightweight Retrieval-Augmented Generation (RAG) System

A minimal, fast, and reliable Retrieval-Augmented Generation (RAG) system built with FastAPI that crawls websites, extracts readable content, indexes it with embeddings, and answers questions strictly based on crawled data — ensuring hallucination-free responses.

🚀 Overview

Pipeline:
Crawl → Clean → Chunk → Embed → Store → Retrieve → Generate Answer

🧩 Components

Crawl: Collects in-domain pages (respects robots.txt and crawl limits).

Clean: Extracts readable text and removes HTML boilerplate.

Chunk: Splits text into ~800-character segments with 200-character overlap.

Embed: Uses all-MiniLM-L6-v2 from sentence-transformers for embeddings.

Index: Stores embeddings using FAISS (with scikit-learn fallback).

Ask: Retrieves top-k similar chunks and generates answers based only on retrieved context.

⚙️ Setup
1. Create Virtual Environment
python -m venv venv
# Activate
source venv/bin/activate        # mac/linux
venv\Scripts\activate           # windows

2. Install Dependencies

Create a requirements.txt file with:

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


Install them:

pip install -r requirements.txt

▶️ Run the API
uvicorn rag_service:app --reload --port 8000


Access Swagger docs: http://127.0.0.1:8000/docs

🧭 API Endpoints
POST /crawl

Crawl a website (within same domain).

Request Example:

{
  "start_url": "https://en.wikipedia.org/wiki/OpenAI",
  "max_pages": 5,
  "max_depth": 2,
  "crawl_delay_ms": 200
}


Response Example:

{
  "page_count": 5,
  "skipped_count": 0,
  "urls": [
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/Artificial_intelligence"
  ],
  "ms": 1500
}

POST /index

Embed and index crawled pages.

Request Example:

{ "chunk_size": 800, "chunk_overlap": 200 }


Response Example:

{
  "vector_count": 125,
  "index_time_ms": 3500
}

POST /ask

Retrieve relevant content and generate grounded answers.

Answerable Example:

{ "question": "Who founded OpenAI?", "top_k": 3, "refusal_threshold": 0.25 }


Response Example:

{
  "answer": "OpenAI was founded by Elon Musk, Sam Altman, Greg Brockman, Ilya Sutskever, John Schulman, and Wojciech Zaremba.",
  "sources": [
    { "url": "https://en.wikipedia.org/wiki/OpenAI", "snippet": "OpenAI is an AI research laboratory consisting of..." }
  ],
  "timings": { "retrieval_ms": 12, "generation_ms": 8, "total_ms": 20 }
}


Unanswerable Example:

{ "question": "Who invented a time machine in 1800?", "top_k": 3, "refusal_threshold": 0.25 }


Response Example:

{
  "answer": "not found in crawled content",
  "sources": [
    { "url": "https://en.wikipedia.org/wiki/OpenAI", "snippet": "OpenAI is an AI research laboratory consisting of..." }
  ],
  "timings": { "retrieval_ms": 10, "generation_ms": 0, "total_ms": 10 }
}

🧪 Evaluation

Keep API running, then execute:

python eval_run.py


This evaluates:

Answerable questions (evals/answerable.json)

Unanswerable questions (evals/unanswerable.json)

✅ Sample answerable.json
[
  { "question": "Who founded OpenAI?", "expected_answer": "Elon Musk, Sam Altman, Greg Brockman, Ilya Sutskever, John Schulman, Wojciech Zaremba" }
]

🚫 Sample unanswerable.json
[
  { "question": "Who invented a time machine in 1800?", "expected_answer": "not found in crawled content" }
]


Run unit tests:

pytest -q

🏗️ Folder Structure
sde_intern_rag/
├── rag_service.py
├── requirements.txt
├── README.md
├── eval_run.py
├── test_crawler.py
└── evals/
    ├── answerable.json
    └── unanswerable.json

💡 Design & Trade-offs
Component	Decision	Reason
Chunk Size	800 chars with 200 overlap	Balances recall & precision
Embeddings	all-MiniLM-L6-v2	Compact, fast, open-source
Vector Index	FAISS (sklearn fallback)	Optimized for speed & portability
Answer Policy	Only from retrieved content	Ensures factual grounding
Refusal Logic	"not found in crawled content"	Avoids hallucinations
Observability	Logs retrieval & generation latency	Transparent performance tracking
🌟 Key Features

✅ Retrieval-grounded & hallucination-free responses

⚡ Fast vector search using FAISS

🧠 Compact MiniLM embeddings

🔌 Simple REST API via FastAPI

🧍 Refusal mechanism for missing info

🧾 Fully testable via evaluation scripts & pytest

🧰 Tech Stack

FastAPI, FAISS, SentenceTransformers, BeautifulSoup,
PyTorch, Transformers, scikit-learn, NumPy
