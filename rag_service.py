import os
import time
import pickle
import logging
import sqlite3
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup
import tldextract
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----- Config / Defaults -----
CHUNK_SIZE_DEFAULT = 800
CHUNK_OVERLAP_DEFAULT = 200
MAX_PAGES_DEFAULT = 40
CRAWL_DELAY_DEFAULT_MS = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small model
VECTOR_INDEX_PATH = "vector_index.pkl"
MAPPING_PATH = "id_to_meta.pkl"
PAGE_DB = "pages.db"
LOGLEVEL = os.getenv("LOGLEVEL", "INFO")

logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_service")

# ----- Utilities -----
def domain_of(url: str) -> str:
    e = tldextract.extract(url)
    return f"{e.domain}.{e.suffix}" if e.suffix else e.domain

def same_registrable_domain(a: str, b: str) -> bool:
    try:
        return domain_of(a) == domain_of(b)
    except Exception:
        return False

def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = urljoin(base, href.strip())
    parsed = urlparse(href)
    if parsed.scheme not in ("http", "https"):
        return None
    return parsed._replace(fragment="").geturl()

import urllib.robotparser
def can_fetch_url(base_url: str, target_url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, target_url)
    except Exception:
        return True

# ----- Page fetching & cleaning -----
def fetch_html(url: str, timeout: int = 10) -> Optional[str]:
    headers = {"User-Agent": "sde-intern-rag-bot/1.0 (+https://example.com)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.debug(f"fetch_html failed for {url}: {e}")
        return None

def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for sel in soup(["script", "style", "nav", "header", "footer", "form", "aside", "noscript", "svg"]):
        sel.decompose()
    main = soup.find("main") or soup.find("article")
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        divs = soup.find_all("div")
        if divs:
            best = max(divs, key=lambda d: len(d.get_text(" ", strip=True)))
            text = best.get_text(separator="\n", strip=True)
            if len(text) < 200:
                body = soup.find("body")
                text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
    text = ' '.join(text.split())
    return text.strip()

# ----- SQLite Persistence -----
def init_page_db(db_path=PAGE_DB):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY,
            text TEXT,
            crawled_at REAL
        );
    """)
    conn.commit()
    conn.close()

def save_page(url: str, text: str, db_path=PAGE_DB):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO pages (url, text, crawled_at) VALUES (?, ?, ?);",
              (url, text, time.time()))
    conn.commit()
    conn.close()

def load_all_pages(db_path=PAGE_DB) -> Dict[str, str]:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT url, text FROM pages;")
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}

# ----- Crawler -----
def crawl(start_url: str, max_pages: int = MAX_PAGES_DEFAULT, max_depth: int = 3, crawl_delay_ms: int = CRAWL_DELAY_DEFAULT_MS) -> Dict[str, Any]:
    init_page_db()
    start_time = time.time()
    start_domain = domain_of(start_url)
    q = deque([(start_url, 0)])
    seen, results = set(), []
    added, skipped = 0, 0
    while q and added < max_pages:
        url, depth = q.popleft()
        if url in seen:
            continue
        seen.add(url)
        if depth > max_depth:
            skipped += 1
            continue
        if not same_registrable_domain(start_url, url):
            skipped += 1
            continue
        if not can_fetch_url(start_url, url):
            skipped += 1
            continue
        html = fetch_html(url)
        if not html:
            skipped += 1
            continue
        text = extract_main_text(html)
        if not text or len(text) < 20:
            skipped += 1
            continue
        save_page(url, text)
        results.append(url)
        added += 1
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            norm = normalize_url(url, a.get("href"))
            if norm and norm not in seen and same_registrable_domain(start_url, norm):
                q.append((norm, depth + 1))
        time.sleep(crawl_delay_ms / 1000.0)
    return {"page_count": added, "skipped_count": skipped, "urls": results, "ms": int((time.time() - start_time) * 1000)}

# ----- Chunker -----
def chunk_text(text: str, chunk_size=CHUNK_SIZE_DEFAULT, chunk_overlap=CHUNK_OVERLAP_DEFAULT) -> List[str]:
    text = text.replace("\r", "")
    length = len(text)
    if length <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= length:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks

# ----- Vector Index -----
class VectorIndex:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = None
        self.id_to_meta = {}
        self._sklearn_vectors = None

    def _build_sklearn(self, vectors: np.ndarray):
        neigh = NearestNeighbors(metric="cosine")
        neigh.fit(vectors)
        return neigh

    def build_from_docs(self, docs: Dict[str, str], chunk_size=CHUNK_SIZE_DEFAULT, chunk_overlap=CHUNK_OVERLAP_DEFAULT):
        all_chunks, metas = [], []
        for url, text in docs.items():
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                metas.append({"url": url, "chunk_index": i, "snippet": ch[:500]})
        if not all_chunks:
            raise RuntimeError("No chunks to index")
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms
        self.index = self._build_sklearn(embeddings)
        self._sklearn_vectors = embeddings
        self.id_to_meta = {i: m for i, m in enumerate(metas)}
        self._persist()

    def _persist(self):
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump(self.id_to_meta, f)
        pickle.dump({"nn": self.index, "vectors": self._sklearn_vectors}, open("sklearn_index.pkl", "wb"))

    def load(self):
        if os.path.exists(MAPPING_PATH):
            self.id_to_meta = pickle.load(open(MAPPING_PATH, "rb"))
        if os.path.exists("sklearn_index.pkl"):
            data = pickle.load(open("sklearn_index.pkl", "rb"))
            self.index = data["nn"]
            self._sklearn_vectors = data["vectors"]

    def query(self, text: str, top_k=5) -> List[Dict[str, Any]]:
        emb = self.embedder.encode([text], convert_to_numpy=True)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        distances, idxs = self.index.kneighbors(emb, n_neighbors=top_k)
        scores = (1.0 - distances[0]).tolist()
        idxs = idxs[0].tolist()
        return [{"id": int(i), "score": float(s), "meta": self.id_to_meta[int(i)]} for i, s in zip(idxs, scores)]

VECTOR = VectorIndex()

# ----- Offline text generator -----
def generate_with_hf(prompt: str, max_tokens=100) -> str:
    snippet = prompt[:120].replace("\n", " ")
    return f"(offline mode) Concise answer generated locally. Prompt snippet: '{snippet}...'"


# ----- Prompt builder -----
def build_grounded_prompt(retrieved: List[Dict[str, Any]], question: str) -> str:
    top_chunks = sorted(retrieved, key=lambda r: r["score"], reverse=True)[:3]
    header = ("You are a concise assistant. ONLY use the CONTEXT below to answer the question. "
              "Answer in 1-3 sentences. If not enough info, say 'not found in crawled content'.\n\n")
    ctx_parts = []
    for i, r in enumerate(top_chunks):
        snippet = r["meta"].get("snippet", "")[:300]
        url = r["meta"].get("url", "")
        ctx_parts.append(f"[[{i}]] URL: {url}\nSnippet: {snippet}")
    ctx = "\n---\n".join(ctx_parts)
    return f"{header}\nCONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nAnswer concisely and include citations like [URL]."

# ----- Refusal logic -----
def should_refuse(retrieved: List[Dict[str, Any]], similarity_threshold=0.35) -> bool:
    if not retrieved:
        return True
    top_score = max(r["score"] for r in retrieved)
    return top_score < similarity_threshold

# ----- FastAPI App -----
app = FastAPI(title="RAG Service")

class CrawlRequest(BaseModel):
    start_url: str
    max_pages: int = MAX_PAGES_DEFAULT
    max_depth: int = 3
    crawl_delay_ms: int = CRAWL_DELAY_DEFAULT_MS

class IndexRequest(BaseModel):
    chunk_size: int = CHUNK_SIZE_DEFAULT
    chunk_overlap: int = CHUNK_OVERLAP_DEFAULT

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    refusal_threshold: float = 0.35

@app.on_event("startup")
def startup_load():
    init_page_db()
    if os.path.exists(MAPPING_PATH) or os.path.exists("sklearn_index.pkl"):
        try:
            VECTOR.load()
            logger.info("Loaded existing vector index.")
        except Exception as e:
            logger.warning(f"Index load failed: {e}")

@app.post("/crawl")
def api_crawl(req: CrawlRequest):
    start = time.time()
    result = crawl(req.start_url, req.max_pages, req.max_depth, req.crawl_delay_ms)
    result["total_ms"] = int((time.time() - start) * 1000)
    return result

@app.post("/index")
def api_index(req: IndexRequest):
    docs = load_all_pages()
    VECTOR.build_from_docs(docs, req.chunk_size, req.chunk_overlap)
    return {"vector_count": len(VECTOR.id_to_meta)}

@app.post("/ask")
def api_ask(req: AskRequest):
    total_start = time.time()
    retrieval_start = time.time()
    retrieved = VECTOR.query(req.question, top_k=req.top_k)
    retrieval_ms = int((time.time() - retrieval_start) * 1000)

    if should_refuse(retrieved, req.refusal_threshold):
        total_ms = int((time.time() - total_start) * 1000)
        return {
            "answer": "not found in crawled content",
            "sources": [],
            "timings": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": 0,
                "total_ms": total_ms
            }
        }

    prompt = build_grounded_prompt(retrieved, req.question)
    gen_start = time.time()
    ans = generate_with_hf(prompt)
    generation_ms = int((time.time() - gen_start) * 1000)
    total_ms = int((time.time() - total_start) * 1000)

    sources = [{"url": r["meta"]["url"], "snippet": r["meta"]["snippet"], "score": r["score"]} for r in retrieved]
    return {
        "answer": ans,
        "sources": sources,
        "timings": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms
        }
    }
