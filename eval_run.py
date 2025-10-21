# eval_run.py
import os
import time
import json
import requests

ROOT = os.path.dirname(__file__)
EVAL_DIR = os.path.join(ROOT, "evals")
SERVER = os.environ.get("RAG_SERVER", "http://127.0.0.1:8000")

def load_eval(name):
    with open(os.path.join(EVAL_DIR, f"{name}.json")) as f:
        return json.load(f)

def call(endpoint, payload):
    url = f"{SERVER}{endpoint}"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def run_eval(eval_name):
    cfg = load_eval(eval_name)
    print(f"\n=== Running eval: {eval_name} ===")
    # 1) Crawl
    crawl_payload = {
        "start_url": cfg["start_url"],
        "max_pages": cfg["crawl"].get("max_pages", 10),
        "max_depth": cfg["crawl"].get("max_depth", 2),
        "crawl_delay_ms": cfg["crawl"].get("crawl_delay_ms", 200)
    }
    print("Crawling:", crawl_payload)
    crawl_res = call("/crawl", crawl_payload)
    print("Crawl result:", crawl_res)

    # 2) Index
    print("Indexing crawled pages...")
    index_res = call("/index", {"chunk_size": 800, "chunk_overlap": 200})
    print("Index result:", index_res)

    # 3) Ask
    ask_payload = {"question": cfg["question"], "top_k": cfg.get("top_k", 5)}
    print("Asking:", ask_payload)
    ask_res = call("/ask", ask_payload)
    print("Ask result:", json.dumps(ask_res, indent=2))

    # 4) Evaluate
    if cfg.get("expect_refusal"):
        ok = (ask_res.get("answer", "").strip().lower() == "not found in crawled content")
        print("Expect refusal:", cfg.get("expect_refusal"), "=>", "PASS" if ok else "FAIL")
        return ok
    else:
        expected = cfg.get("expected_phrase", "").lower()
        found = False
        for src in ask_res.get("sources", []):
            snippet = src.get("snippet", "").lower()
            if expected in snippet:
                found = True
                break
        print(f"Expected phrase '{cfg.get('expected_phrase')}' found in top-k sources? =>", found)
        return found

if __name__ == "__main__":
    # run both evals
    ok1 = run_eval("answerable")
    ok2 = run_eval("unanswerable")
    print("\nSummary:")
    print("answerable:", "PASS" if ok1 else "FAIL")
    print("unanswerable:", "PASS" if ok2 else "FAIL")
    if ok1 and ok2:
        print("\nAll evals passed ✅")
    else:
        print("\nSome evals failed ❌")
