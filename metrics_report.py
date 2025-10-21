import time
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

# Example questions (you can change these)
ANSWERABLE_QUESTION = {
    "question": "What is Python and who created it?",
    "top_k": 3
}

UNANSWERABLE_QUESTION = {
    "question": "Who won the FIFA World Cup in 2022?",
    "top_k": 3
}

def ask_question(payload):
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    total_time = (time.time() - start_time) * 1000  # ms
    data = response.json()

    return {
        "question": payload["question"],
        "answer": data.get("answer"),
        "sources": data.get("sources", []),
        "status": "refusal" if "not found" in data.get("answer", "").lower() else "answered",
        "total_time_ms": round(total_time, 2)
    }

def main():
    print("🧠 Running evaluation on /ask endpoint...\n")

    # Run two test queries
    results = []
    for query in [ANSWERABLE_QUESTION, UNANSWERABLE_QUESTION]:
        result = ask_question(query)
        print(f"Q: {result['question']}")
        print(f"→ A: {result['answer']}")
        print(f"⏱  Time: {result['total_time_ms']} ms\n")
        results.append(result)

    # Calculate simple metrics
    answerable_correct = sum(1 for r in results if r["status"] == "answered")
    unanswerable_refusals = sum(1 for r in results if r["status"] == "refusal")
    avg_time = sum(r["total_time_ms"] for r in results) / len(results)

    metrics = {
        "total_queries": len(results),
        "answerable_correct": answerable_correct,
        "unanswerable_refusals": unanswerable_refusals,
        "average_total_ms": round(avg_time, 2),
        "details": results
    }

    # Save as JSON
    with open("metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("📊 Metrics report generated: metrics_report.json\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
