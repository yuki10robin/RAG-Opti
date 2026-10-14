# rag.py
import json, os, time
import faiss
from ollama import Client
from sentence_transformers import SentenceTransformer

from cache import SemanticCache

LARGE_MODEL = os.getenv("LARGE_MODEL", "llama3:latest")
SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5vl:3b")
TOP_K = 3

# Industry benchmark list prices per 1M tokens in USD (hosted 8B vs 3B Llama/Qwen class models)
MODEL_PRICING = {
    LARGE_MODEL: {"input": 0.20, "output": 0.20},  # $0.20 per 1M tokens
    SMALL_MODEL: {"input": 0.06, "output": 0.06},  # $0.06 per 1M tokens
}

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Safely load document index and chunks
if os.path.exists("data/index.faiss") and os.path.exists("data/chunks.json"):
    index = faiss.read_index("data/index.faiss")
    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
else:
    print("[RAG] Warning: data/index.faiss or data/chunks.json not found. Run ingest.py to index documents.")
    index = faiss.IndexFlatIP(384)
    chunks = []

client = Client()    # talks to your local Ollama at http://localhost:11434
semantic_cache = SemanticCache()

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, say: "
    "\"I don't know based on the provided documents.\" Be concise."
)

COMPLEXITY_KEYWORDS = {
    "compare", "difference", "why", "how", "analyze", "analysis",
    "tradeoff", "architecture", "relationship", "evaluate", "detailed",
    "step by step", "mechanism", "workflow", "explain"
}

def route_query(question: str) -> tuple[str, str]:
    """
    Route queries between small and large models based on question complexity.
    Returns (model_name, route_reason).
    """
    q_clean = question.strip().lower()
    words = q_clean.split()
    matched = [kw for kw in COMPLEXITY_KEYWORDS if kw in q_clean]

    if matched:
        return LARGE_MODEL, f"Complex analytical question (triggers: {', '.join(matched[:2])})"
    elif len(words) > 12 or q_clean.count("?") > 1:
        return LARGE_MODEL, f"Multi-part/lengthy query ({len(words)} words)"
    else:
        return SMALL_MODEL, f"Factual/definitional query ({len(words)} words)"

def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = MODEL_PRICING.get(model, MODEL_PRICING[LARGE_MODEL])
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000

def retrieve(question, k=TOP_K):
    if index.ntotal == 0 or not chunks:
        return []
    vec = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(vec, min(k, index.ntotal))
    return [
        {"score": float(s), **chunks[i]}
        for s, i in zip(scores[0], ids[0]) if i != -1 and i < len(chunks)
    ]

def answer(question, force_model=None, bypass_cache=False):
    start = time.perf_counter()
    q_vec = embedder.encode([question], normalize_embeddings=True).astype("float32")

    # Check semantic cache first (unless bypassed)
    if not bypass_cache:
        cached_entry, sim_score = semantic_cache.lookup(q_vec)
        if cached_entry is not None:
            cached_model = cached_entry.get("model", LARGE_MODEL)
            cached_ans = cached_entry.get("answer", "")
            # Baseline cost calculation: cost if served by LARGE_MODEL without cache (~1000 prompt + answer tokens)
            est_tokens_in = 1000
            est_tokens_out = max(20, len(cached_ans.split()) * 2)
            baseline_cost = compute_cost(LARGE_MODEL, est_tokens_in, est_tokens_out)
            return {
                "answer": cached_ans,
                "model": cached_model,
                "route_reason": "Semantic cache hit (bypassed LLM inference)",
                "sources": cached_entry.get("sources", []),
                "top_score": sim_score,
                "cached": True,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": round((time.perf_counter() - start) * 1000),
                "cost_usd": 0.0,
                "baseline_cost_usd": round(baseline_cost, 6),
                "cost_saved_usd": round(baseline_cost, 6),
            }

    # Model routing
    routed_model, route_reason = route_query(question)
    target_model = force_model or routed_model

    hits = retrieve(question)
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    # LLM execution with fallback to LARGE_MODEL if small model fails
    actual_model = target_model
    chat_options = {"num_predict": 120}  # Bound generation length for concise, low-latency responses
    try:
        response = client.chat(
            model=target_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            options=chat_options,
        )
    except Exception as e:
        if target_model != LARGE_MODEL:
            route_reason += f" (fallback to {LARGE_MODEL} due to: {e})"
            actual_model = LARGE_MODEL
            response = client.chat(
                model=LARGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                options=chat_options,
            )
        else:
            raise e

    ans_text = response.message.content
    sources = sorted({h["source"] for h in hits})
    in_tokens = response.prompt_eval_count or 0
    out_tokens = response.eval_count or 0

    # Cost computation
    actual_cost = compute_cost(actual_model, in_tokens, out_tokens)
    baseline_cost = compute_cost(LARGE_MODEL, in_tokens, out_tokens)
    cost_saved = max(0.0, baseline_cost - actual_cost)

    # Store query and response in semantic cache (unless bypassed)
    if not bypass_cache:
        semantic_cache.store(
            query_vec=q_vec,
            question=question,
            answer=ans_text,
            sources=sources,
            model=actual_model,
        )

    return {
        "answer": ans_text,
        "model": actual_model,
        "route_reason": route_reason,
        "sources": sources,
        "top_score": hits[0]["score"] if hits else 0.0,
        "cached": False,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "cost_usd": round(actual_cost, 6),
        "baseline_cost_usd": round(baseline_cost, 6),
        "cost_saved_usd": round(cost_saved, 6),
    }


if __name__ == "__main__":
    import sys
    test_q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is edge detection?"
    print(f"Query: {test_q}\nThinking...")
    res = answer(test_q)
    print("\n--- Answer ---")
    print(res["answer"])
    print(f"\nModel: {res['model']} ({res['route_reason']})")
    print(f"Cached: {res['cached']}")
    print(f"Latency: {res['latency_ms']} ms")
    print(f"Cost: ${res['cost_usd']:.6f} (Saved vs Baseline: ${res['cost_saved_usd']:.6f})")
    print("\n--- Sources ---")
    for s in res["sources"]:
        print(f"- {s}")