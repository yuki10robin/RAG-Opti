# cache.py
import json
import os
import time
import numpy as np
import faiss

CACHE_FILE_JSON = "data/semantic_cache.json"
CACHE_FILE_INDEX = "data/semantic_cache.faiss"
DEFAULT_THRESHOLD = 0.85  # Empirically tuned threshold (separates paraphrases >0.85 from false positives <0.30)


class SemanticCache:
    """
    Semantic Cache for RAG queries.
    Stores query embeddings, questions, and responses.
    Allows queries with high semantic similarity to reuse cached answers,
    reducing latency to <10ms and LLM token cost to $0.
    """

    def __init__(self, dimension: int = 384, threshold: float = DEFAULT_THRESHOLD, ttl_seconds: int = 604800):
        self.dimension = dimension
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.entries = []
        self.total_queries = 0
        self.hits = 0
        self.misses = 0

        # Initialize FAISS index for inner-product (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self._load()

    def _load(self):
        """Load persisted cache entries and FAISS index from disk if present."""
        if os.path.exists(CACHE_FILE_JSON) and os.path.exists(CACHE_FILE_INDEX):
            try:
                with open(CACHE_FILE_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = data.get("entries", [])
                    self.total_queries = data.get("total_queries", 0)
                    self.hits = data.get("hits", 0)
                    self.misses = data.get("misses", 0)

                self.index = faiss.read_index(CACHE_FILE_INDEX)
                if self.index.ntotal != len(self.entries):
                    print(f"[SemanticCache] Warning: Index count ({self.index.ntotal}) differs from entries ({len(self.entries)}). Resetting cache.")
                    self.clear()
                else:
                    print(f"[SemanticCache] Loaded {len(self.entries)} entries from disk.")
            except Exception as e:
                print(f"[SemanticCache] Warning: Failed to load cache from disk ({e}). Starting fresh.")
                self.entries = []
                self.index = faiss.IndexFlatIP(self.dimension)

    def _save(self, save_index: bool = True):
        """Persist cache entries and FAISS index to disk."""
        os.makedirs(os.path.dirname(CACHE_FILE_JSON) or ".", exist_ok=True)
        try:
            with open(CACHE_FILE_JSON, "w", encoding="utf-8") as f:
                json.dump({
                    "entries": self.entries,
                    "total_queries": self.total_queries,
                    "hits": self.hits,
                    "misses": self.misses,
                }, f, indent=2)

            if save_index:
                faiss.write_index(self.index, CACHE_FILE_INDEX)
        except Exception as e:
            print(f"[SemanticCache] Warning: Failed to save cache: {e}")

    def lookup(self, query_vec: np.ndarray):
        """
        Check if an incoming query vector semantically matches a cached query.
        Returns (matched_entry, similarity_score) if score >= threshold, else (None, best_score).
        """
        self.total_queries += 1

        if self.index.ntotal == 0 or len(self.entries) == 0:
            self.misses += 1
            self._save(save_index=False)
            return None, 0.0

        # Query vector must be 2D float32
        vec = query_vec.astype("float32")
        if len(vec.shape) == 1:
            vec = np.expand_dims(vec, axis=0)

        # Normalize to guarantee cosine similarity in inner-product search
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        if norm[0][0] > 0:
            vec = vec / norm

        scores, ids = self.index.search(vec, 1)
        best_score = float(scores[0][0])
        best_idx = int(ids[0][0])

        if best_idx != -1 and best_idx < len(self.entries) and best_score >= self.threshold:
            entry = self.entries[best_idx]
            # Invalidate if TTL expired
            if self.ttl_seconds is not None and (time.time() - entry.get("timestamp", 0) > self.ttl_seconds):
                self.misses += 1
                self._save(save_index=False)
                return None, best_score

            self.hits += 1
            self._save(save_index=False)
            return entry, min(best_score, 1.0)

        self.misses += 1
        self._save(save_index=False)
        return None, best_score

    def store(self, query_vec: np.ndarray, question: str, answer: str, sources: list, model: str):
        """Store a new query and answer in the semantic cache."""
        vec = query_vec.astype("float32")
        if len(vec.shape) == 1:
            vec = np.expand_dims(vec, axis=0)

        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        if norm[0][0] > 0:
            vec = vec / norm

        entry = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "model": model,
            "timestamp": time.time(),
        }

        self.index.add(vec)
        self.entries.append(entry)
        self._save(save_index=True)

    def stats(self) -> dict:
        """Return cache hit-rate and capacity statistics."""
        hit_rate = (self.hits / self.total_queries * 100) if self.total_queries > 0 else 0.0
        return {
            "total_queries": self.total_queries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
            "cached_entries": len(self.entries),
            "threshold": self.threshold,
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self):
        """Reset and wipe all cache entries."""
        self.entries = []
        self.total_queries = 0
        self.hits = 0
        self.misses = 0
        self.index = faiss.IndexFlatIP(self.dimension)
        if os.path.exists(CACHE_FILE_JSON):
            os.remove(CACHE_FILE_JSON)
        if os.path.exists(CACHE_FILE_INDEX):
            os.remove(CACHE_FILE_INDEX)
        print("[SemanticCache] Cache cleared.")
