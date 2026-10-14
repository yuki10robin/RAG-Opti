# test_cache.py
import time
import numpy as np
from cache import SemanticCache, DEFAULT_THRESHOLD
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("=== 1. Basic Semantic Cache Hit/Miss Test ===")
cache = SemanticCache(threshold=DEFAULT_THRESHOLD, ttl_seconds=2)
cache.clear()

q1 = "What is edge detection?"
v1 = embedder.encode([q1], normalize_embeddings=True).astype("float32")

# Lookup before storing -> should be MISS
hit, score = cache.lookup(v1)
assert hit is None, "Expected initial lookup to be a MISS"
print(f"Cold lookup '{q1}': hit={hit is not None}, score={score:.4f} [PASS]")

# Store simulated answer
cache.store(
    query_vec=v1,
    question=q1,
    answer="Edge detection is an image processing technique for finding boundaries of objects.",
    sources=["data.pdf/Edge detection.pdf"],
    model="llama3:latest"
)

# Test exact same question -> should be HIT (~1.0000)
hit, score = cache.lookup(v1)
assert hit is not None, "Expected identical query to be a HIT"
print(f"Lookup identical '{q1}': hit=True, score={score:.4f} [PASS]")

# Test rephrased question -> should be HIT (>=0.85)
q2 = "Can you explain edge detection?"
v2 = embedder.encode([q2], normalize_embeddings=True).astype("float32")
hit, score = cache.lookup(v2)
assert hit is not None, f"Expected paraphrased query to be a HIT, got score={score}"
print(f"Lookup rephrased '{q2}': hit=True, score={score:.4f} [PASS]")

# Test unrelated question -> should be MISS (<0.85)
q3 = "What is supply chain management?"
v3 = embedder.encode([q3], normalize_embeddings=True).astype("float32")
hit, score = cache.lookup(v3)
assert hit is None, f"Expected unrelated query to be a MISS, got score={score}"
print(f"Lookup unrelated '{q3}': hit=False, score={score:.4f} [PASS]")

print("\n=== 2. TTL Expiration Test ===")
print("Waiting 2.5 seconds for TTL expiration (ttl_seconds=2)...")
time.sleep(2.5)
hit_expired, score_expired = cache.lookup(v1)
assert hit_expired is None, "Expected entry to expire after TTL exceeded"
print(f"Lookup after TTL: hit={hit_expired is not None} (expired as expected) [PASS]")

print("\n=== 3. Threshold Calibration & Separation Sweep ===")
# Calibrate threshold against semantically equivalent vs distinct pairs
test_pairs = [
    # Positive pairs: direct rephrasings of the same question intent
    ("What is edge detection?", "Can you explain edge detection?", True),
    ("What is tokenization?", "Can you explain tokenization?", True),
    ("What is the Bullwhip Effect?", "Can you explain the Bullwhip Effect?", True),
    ("Explain Canny edge detector steps", "What are the steps of Canny edge detection?", True),
    # Negative pairs: completely distinct topics
    ("What is edge detection?", "What is tokenization?", False),
    ("Define BLEU score", "How does Canny edge detection work?", False),
    ("What is cloud computing?", "Explain Naive Bayes classifier", False),
    ("What is supply chain management?", "What is edge detection?", False),
]

similarities = []
for qa, qb, is_match in test_pairs:
    va = embedder.encode([qa], normalize_embeddings=True).astype("float32")
    vb = embedder.encode([qb], normalize_embeddings=True).astype("float32")
    sim = float((va @ vb.T)[0, 0])
    similarities.append((qa, qb, is_match, sim))
    label = "MATCH" if is_match else "DISTINCT"
    print(f"[{label:8s}] Sim: {sim:.4f} | '{qa}' vs '{qb}'")

pos_sims = [s for _, _, m, s in similarities if m]
neg_sims = [s for _, _, m, s in similarities if not m]
min_pos = min(pos_sims)
max_neg = max(neg_sims)

print(f"\nPositive pair min sim: {min_pos:.4f} | Negative pair max sim: {max_neg:.4f}")
print(f"Separation margin: {min_pos - max_neg:.4f}")
assert min_pos > DEFAULT_THRESHOLD > max_neg, (
    f"Threshold {DEFAULT_THRESHOLD} does not cleanly separate classes! (min_pos={min_pos:.4f}, max_neg={max_neg:.4f})"
)
print(f"Tuned threshold {DEFAULT_THRESHOLD} verified: cleanly separates matches ({min_pos:.4f}+) from false positives (<={max_neg:.4f}) [PASS]")

print("\nCache Stats:", cache.stats())
print("All semantic cache tests passed!")
