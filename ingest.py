# ingest.py
import glob
import json
import os
import sys

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_WORDS = 300   # words per chunk
OVERLAP = 50        # words shared between neighbouring chunks


def extract_text_from_pdf(path):
    """Extract text from all pages in a PDF file."""
    try:
        reader = PdfReader(path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n".join(pages_text)
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return ""


def load_docs(folder="data.pdf"):
    """Read every .pdf, .txt, and .md file under the folder."""
    docs = []
    for path in glob.glob(f"{folder}/**/*", recursive=True):
        if path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
            if text.strip():
                docs.append((path, text))
        elif path.lower().endswith((".txt", ".md")):
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
                if text.strip():
                    docs.append((path, text))
    return docs


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    """Split text into overlapping pieces of about `size` words."""
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


if __name__ == "__main__":
    docs = load_docs()
    if not docs:
        print("No supported files (.pdf, .txt, .md) found in data.pdf. Add some and try again.")
        sys.exit(1)
    print(f"Loaded {len(docs)} documents")

    # 1. Chunk every document, remembering where each chunk came from
    chunks = []
    for path, text in docs:
        for piece in chunk_text(text):
            chunks.append({"source": path, "text": piece})
    print(f"Created {len(chunks)} chunks")

    # 2. Turn each chunk into a vector (embedding)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,   # makes inner product equal cosine similarity
        show_progress_bar=True,
    ).astype("float32")

    # 3. Put the vectors in a FAISS index for fast similarity search
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # 4. Save the index and the chunk texts
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/index.faiss")
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print("Saved data/index.faiss and data/chunks.json")

    # Invalidate semantic cache so stale answers from old document versions are pruned
    from cache import SemanticCache
    SemanticCache().clear()
    print("[Ingestion] Semantic cache invalidated.")