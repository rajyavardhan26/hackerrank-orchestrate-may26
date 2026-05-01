"""Local embedding-based vector store for document retrieval."""

import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, CACHE_DIR
from corpus_loader import Chunk, build_corpus_index


class EmbeddingStore:
    """Manages embeddings and similarity search over the support corpus."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = np.array([])
        self._fitted = False

    def _cache_path(self) -> Path:
        safe_name = self.model_name.replace("/", "_")
        return CACHE_DIR / f"embeddings_{safe_name}.pkl"

    def fit(self, chunks: List[Chunk], use_cache: bool = True):
        """Encode all chunks and build the index."""
        self.chunks = chunks
        cache_path = self._cache_path()

        if use_cache and cache_path.exists():
            print(f"[Embeddings] Loading cached index from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.embeddings = cached["embeddings"]
            self.chunks = cached["chunks"]
            self._fitted = True
            return

        print(f"[Embeddings] Encoding {len(chunks)} chunks with {self.model_name} ...")
        texts = [c.text for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self._fitted = True

        if use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump({"embeddings": self.embeddings, "chunks": self.chunks}, f)
            print(f"[Embeddings] Cached index to {cache_path}")

    def search(self, query: str, top_k: int = 5,
               company_filter: str = None) -> List[Tuple[Chunk, float]]:
        """Return top-k most similar chunks with scores."""
        if not self._fitted:
            raise RuntimeError("Store not fitted. Call fit() first.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_vec, self.embeddings)[0]

        indices = np.argsort(sims)[::-1]
        results = []
        for idx in indices:
            chunk = self.chunks[idx]
            if company_filter and chunk.company != company_filter:
                continue
            results.append((chunk, float(sims[idx])))
            if len(results) >= top_k:
                break
        return results


def get_store() -> EmbeddingStore:
    """Singleton-like helper: build or load the embedding store."""
    store = EmbeddingStore()
    chunks, _ = build_corpus_index()
    store.fit(chunks, use_cache=True)
    return store


if __name__ == "__main__":
    store = get_store()
    res = store.search("How do I reset my password?", top_k=3)
    for chunk, score in res:
        print(f"[{score:.3f}] {chunk.company}/{chunk.product_area} — {chunk.text[:120]}...")
