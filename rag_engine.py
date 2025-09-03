import json
import os
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional: cross-encoder reranker (to improve final ordering quality)
try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False


class RAGEngine:
    """
    Handles text embedding, ANN indexing (FAISS), dense retrieval,
    optional cross-encoder reranking, and persistence.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",                    # multilingual, robust default
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_hnsw: bool = True,
        hnsw_m: int = 32,
        ef_search: int = 64,
    ):
        # Embedding model
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # Build FAISS index (HNSW for scalable retrieval; fallback to Flat if desired)
        if use_hnsw:
            # Inner Product with normalized vectors ~= cosine similarity
            index = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = ef_search
            self.index = index
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        self.metadata: List[Dict] = []

        # Optional reranker
        self.reranker = None
        if _HAS_RERANKER:
            try:
                self.reranker = CrossEncoder(reranker_name)
            except Exception:
                self.reranker = None

    # ---------- Embedding ----------
    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts with L2 normalization so inner product equals cosine similarity.
        """
        return self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
        ).astype("float32")

    # ---------- Index building ----------
    def add_documents(self, documents: List[Dict]):
        """
        documents: list of {"text": str, "source": str}
        """
        if not documents:
            return
        inputs = [f"passage: {d['text']}" for d in documents]
        emb = self._embed(inputs)
        self.index.add(emb)
        self.metadata.extend(documents)

    # ---------- Retrieval (+ optional reranking) ----------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.30,
        fetch_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Vector search first (fetch_k), filter by score threshold,
        optionally rerank with a cross-encoder, then return top_k.
        """
        if fetch_k is None:
            fetch_k = max(top_k * 8, 32)

        q_vec = self._embed([f"query: {query}"])
        scores, idxs = self.index.search(q_vec, fetch_k)

        candidates: List[Dict] = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            score = float(s)
            if score < min_score:
                continue
            meta = self.metadata[i]
            candidates.append(
                {"text": meta["text"], "source": meta.get("source", "unknown"), "score": score}
            )

        if not candidates:
            return []

        # Optional reranking (cross-encoder)
        if self.reranker is not None:
            pairs = [(query, c["text"]) for c in candidates]
            rr_scores = self.reranker.predict(pairs).tolist()
            for c, r in zip(candidates, rr_scores):
                c["rerank_score"] = float(r)
            candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:top_k]

    # ---------- Persistence ----------
    def save(self, index_path: str = "faiss_index.bin", meta_path: str = "metadata.json"):
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self, index_path: str = "faiss_index.bin", meta_path: str = "metadata.json"):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)