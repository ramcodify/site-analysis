"""BuildSight AI — Sentence Transformers & Vector Knowledge Retriever

Implements dense neural embeddings (Sentence Transformers) and TF-IDF / Cosine Similarity
vector retrieval across structured OSHA regulations and construction knowledge chunks with source traceability.
"""

import math
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from app.graphrag.knowledge_ingestion import knowledge_ingestion

logger = logging.getLogger(__name__)


class VectorKnowledgeRetriever:
    """Sentence Transformers & Vector search over ingested safety and project knowledge chunks."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.chunks = knowledge_ingestion.get_all_chunks()
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self._st_model = None
        self._st_embeddings: Optional[np.ndarray] = None
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', text) if len(w) > 2]

    def _build_index(self):
        """Compute Sentence Transformer embeddings and TF-IDF vectors for all knowledge chunks."""
        self.chunks = knowledge_ingestion.get_all_chunks()
        num_docs = len(self.chunks)
        if num_docs == 0:
            return

        # 1. Attempt Sentence Transformers embedding if installed
        try:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
            if self._st_model is None:
                self._st_model = SentenceTransformer(self.embedding_model_name)
            texts = [f"{c['doc_title']} - {c['heading']}: {c['text']}" for c in self.chunks]
            self._st_embeddings = self._st_model.encode(texts, normalize_embeddings=True)
            logger.info(f"✓ Sentence Transformers embeddings generated: {self.embedding_model_name} ({len(texts)} chunks)")
        except Exception as e:
            logger.debug(f"Sentence Transformers optional fallback to TF-IDF: {e}")
            self._st_embeddings = None

        # 2. Compute Document Frequencies & TF-IDF vectors
        df: Dict[str, int] = {}
        doc_tokens = [self._tokenize(f"{c['doc_title']} {c['heading']} {c['text']}") for c in self.chunks]

        for tokens in doc_tokens:
            unique_words = set(tokens)
            for w in unique_words:
                df[w] = df.get(w, 0) + 1

        self.idf = {w: math.log((num_docs + 1) / (count + 1)) + 1.0 for w, count in df.items()}

        self.doc_vectors = []
        for tokens in doc_tokens:
            vec: Dict[str, float] = {}
            total = max(1, len(tokens))
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            norm_sq = 0.0
            for t, count in tf.items():
                val = (count / total) * self.idf.get(t, 1.0)
                vec[t] = val
                norm_sq += val * val
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            self.doc_vectors.append({t: val / norm for t, val in vec.items()})

    def search(self, query: str, top_k: int = 4, min_score: float = 0.05) -> List[Dict[str, Any]]:
        """Retrieve top-K most relevant knowledge chunks using dense Sentence Transformers / TF-IDF."""
        if not self.chunks or not self.doc_vectors:
            self._build_index()

        # Try dense Sentence Transformers semantic similarity first
        if self._st_model is not None and self._st_embeddings is not None:
            try:
                q_emb = self._st_model.encode([query], normalize_embeddings=True)[0]
                sims = np.dot(self._st_embeddings, q_emb)
                top_indices = np.argsort(sims)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    score = float(sims[idx])
                    if score >= min_score:
                        c = self.chunks[idx]
                        results.append({
                            "chunk_id": c["chunk_id"],
                            "doc_title": c["doc_title"],
                            "source": c["source"],
                            "category": c["category"],
                            "section": c["heading"],
                            "text": c["text"],
                            "similarity_score": round(score, 4),
                            "embedding_engine": "SentenceTransformers",
                        })
                if results:
                    return results
            except Exception as e:
                logger.debug(f"Dense retrieval error, using TF-IDF: {e}")

        # TF-IDF Cosine Retrieval
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        q_tf: Dict[str, int] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        q_total = max(1, len(q_tokens))
        q_vec: Dict[str, float] = {}
        q_norm_sq = 0.0
        for t, count in q_tf.items():
            if t in self.idf:
                val = (count / q_total) * self.idf[t]
                q_vec[t] = val
                q_norm_sq += val * val

        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0
        q_vec = {t: val / q_norm for t, val in q_vec.items()}

        scores: List[Tuple[int, float]] = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            dot = sum(doc_vec.get(t, 0.0) * q_val for t, q_val in q_vec.items())
            if dot >= min_score:
                scores.append((idx, dot))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "doc_title": c["doc_title"],
                "source": c["source"],
                "category": c["category"],
                "section": c["heading"],
                "text": c["text"],
                "similarity_score": round(score, 4),
                "embedding_engine": "TF-IDF Vector Space",
            })
        return results


vector_retriever = VectorKnowledgeRetriever()
