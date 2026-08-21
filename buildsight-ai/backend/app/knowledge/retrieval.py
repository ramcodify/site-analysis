"""BuildSight AI — Local Safety Knowledge Retrieval

Modular interface for safety regulation knowledge retrieval.
Supports local documents (TXT, Markdown) for now.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SafetyKnowledgeRetrieval:
    """Local safety knowledge retrieval system.

    Loads safety documents from the data/safety_knowledge/ directory
    and provides simple text-based retrieval.
    """

    def __init__(self, knowledge_dir: str = ""):
        self.knowledge_dir = knowledge_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "safety_knowledge"
        )
        self._documents: list[dict] = []
        self._loaded = False

    def load(self) -> bool:
        """Load safety documents from the knowledge directory."""
        try:
            knowledge_path = Path(self.knowledge_dir)
            if not knowledge_path.exists():
                logger.warning(f"Knowledge directory not found: {self.knowledge_dir}")
                return False

            for file_path in knowledge_path.glob("*"):
                if file_path.suffix.lower() in ('.txt', '.md', '.markdown'):
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    self._documents.append({
                        "source": file_path.name,
                        "content": content,
                    })

            if self._documents:
                self._loaded = True
                logger.info(f"✓ Loaded {len(self._documents)} safety knowledge documents")
            else:
                logger.info("No safety knowledge documents found")

            return self._loaded
        except Exception as e:
            logger.error(f"Failed to load safety knowledge: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "documents": len(self._documents),
            "directory": self.knowledge_dir,
        }

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Simple keyword-based search across safety documents.

        This is a basic implementation. Can be replaced with:
        - Local embeddings + vector search
        - GraphRAG with knowledge graph
        """
        if not self._loaded:
            return []

        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        for doc in self._documents:
            content_lower = doc["content"].lower()
            # Score based on keyword matches
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                # Extract relevant snippet
                for word in query_words:
                    idx = content_lower.find(word)
                    if idx >= 0:
                        start = max(0, idx - 100)
                        end = min(len(doc["content"]), idx + 200)
                        snippet = doc["content"][start:end].strip()
                        results.append({
                            "source": doc["source"],
                            "snippet": snippet,
                            "score": score,
                        })
                        break

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
