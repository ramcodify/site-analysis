"""BuildSight AI — Hybrid Knowledge Retriever

Combines:
  1. Multi-hop Graph Traversal (NetworkX MultiDiGraph)
  2. TF-IDF Cosine Vector Knowledge Search (OSHA 1926 & SOPs)
  3. Live MongoDB Telemetry & ML Predictions (Delay Engine & Progress Records)
"""

from typing import Dict, Any, List, Optional
from collections import Counter

from app.database.mongodb import get_db
from app.graphrag.vector_retriever import VectorKnowledgeRetriever
from app.graphrag.graph_retriever import graph_retriever
from app.ai.delay_predictor import delay_predictor


class HybridRetriever:
    """Hybrid multi-modal knowledge retriever combining graph paths and vector embeddings."""

    def __init__(self):
        self.vector_retriever = VectorKnowledgeRetriever()
        self.graph_retriever = graph_retriever
        self.delay_predictor = delay_predictor

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute hybrid search combining vector retrieval, graph traversal, and ML forecasts."""
        db = get_db()
        q_lower = query.lower()

        # 1. Vector Search across OSHA and SOPs
        doc_chunks = self.vector_retriever.search(query, top_k=4)

        # 2. Graph Traversal
        graph_res = self.graph_retriever.traverse(query)
        entities = graph_res.get("entities", {})
        graph_nodes = graph_res.get("nodes", [])
        graph_edges = graph_res.get("relationships", [])

        # 3. Determine target worker code
        worker_code = None
        if entities.get("worker_codes"):
            worker_code = entities["worker_codes"][0]

        # 4. Aggregate Missing PPE frequencies from MongoDB violations
        ppe_frequency = None
        if "frequently missing" in q_lower or "most" in q_lower or "ppe" in q_lower:
            viols = list(db["violations"].find({}))
            vtypes = [v.get("violation_type") for v in viols if v.get("violation_type")]
            counts = Counter(vtypes)
            if counts:
                top_item = counts.most_common(1)[0][0]
                ppe_frequency = {
                    "counts": dict(counts),
                    "top_missing_item": top_item,
                }

        # 5. Delay Prediction Forecast if relevant
        delay_prediction = None
        if any(k in q_lower for k in ["delay", "schedule", "progress", "forecast", "behind"]):
            latest_prog = db["progress_records"].find_one({}, sort=[("timestamp", -1)])
            viols_count = db["violations"].count_documents({})
            open_viols = list(db["violations"].find({"status": "OPEN"}))
            wcodes = [v.get("worker_code") for v in open_viols if v.get("worker_code")]
            rep_viols = len([w for w, c in Counter(wcodes).items() if c >= 2])

            p_act = latest_prog.get("overall_progress_percentage", 45.0) if latest_prog else 45.0
            p_plan = 65.0
            s_idx = latest_prog.get("current_stage_index", 3) if latest_prog else 3

            delay_prediction = self.delay_predictor.predict(
                planned_progress_pct=p_plan,
                actual_progress_pct=p_act,
                current_stage_idx=s_idx,
                stage_elapsed_days=25.0,
                planned_stage_days=20.0,
                active_worker_count=12,
                total_violations=viols_count,
                repeated_violations=rep_viols,
                safety_interruptions=2,
            )

        track_ids = entities.get("track_ids", [])
        track_id = track_ids[0] if track_ids else None

        return {
            "doc_chunks": doc_chunks,
            "graph_entities": [n.get("node_id", "") for n in graph_nodes],
            "relationships_used": [f"{e['source']} -[{e['relation']}]-> {e['target']}" for e in graph_edges],
            "worker_code": worker_code,
            "track_id": track_id,
            "track_ids": track_ids,
            "ppe_frequency": ppe_frequency,
            "delay_prediction": delay_prediction,
            "entities": entities,
        }


hybrid_retriever = HybridRetriever()
