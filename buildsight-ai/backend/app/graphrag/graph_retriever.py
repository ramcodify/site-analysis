"""BuildSight AI — Dynamic Graph Retriever

Extracts entities and executes multi-hop traversal over the in-memory NetworkX MultiDiGraph.
"""

import re
from typing import Dict, Any, List, Optional
import networkx as nx

from app.graphrag.graph_builder import knowledge_graph
from app.graphrag.graph_schema import NodeType, RelationType
from app.ai.progress_analyzer import CONSTRUCTION_STAGES


class GraphRetriever:
    """Multi-hop knowledge graph traversal retriever."""

    def __init__(self):
        self.kg = knowledge_graph

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract domain entity keywords from natural language query."""
        q_lower = query.lower()
        entities: Dict[str, Any] = {
            "worker_codes": [],
            "track_ids": [],
            "stages": [],
            "zones": [],
            "violation_types": [],
            "risk_levels": [],
        }

        # 1. Match Track IDs (e.g., "track id 43", "track 43", "track_id 43", "track#43", "id 43", "track-43")
        track_matches = re.findall(r'\b(?:track(?:_id|[-_ ]?id)?|id|track#)[-_ ]?(\d{1,5})\b', q_lower)
        for t in track_matches:
            tid = int(t)
            if tid not in entities["track_ids"]:
                entities["track_ids"].append(tid)

        # 2. Match Worker Codes (e.g., W001, W043, EMP-001, EMP-043, Worker-01, Worker 43)
        # Match EMP codes (e.g. EMP-001, EMP-43, emp 043)
        emp_matches = re.findall(r'\b(?:emp(?:loyee)?[-_ ]?)(\d{1,4})\b', q_lower)
        for m in emp_matches:
            emp_code = f"EMP-{int(m):03d}"
            if emp_code not in entities["worker_codes"]:
                entities["worker_codes"].append(emp_code)

        # Match W codes (e.g. W001, Worker-01, Worker 43)
        w_matches = re.findall(r'\b(?:w|worker[-_ ]?)(\d{1,4})\b', q_lower)
        for m in w_matches:
            w_code = f"W{int(m):03d}"
            if w_code not in entities["worker_codes"]:
                entities["worker_codes"].append(w_code)

        # Direct alphanumeric codes (e.g., EMP-001, W001, W043)
        direct_codes = re.findall(r'\b(?:emp|w)[-_]?\d{1,4}\b', q_lower)
        for code in direct_codes:
            norm = code.upper().replace(" ", "-")
            if norm not in entities["worker_codes"]:
                entities["worker_codes"].append(norm)

        # 3. Match Risk Level Tiers
        risk_terms = ["critical", "high", "medium", "low", "safe"]
        for r in risk_terms:
            if r in q_lower:
                entities["risk_levels"].append(r.upper())

        # 4. Match Construction Stages
        for idx, stage in enumerate(CONSTRUCTION_STAGES):
            if stage.lower() in q_lower or stage.lower().replace(" ", "") in q_lower:
                entities["stages"].append({"index": idx, "name": stage})

        # Check for stage words like "structural", "foundation", "excavation", etc.
        stage_keywords = {
            "prep": 0, "site prep": 0, "excavation": 1, "foundation": 2,
            "structural": 3, "structure": 3, "brickwork": 4, "masonry": 4, "brick": 4,
            "roofing": 5, "roof": 5, "plastering": 6, "plaster": 6,
            "electrical": 7, "plumbing": 7, "finishing": 8, "finish": 8
        }
        for kw, s_idx in stage_keywords.items():
            if kw in q_lower and not any(s["index"] == s_idx for s in entities["stages"]):
                entities["stages"].append({"index": s_idx, "name": CONSTRUCTION_STAGES[s_idx]})

        # 5. Match PPE Items / Violations
        ppe_types = ["helmet", "hardhat", "vest", "safety_vest", "gloves", "face_mask", "mask", "danger_zone", "boots", "zone"]
        for p in ppe_types:
            if p in q_lower:
                entities["violation_types"].append(p)

        return entities

    def traverse(self, query: str, depth: int = 2) -> Dict[str, Any]:
        """Execute subgraph extraction based on extracted query entities."""
        entities = self.extract_entities(query)
        graph = self.kg.graph

        nodes_found: List[Dict[str, Any]] = []
        edges_found: List[Dict[str, Any]] = []
        subgraph_nodes = set()

        # Find matching worker nodes
        for w_code in entities["worker_codes"]:
            for node, attrs in graph.nodes(data=True):
                if attrs.get("worker_code") == w_code or node == f"WORKER-{w_code}":
                    subgraph_nodes.add(node)

        # Find matching stage nodes
        for stage in entities["stages"]:
            for node, attrs in graph.nodes(data=True):
                if attrs.get("stage_index") == stage["index"] or stage["name"] in node:
                    subgraph_nodes.add(node)

        # Expand neighborhood
        expanded_nodes = set(subgraph_nodes)
        for node in subgraph_nodes:
            if node in graph:
                # 1-hop neighbors
                neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
                expanded_nodes.update(neighbors)

        # Collect node metadata and relationships
        for node in expanded_nodes:
            if node in graph:
                attrs = dict(graph.nodes[node])
                attrs["node_id"] = node
                nodes_found.append(attrs)

        for u, v, k, data in graph.edges(keys=True, data=True):
            if u in expanded_nodes and v in expanded_nodes:
                edges_found.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", "RELATED_TO"),
                    "data": data,
                })

        return {
            "entities": entities,
            "nodes": nodes_found,
            "relationships": edges_found,
        }


graph_retriever = GraphRetriever()
