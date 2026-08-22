"""BuildSight AI — Dynamic Construction Knowledge Graph Builder

Populates and synchronizes an in-memory NetworkX MultiDiGraph directly from MongoDB collections:
  - registered_workers
  - workers (sessions)
  - violations
  - progress_records
  - danger_zones
  - delay_predictions
"""

import networkx as nx
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.database.mongodb import get_db
from app.graphrag.graph_schema import NodeType, RelationType
from app.ai.progress_analyzer import CONSTRUCTION_STAGES

logger = logging.getLogger(__name__)


class ConstructionKnowledgeGraph:
    """Dynamic Knowledge Graph for construction site intelligence."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._last_sync: Optional[datetime] = None

    def sync_from_mongodb(self):
        """Synchronize the entire graph from MongoDB collections idempotently."""
        db = get_db()
        self.graph.clear()

        # 1. Project Root Node
        proj_id = "PROJ-BUILDSIGHT-01"
        self.graph.add_node(
            proj_id,
            node_type=NodeType.PROJECT,
            name="BuildSight AI Flagship Site",
            status="ACTIVE",
            created_at=datetime.now().isoformat(),
        )

        # 2. Construction Stage Chain Nodes
        stage_nodes = []
        for idx, stage_name in enumerate(CONSTRUCTION_STAGES):
            s_node_id = f"STAGE-{idx:02d}-{stage_name.replace(' ', '_')}"
            self.graph.add_node(
                s_node_id,
                node_type=NodeType.CONSTRUCTION_STAGE,
                name=stage_name,
                stage_index=idx,
            )
            self.graph.add_edge(proj_id, s_node_id, relation=RelationType.CONTAINS)
            stage_nodes.append(s_node_id)
            if idx > 0:
                self.graph.add_edge(stage_nodes[idx - 1], s_node_id, relation=RelationType.NEXT_STAGE)

        # 3. PPE Item Core Nodes
        ppe_items = ["helmet", "safety_vest", "gloves", "face_mask"]
        for item in ppe_items:
            item_id = f"PPE-{item.upper()}"
            self.graph.add_node(
                item_id,
                node_type=NodeType.PPE_ITEM,
                name=item.replace("_", " ").title(),
                category="Standard PPE",
            )
            self.graph.add_edge(proj_id, item_id, relation=RelationType.REQUIRES)

        # 4. Registered Workers
        workers_cursor = db["registered_workers"].find({})
        for w in workers_cursor:
            wcode = w.get("worker_code")
            if not wcode:
                continue
            w_id = f"WORKER-{wcode}"
            self.graph.add_node(
                w_id,
                node_type=NodeType.WORKER,
                worker_code=wcode,
                name=w.get("name", "Unknown Worker"),
                employee_number=w.get("employee_number", ""),
                department=w.get("department", ""),
                role=w.get("role", ""),
                active_status=w.get("active_status", "ACTIVE"),
            )
            self.graph.add_edge(proj_id, w_id, relation=RelationType.CONTAINS)

        # 4b. Live Tracked Workers (from in-memory tracker + MongoDB sessions)
        try:
            from app.services.video_processor import video_processor
            live_workers = video_processor.tracker.get_all_workers()
            for lw in live_workers:
                track_node_id = f"TRACK-{lw.worker_id}"
                if track_node_id not in self.graph:
                    self.graph.add_node(
                        track_node_id,
                        node_type=NodeType.WORKER,
                        track_id=lw.worker_id,
                        name=lw.name or f"Unknown Worker (Track #{lw.worker_id})",
                        worker_code=lw.worker_code or lw.permanent_worker_id or f"TRACK-{lw.worker_id}",
                        role="Site Operative",
                        risk_score=lw.risk_score,
                        risk_level=lw.risk_level,
                        compliance_status=lw.compliance_status,
                        is_live=True,
                    )
                    self.graph.add_edge(proj_id, track_node_id, relation=RelationType.CONTAINS)
                    if lw.permanent_worker_id and f"WORKER-{lw.permanent_worker_id}" in self.graph:
                        self.graph.add_edge(track_node_id, f"WORKER-{lw.permanent_worker_id}", relation=RelationType.RELATED_TO)
                    # Edges to missing PPE
                    if lw.helmet is False:
                        self.graph.add_edge(track_node_id, "PPE-HELMET", relation=RelationType.MISSING)
                    if lw.vest is False:
                        self.graph.add_edge(track_node_id, "PPE-SAFETY_VEST", relation=RelationType.MISSING)
        except Exception as e:
            logger.debug(f"Live worker graph sync note: {e}")

        # Also add worker sessions from MongoDB
        try:
            sessions_cursor = db["worker_sessions"].find({}).sort("last_seen", -1).limit(40)
            for s in sessions_cursor:
                tid = s.get("track_id")
                if tid is not None:
                    track_node_id = f"TRACK-{tid}"
                    if track_node_id not in self.graph:
                        self.graph.add_node(
                            track_node_id,
                            node_type=NodeType.WORKER,
                            track_id=tid,
                            name=s.get("name") or f"Worker (Track #{tid})",
                            worker_code=s.get("worker_code") or f"TRACK-{tid}",
                            is_live=bool(s.get("is_live", False)),
                        )
                        self.graph.add_edge(proj_id, track_node_id, relation=RelationType.CONTAINS)
        except Exception as e:
            logger.debug(f"Session graph sync note: {e}")

        # 5. Danger Zones
        zones_cursor = db["danger_zones"].find({})
        for z in zones_cursor:
            zid = z.get("zone_id", "zone-default")
            z_node_id = f"ZONE-{zid}"
            self.graph.add_node(
                z_node_id,
                node_type=NodeType.ZONE,
                zone_id=zid,
                name=z.get("name", "Restricted Area"),
                zone_type=z.get("zone_type", "RESTRICTED"),
                risk_weight=z.get("risk_weight", 30.0),
            )
            self.graph.add_edge(proj_id, z_node_id, relation=RelationType.CONTAINS)

        # 6. Violations & Edges
        viols_cursor = db["violations"].find({}).sort("timestamp", -1).limit(150)
        for v in viols_cursor:
            vid = v.get("violation_id")
            if not vid:
                continue
            v_node_id = f"VIOL-{vid}"
            wcode = v.get("worker_code")
            vtype = v.get("violation_type", "SAFETY_VIOLATION")

            self.graph.add_node(
                v_node_id,
                node_type=NodeType.VIOLATION,
                violation_id=vid,
                violation_type=vtype,
                severity=v.get("severity", "MEDIUM"),
                risk_score=v.get("risk_score", 0.0),
                status=v.get("status", "OPEN"),
                timestamp=str(v.get("timestamp")),
            )

            # Link to Worker
            if wcode and f"WORKER-{wcode}" in self.graph:
                self.graph.add_edge(f"WORKER-{wcode}", v_node_id, relation=RelationType.VIOLATED)

            # Link to PPE Item
            if "HELMET" in vtype:
                self.graph.add_edge(v_node_id, "PPE-HELMET", relation=RelationType.MISSING)
            elif "VEST" in vtype:
                self.graph.add_edge(v_node_id, "PPE-SAFETY_VEST", relation=RelationType.MISSING)
            elif "GLOVES" in vtype:
                self.graph.add_edge(v_node_id, "PPE-GLOVES", relation=RelationType.MISSING)
            elif "MASK" in vtype:
                self.graph.add_edge(v_node_id, "PPE-FACE_MASK", relation=RelationType.MISSING)

            # Link to current active stage
            current_stage_node = stage_nodes[3] if len(stage_nodes) > 3 else stage_nodes[0]
            self.graph.add_edge(v_node_id, current_stage_node, relation=RelationType.DURING_STAGE)

        # 7. Progress Records
        prog_cursor = db["progress_records"].find({}).sort("timestamp", -1).limit(10)
        for p in prog_cursor:
            pid = str(p.get("_id"))
            p_node_id = f"PROG-{pid}"
            cstage = p.get("current_stage", "Structural Work")
            self.graph.add_node(
                p_node_id,
                node_type=NodeType.PROGRESS_RECORD,
                current_stage=cstage,
                stage_confidence=p.get("stage_confidence", 0.0),
                stage_completion=p.get("stage_completion_percentage", 0.0),
                overall_progress=p.get("overall_progress_percentage", 0.0),
                timestamp=str(p.get("timestamp")),
            )
            self.graph.add_edge(proj_id, p_node_id, relation=RelationType.RECORDED_BY)

            # Link progress to matching stage
            for s_node in stage_nodes:
                if cstage in s_node:
                    self.graph.add_edge(p_node_id, s_node, relation=RelationType.DURING_STAGE)
                    break

        self._last_sync = datetime.now()
        logger.info(f"✓ Knowledge Graph synchronized: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def get_graph_stats(self) -> Dict[str, Any]:
        """Return counts of nodes and relationships by type."""
        if self.graph.number_of_nodes() == 0:
            self.sync_from_mongodb()

        node_types_count = {}
        for _, data in self.graph.nodes(data=True):
            ntype = str(data.get("node_type", "Unknown"))
            node_types_count[ntype] = node_types_count.get(ntype, 0) + 1

        rel_types_count = {}
        for _, _, data in self.graph.edges(data=True):
            rtype = str(data.get("relation", "RELATED_TO"))
            rel_types_count[rtype] = rel_types_count.get(rtype, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_distribution": node_types_count,
            "edge_distribution": rel_types_count,
            "last_synced": self._last_sync.isoformat() if self._last_sync else None,
        }

    def get_subgraph_for_visualization(self, max_nodes: int = 60) -> Dict[str, Any]:
        """Export nodes and links in a D3/ForceGraph compatible JSON shape."""
        if self.graph.number_of_nodes() == 0:
            self.sync_from_mongodb()

        nodes_list = []
        node_ids = set()
        for node_id, data in list(self.graph.nodes(data=True))[:max_nodes]:
            node_ids.add(node_id)
            nodes_list.append({
                "id": node_id,
                "label": data.get("name") or data.get("worker_code") or data.get("violation_type") or node_id,
                "type": str(data.get("node_type", "Entity")),
                "properties": {k: str(v) for k, v in data.items() if k not in ["name", "node_type"]},
            })

        edges_list = []
        for u, v, data in self.graph.edges(data=True):
            if u in node_ids and v in node_ids:
                edges_list.append({
                    "source": u,
                    "target": v,
                    "relation": str(data.get("relation", "RELATED_TO")),
                })

        return {
            "nodes": nodes_list,
            "links": edges_list,
            "total_nodes": len(nodes_list),
            "total_links": len(edges_list),
        }


knowledge_graph = ConstructionKnowledgeGraph()
