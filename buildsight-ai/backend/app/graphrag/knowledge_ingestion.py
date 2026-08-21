"""BuildSight AI — Knowledge Base Ingestion & Chunking Engine

Ingests comprehensive construction safety regulations, OSHA compliance standards, stage guidelines,
and hazard definitions with source traceability into structured knowledge chunks.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SAFETY_KNOWLEDGE_DIR = DATA_DIR / "safety_knowledge"

INITIAL_KNOWLEDGE_DOCS = [
    {
        "doc_id": "DOC-OSHA-1926-PPE",
        "title": "OSHA 1926 Safety and Health Regulations for Construction — PPE",
        "source": "OSHA 1926 Subpart E",
        "category": "Safety Standard",
        "sections": [
            {
                "section_id": "SEC-HELMET-100",
                "heading": "Head Protection Requirements (1926.100)",
                "content": "Employees working in areas where there is a possible danger of head injury from impact, falling or flying objects, or electrical shock and burns shall be protected by protective helmets (hard hats). Helmets must meet ANSI Z89.1 standards. Failure to wear a helmet in active work areas is classified as a HIGH severity safety violation.",
            },
            {
                "section_id": "SEC-VEST-200",
                "heading": "High-Visibility Safety Apparel (1926.200)",
                "content": "All site personnel and workers exposed to vehicular traffic or heavy moving machinery must wear high-visibility safety vests (Class 2 or Class 3). Reflective vest stripes must remain unobstructed during all active shifts.",
            },
            {
                "section_id": "SEC-GLOVES-300",
                "heading": "Hand and Cut Protection Guidelines (1926.95)",
                "content": "Workers handling structural steel, masonry blocks, concrete pouring, or abrasive power tools must wear heavy-duty cut-resistant safety gloves to prevent lacerations, punctures, and chemical burns.",
            },
            {
                "section_id": "SEC-MASK-400",
                "heading": "Respiratory Protection in High Dust & Masonry Zones (1926.1153)",
                "content": "During excavation, concrete cutting, dry plastering, and masonry brickwork, workers exposed to respirable crystalline silica dust must wear appropriate N95 or P100 face masks to maintain respiratory safety.",
            },
        ]
    },
    {
        "doc_id": "DOC-OSHA-1926-FALL",
        "title": "OSHA 1926 Fall Protection Standards & Edge Security",
        "source": "OSHA 1926 Subpart M (1926.501)",
        "category": "Fall Protection",
        "sections": [
            {
                "section_id": "SEC-FALL-6FT",
                "heading": "Unprotected Sides & Leading Edges (6-Foot Rule)",
                "content": "Each employee on a walking/working surface with an unprotected side or edge 6 feet (1.8 m) or more above a lower level shall be protected from falling by guardrail systems, safety net systems, or personal fall arrest systems (harness and anchored lifeline).",
            },
            {
                "section_id": "SEC-ROOF-FALL",
                "heading": "Roofing & Structural Truss Fall Safeguards",
                "content": "During roofing operations, workers on low-slope roofs must use warning line systems combined with guardrails or personal fall arrest systems. Steep roofs (>4:12 pitch) strictly require full fall arrest harnesses.",
            }
        ]
    },
    {
        "doc_id": "DOC-OSHA-1926-ELECTRICAL",
        "title": "OSHA 1926 Electrical Safety & Lockout/Tagout Standards",
        "source": "OSHA 1926 Subpart K (1926.400)",
        "category": "Electrical Safety",
        "sections": [
            {
                "section_id": "SEC-GFCI-PROTECT",
                "heading": "Ground Fault Circuit Interrupters (GFCI)",
                "content": "All 120-volt, single-phase, 15- and 20-ampere receptacle outlets on construction sites that are not part of the permanent wiring shall have approved Ground Fault Circuit Interrupters (GFCI) for personnel protection.",
            },
            {
                "section_id": "SEC-OVERHEAD-POWER",
                "heading": "Clearance from High-Voltage Overhead Power Lines",
                "content": "Equipment such as cranes, boom trucks, and scaffolds must maintain a minimum radial clearance of 10 feet (3 meters) from energized power lines up to 50kV, plus 0.4 inches per additional kV.",
            }
        ]
    },
    {
        "doc_id": "DOC-SITE-STAGES-01",
        "title": "Standard Construction Lifecycle Stages & Milestones",
        "source": "BuildSight Project Execution Guide",
        "category": "Project Management",
        "sections": [
            {
                "section_id": "SEC-STAGES-SUMMARY",
                "heading": "Nine Core Construction Stages",
                "content": "The construction lifecycle comprises 9 discrete stages: 1. Site Preparation (clearing, boundary survey), 2. Excavation (soil removal, trenching), 3. Foundation (slab, rebar footing), 4. Structural Work (columns, beams, steel framework), 5. Brickwork (masonry walls), 6. Roofing (trusses, waterproofing), 7. Plastering (internal/external mortar coating), 8. Electrical and Plumbing (conduits, piping), and 9. Finishing (painting, glazing, handover).",
            },
            {
                "section_id": "SEC-DELAY-RISK",
                "heading": "Schedule Variance and Delay Drivers",
                "content": "Delays predominantly stem from negative progress variance, prolonged stage durations exceeding planned baselines, worker availability shortages, and critical safety hazard work stoppages. Unresolved high-severity violations compound schedule slippage.",
            },
        ]
    },
    {
        "doc_id": "DOC-ZONES-HAZARDS-02",
        "title": "Site Danger Zones and Restricted Area Protocols",
        "source": "Site Safety Plan & Hazard Analysis",
        "category": "Zone Safety",
        "sections": [
            {
                "section_id": "SEC-EXCAVATION-PIT",
                "heading": "Excavation Pit & Trench Safety (1926.651)",
                "content": "Excavation pits deeper than 1.5 meters are classified as RESTRICTED danger zones. Unauthorized entry without shoring protection, trench shields, or designated spotters incurs immediate CRITICAL risk scoring and stop-work orders.",
            },
            {
                "section_id": "SEC-CRANE-RADIUS",
                "heading": "Crane Swing Radius & Overhead Loads (1926.550)",
                "content": "The radius beneath operating cranes and mobile hoists is a WARNING/RESTRICTED hazard zone. Personnel are strictly prohibited from standing under suspended loads regardless of PPE status.",
            },
        ]
    }
]


class KnowledgeIngestion:
    """Manages parsing, chunking, and storage of construction safety and project documents."""

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self._load_builtins()

    def _load_builtins(self):
        """Parse structured documents into fine-grained traceable knowledge chunks."""
        self.chunks = []
        for doc in INITIAL_KNOWLEDGE_DOCS:
            for sec in doc["sections"]:
                chunk = {
                    "chunk_id": f"{doc['doc_id']}_{sec['section_id']}",
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["title"],
                    "source": doc["source"],
                    "category": doc["category"],
                    "section_id": sec["section_id"],
                    "heading": sec["heading"],
                    "text": sec["content"],
                }
                self.chunks.append(chunk)

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return self.chunks


knowledge_ingestion = KnowledgeIngestion()
