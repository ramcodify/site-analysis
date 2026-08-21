"""BuildSight AI — Compliance Engine

Detects multi-item PPE violations (Helmet, Safety Vest, Gloves, Face Mask),
danger zone intrusions, and high-risk threshold spikes with state-transition
episodic deduplication to ensure storage efficiency and zero report duplication.
"""

import uuid
import time
import logging
from typing import Optional, List, Dict, Tuple, Set
from app.ai.worker_tracker import TrackedWorkerState
from app.config import settings

logger = logging.getLogger(__name__)

# Standard Violation Types Taxonomy
VIOLATION_TYPES = {
    "NO_PPE":                  {"severity": "CRITICAL", "description": "Worker missing all required protective equipment"},
    "MISSING_HELMET_AND_VEST": {"severity": "CRITICAL", "description": "Worker detected without protective hard hat and safety vest"},
    "MISSING_HELMET":          {"severity": "HIGH",     "description": "Worker detected without protective hard hat"},
    "MISSING_SAFETY_VEST":     {"severity": "MEDIUM",   "description": "Worker detected without high-visibility safety vest"},
    "MISSING_GLOVES":          {"severity": "MEDIUM",   "description": "Worker detected without safety gloves"},
    "MISSING_FACE_MASK":       {"severity": "LOW",      "description": "Worker detected without protective face mask"},
    "DANGER_ZONE":             {"severity": "CRITICAL", "description": "Worker entered a restricted danger zone"},
    "UNSAFE_ACTIVITY":         {"severity": "HIGH",     "description": "Unsafe activity detected on site"},
    "HIGH_RISK":               {"severity": "HIGH",     "description": "Worker risk score critically elevated"},
}


class ComplianceEngine:
    """
    Episodic State-Transition Compliance Evaluator.
    
    Guarantees:
    1. Single insertion per continuous non-compliance episode (zero frame-by-frame spam).
    2. Explicit state-transition logging (e.g. missing both -> wears helmet but no vest).
    3. Automatic resolution of previous violations when compliance is restored.
    4. Enriched attribution for registered workers (worker_code, employee_number, name).
    """

    def __init__(self):
        # Memory storage: violation_id → dict
        self._violations: Dict[str, dict] = {}
        # Active episode map: worker_key -> {
        #   "missing_signature": tuple,
        #   "violation_id": str,
        #   "violation_type": str,
        #   "start_time": float,
        #   "last_seen": float,
        #   "last_bbox": tuple,
        # }
        self._active_episodes: Dict[str, dict] = {}
        self._last_event_emitted: Dict[str, float] = {}
        self.active_violation_count = 0
        self.total_violation_count = 0

    def analyze_worker(
        self, worker: TrackedWorkerState, source_id: str = "webcam"
    ) -> List[dict]:
        """
        Evaluate a worker's PPE compliance state.
        Returns a list of NEW violation events only when a sustained state transition occurs.
        Consolidates fragmented track IDs and enforces a 30s cooldown per person to prevent storage bloat.
        """
        # Skip if PPE status has not been evaluated yet
        if worker.helmet is None and worker.vest is None:
            return []

        now = time.time()
        wx1, wy1, wx2, wy2 = worker.bbox
        wcx = (wx1 + wx2) / 2.0
        wcy = (wy1 + wy2) / 2.0

        # Determine worker identity key (registered code or spatially stabilized track key)
        worker_key = worker.worker_code if worker.worker_code else None

        if not worker_key:
            # Check if there is an active unknown episode nearby (within 8.0s and similar frame location)
            for k, ep in list(self._active_episodes.items()):
                if k.startswith("unknown_") or k.startswith("track_"):
                    if (now - ep["last_seen"]) <= 8.0:
                        ep_box = ep.get("last_bbox", worker.bbox)
                        ecx = (ep_box[0] + ep_box[2]) / 2.0
                        ecy = (ep_box[1] + ep_box[3]) / 2.0
                        # Distance check
                        dist = ((wcx - ecx) ** 2 + (wcy - ecy) ** 2) ** 0.5
                        if dist < 250:  # Same person standing/moving in front of camera
                            worker_key = k
                            break

            if not worker_key:
                worker_key = f"track_{worker.worker_id}"

        # 1. Identify missing PPE items
        missing_items = []
        if worker.helmet is False:
            missing_items.append("helmet")
        if worker.vest is False:
            missing_items.append("safety_vest")
        if getattr(worker, "gloves", None) is False:
            missing_items.append("gloves")
        if getattr(worker, "face_mask", None) is False:
            missing_items.append("face_mask")

        current_sig = tuple(sorted(missing_items))
        active_ep = self._active_episodes.get(worker_key)

        new_events = []

        # ── Case A: Worker is in the SAME or similar non-compliance state ──
        if active_ep and active_ep["missing_signature"] == current_sig:
            # Update existing active episode duration & last seen
            vid = active_ep["violation_id"]
            if vid in self._violations:
                duration = now - active_ep["start_time"]
                self._violations[vid]["duration_seconds"] = round(duration, 1)
                self._violations[vid]["last_seen"] = now
                self._violations[vid]["risk_score"] = round(worker.risk_score, 1)
                self._violations[vid]["worker_id"] = worker.worker_id
            active_ep["last_seen"] = now
            active_ep["last_bbox"] = worker.bbox
            return []

        # ── Case B: State changed or initial detection ──
        # If an active episode exists and we have an event cooldown (< 30s), just update the existing violation
        last_emitted = self._last_event_emitted.get(worker_key, 0.0)
        if active_ep and (now - last_emitted) < 30.0 and current_sig:
            # Update current violation type and items without spamming new files/database records
            vtype = self._determine_violation_type(missing_items)
            vid = active_ep["violation_id"]
            if vid in self._violations:
                self._violations[vid]["violation_type"] = vtype
                self._violations[vid]["missing_items"] = missing_items
                self._violations[vid]["last_seen"] = now
                self._violations[vid]["duration_seconds"] = round(now - active_ep["start_time"], 1)
                self._violations[vid]["worker_id"] = worker.worker_id
            active_ep["missing_signature"] = current_sig
            active_ep["violation_type"] = vtype
            active_ep["last_seen"] = now
            active_ep["last_bbox"] = worker.bbox
            return []

        # 1. Close/resolve previous open violation if any
        if active_ep:
            old_vid = active_ep["violation_id"]
            if old_vid in self._violations:
                self._violations[old_vid]["status"] = "RESOLVED"
                self._violations[old_vid]["resolved_at"] = __import__("datetime").datetime.now().isoformat()
                self._violations[old_vid]["duration_seconds"] = round(now - active_ep["start_time"], 1)
            del self._active_episodes[worker_key]

        # 2. If current state has violations, create a new episode ONCE
        if current_sig:
            vtype = self._determine_violation_type(missing_items)
            vinfo = VIOLATION_TYPES.get(vtype, {"severity": "MEDIUM", "description": "PPE non-compliance"})
            
            vid = str(uuid.uuid4())
            new_viol = {
                "violation_id": vid,
                "worker_id": worker.worker_id,
                "worker_code": worker.worker_code,
                "permanent_worker_id": worker.worker_code,
                "worker_name": getattr(worker, "name", None),
                "employee_number": getattr(worker, "employee_number", None),
                "source_id": source_id,
                "violation_type": vtype,
                "missing_items": missing_items,
                "severity": vinfo["severity"],
                "risk_score": round(worker.risk_score, 1),
                "status": "OPEN",
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "duration_seconds": 0.0,
                "description": vinfo["description"],
            }

            self._violations[vid] = new_viol
            self._active_episodes[worker_key] = {
                "missing_signature": current_sig,
                "violation_id": vid,
                "violation_type": vtype,
                "start_time": now,
                "last_seen": now,
                "last_bbox": worker.bbox,
            }
            self._last_event_emitted[worker_key] = now
            worker.violation_count += 1
            new_events.append(new_viol)

            logger.info(
                f"🚨 Compliance Event: Worker {worker_key} ({getattr(worker, 'name', 'Unknown')}) → {vtype} (Missing: {', '.join(missing_items)})"
            )

        self._update_counts()
        return new_events

    def _determine_violation_type(self, missing_items: List[str]) -> str:
        """Categorize violation into standard taxonomy based on missing equipment."""
        has_helmet = "helmet" not in missing_items
        has_vest = "safety_vest" not in missing_items
        has_gloves = "gloves" not in missing_items
        has_mask = "face_mask" not in missing_items

        if not has_helmet and not has_vest and not has_gloves and not has_mask:
            return "NO_PPE"
        if not has_helmet and not has_vest:
            return "MISSING_HELMET_AND_VEST"
        if not has_helmet:
            return "MISSING_HELMET"
        if not has_vest:
            return "MISSING_SAFETY_VEST"
        if not has_gloves:
            return "MISSING_GLOVES"
        if not has_mask:
            return "MISSING_FACE_MASK"
        return "NO_PPE"

    def _update_counts(self):
        self.active_violation_count = sum(
            1 for v in self._violations.values() if v.get("status") == "OPEN"
        )
        self.total_violation_count = len(self._violations)

    def get_all_violations(self) -> List[dict]:
        return list(self._violations.values())

    def get_violations_for_worker(self, worker_id: int) -> List[dict]:
        return [v for v in self._violations.values() if v.get("worker_id") == worker_id]

    def update_violation_status(self, violation_id: str, status: str) -> Optional[dict]:
        if violation_id in self._violations:
            self._violations[violation_id]["status"] = status
            if status == "RESOLVED":
                self._violations[violation_id]["resolved_at"] = __import__("datetime").datetime.now().isoformat()
            self._update_counts()
            return self._violations[violation_id]
        return None

    def add_danger_zone_violation(self, worker: TrackedWorkerState, zone: dict, source_id: str) -> List[dict]:
        """Called when worker enters a restricted danger zone."""
        worker_key = worker.worker_code if worker.worker_code else f"track_{worker.worker_id}"
        dz_key = f"{worker_key}_danger_zone"
        now = time.time()

        if dz_key in self._active_episodes:
            vid = self._active_episodes[dz_key]["violation_id"]
            if vid in self._violations:
                self._violations[vid]["duration_seconds"] = round(now - self._active_episodes[dz_key]["start_time"], 1)
            return []

        vid = str(uuid.uuid4())
        zone_name = zone.get("name", "Restricted Area")
        viol = {
            "violation_id": vid,
            "worker_id": worker.worker_id,
            "worker_code": worker.worker_code,
            "permanent_worker_id": worker.worker_code,
            "worker_name": getattr(worker, "name", None),
            "employee_number": getattr(worker, "employee_number", None),
            "source_id": source_id,
            "violation_type": "DANGER_ZONE",
            "severity": "CRITICAL",
            "risk_score": round(max(85.0, worker.risk_score), 1),
            "status": "OPEN",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "duration_seconds": 0.0,
            "description": f"Worker entered restricted zone: {zone_name}",
        }
        self._violations[vid] = viol
        self._active_episodes[dz_key] = {
            "missing_signature": ("danger_zone",),
            "violation_id": vid,
            "violation_type": "DANGER_ZONE",
            "start_time": now,
            "last_seen": now,
        }
        self._update_counts()
        return [viol]

    def delete_violation(self, violation_id: str) -> bool:
        """Delete violation from memory and active episodes."""
        found = self._violations.pop(violation_id, None) is not None
        to_delete = [k for k, v in self._active_episodes.items() if v.get("violation_id") == violation_id]
        for k in to_delete:
            del self._active_episodes[k]
        self._update_counts()
        return found

    def clear(self):
        self._violations.clear()
        self._active_episodes.clear()
        self._update_counts()
