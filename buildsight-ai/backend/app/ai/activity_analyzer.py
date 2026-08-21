"""BuildSight AI — Real-Time Motion-Based Activity Analyzer (Phase 20)

100% real inference — derives worker activity from bounding box history
produced by the YOLO+ByteTrack pipeline. No extra model download required.

Activity classification rules (per-worker, using last N frames):

  Working at Height  — worker bbox is in upper 30% of frame AND bbox area is small
  Near Heavy Machinery — not currently implemented via vision (requires object class)
  Carrying Load       — slow horizontal movement + low position
  Bending             — bbox height/width ratio drops sharply
  Walking             — consistent horizontal velocity
  Standing            — bbox barely moves, upright
  Idle                — no movement for > IDLE_THRESHOLD seconds
  Unknown             — insufficient history
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque
from app.schemas.models import ActivityResult

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
HISTORY_SIZE         = 15      # frames to keep per worker
IDLE_THRESHOLD_S     = 3.0     # seconds without movement → Idle
HEIGHT_THRESHOLD     = 0.30    # bbox top < 30% of frame → Working at Height
HEIGHT_AREA_THRESHOLD= 0.04    # bbox area < 4% of frame area at height
WALK_VELOCITY_MIN    = 3.0     # px/frame to be classified as Walking
CARRY_VELOCITY_MAX   = 2.0     # px/frame (slow movement)
CARRY_Y_THRESHOLD    = 0.65    # bbox bottom > 65% of frame → lower position
BEND_RATIO_DROP      = 0.25    # h/w ratio drops >25% compared to baseline

UNSAFE_ACTIVITIES = {"Working at Height", "Near Heavy Machinery"}


@dataclass
class _BBoxFrame:
    x1: float; y1: float; x2: float; y2: float
    timestamp: float = field(default_factory=time.time)

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def w(self)  -> float: return self.x2 - self.x1
    @property
    def h(self)  -> float: return self.y2 - self.y1
    @property
    def area(self) -> float: return self.w * self.h
    @property
    def aspect(self) -> float: return self.h / max(self.w, 1)


class ActivityAnalyzer:
    """Real-time activity analysis via bounding box motion history."""

    def __init__(self, model_path: str = ""):
        # model_path kept for API compatibility; not used here
        self.model_path = model_path
        self._loaded = True   # Always ready — rule-based
        self._load_error = ""
        # Per-worker history: worker_id → deque of _BBoxFrame
        self._history: dict[int, Deque[_BBoxFrame]] = {}

    def load(self) -> bool:
        logger.info("✓ Activity analyzer ready (motion-based rule engine)")
        return True

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def status(self) -> dict:
        return {
            "loaded": True,
            "model": "motion-rule-engine (built-in)",
            "mode": "real-time motion analysis",
            "tracked_workers": len(self._history),
        }

    # ── Public API ───────────────────────────────────────────────

    def update_worker(self, worker_id: int, bbox: tuple, frame_wh: tuple = (640, 480)):
        """Feed a new bbox observation for a worker (call every frame)."""
        x1, y1, x2, y2 = [float(v) for v in bbox]
        frame_w, frame_h = frame_wh

        if worker_id not in self._history:
            self._history[worker_id] = deque(maxlen=HISTORY_SIZE)

        self._history[worker_id].append(_BBoxFrame(x1, y1, x2, y2))

    def analyze_worker(self, worker_id: int, frame_wh: tuple = (640, 480)) -> ActivityResult:
        """Classify activity for a worker using their bbox history."""
        history = self._history.get(worker_id)
        if not history or len(history) < 3:
            return ActivityResult(
                worker_id=worker_id,
                activity="Unknown",
                confidence=0.5,
                is_unsafe=False,
                model_available=True,
            )

        frame_w, frame_h = frame_wh
        latest = history[-1]
        frame_area = frame_w * frame_h

        activity, confidence = self._classify(history, latest, frame_w, frame_h, frame_area)

        return ActivityResult(
            worker_id=worker_id,
            activity=activity,
            confidence=round(confidence, 2),
            is_unsafe=activity in UNSAFE_ACTIVITIES,
            model_available=True,
        )

    def analyze(self, crop=None, context: Optional[dict] = None) -> ActivityResult:
        """Legacy interface — requires context['worker_id'] and frame_wh."""
        worker_id = (context or {}).get("worker_id", 0)
        bbox = (context or {}).get("bbox", None)
        frame_wh = (context or {}).get("frame_wh", (640, 480))
        if bbox is not None:
            self.update_worker(worker_id, bbox, frame_wh)
        return self.analyze_worker(worker_id, frame_wh)

    def remove_worker(self, worker_id: int):
        self._history.pop(worker_id, None)

    # ── Classification Logic ─────────────────────────────────────

    def _classify(
        self,
        history: Deque[_BBoxFrame],
        latest: _BBoxFrame,
        frame_w: int,
        frame_h: int,
        frame_area: float,
    ) -> tuple[str, float]:
        """Rule-based activity classification."""

        frames = list(history)
        n = len(frames)

        # ── 1. Working at Height ──────────────────────────────────
        rel_top = latest.y1 / frame_h
        rel_area = latest.area / frame_area
        if rel_top < HEIGHT_THRESHOLD and rel_area < HEIGHT_AREA_THRESHOLD:
            return "Working at Height", 0.82

        # ── 2. Idle (no movement) ─────────────────────────────────
        now = time.time()
        oldest = frames[0]
        elapsed = now - oldest.timestamp
        if elapsed < 0.1:
            return "Unknown", 0.4

        dx_total = abs(latest.cx - oldest.cx)
        dy_total = abs(latest.cy - oldest.cy)
        total_motion = (dx_total**2 + dy_total**2) ** 0.5

        velocity = total_motion / elapsed  # px/second
        if velocity < 2.0 and elapsed >= IDLE_THRESHOLD_S:
            return "Idle", 0.85

        # ── 3. Bending / Crouching ────────────────────────────────
        if n >= 5:
            aspect_baseline = sum(f.aspect for f in frames[:3]) / 3
            aspect_current  = sum(f.aspect for f in frames[-3:]) / 3
            if aspect_baseline > 0.01:
                aspect_drop = (aspect_baseline - aspect_current) / aspect_baseline
                if aspect_drop > BEND_RATIO_DROP:
                    return "Bending", 0.74

        # ── 4. Carrying Load (slow + lower half of frame) ─────────
        rel_bottom = latest.y2 / frame_h
        if velocity < CARRY_VELOCITY_MAX * 10 and rel_bottom > CARRY_Y_THRESHOLD and velocity > 1.0:
            return "Carrying Load", 0.68

        # ── 5. Walking ────────────────────────────────────────────
        if velocity > WALK_VELOCITY_MIN * 5:
            return "Walking", 0.78

        # ── 6. Standing ───────────────────────────────────────────
        return "Standing", 0.72
