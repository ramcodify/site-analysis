"""BuildSight AI — Worker Tracker (YOLO11 + ByteTrack)

Maintains temporary ByteTrack IDs across frames using Ultralytics tracking.
"""

import time
import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrackedWorkerState:
    worker_id: int  # Temporary ByteTrack track ID
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float = 0.0
    # Permanent Identity Fields
    permanent_worker_id: Optional[str] = None  # e.g. "W001" or None
    worker_code: Optional[str] = None         # "W001"
    name: Optional[str] = None                # "John Doe" or None
    identity_status: str = "UNKNOWN"          # "REGISTERED" | "UNKNOWN" | "UNCERTAIN"
    recognition_confidence: Optional[float] = None
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    face_crop_base64: Optional[str] = None
    photo_url: Optional[str] = None
    # PPE & Safety
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    gloves: Optional[bool] = None
    face_mask: Optional[bool] = None
    missing_ppe: list = field(default_factory=list)
    compliance_status: str = "UNKNOWN"
    ppe_compliance: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "SAFE"
    risk_factors: list = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    tracking_duration: float = 0.0
    violation_count: int = 0
    activity: str = "Unknown"
    activity_confidence: float = 0.0


class WorkerTracker:
    """YOLO11 + ByteTrack-powered worker tracker."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        input_size: int = 640,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.input_size = input_size
        self.track_buffer = track_buffer
        self.match_threshold = match_threshold

        self._model = None
        self._loaded = False
        self._load_error = ""
        self._device_name = device

        # State: track_id → TrackedWorkerState
        self._workers: dict[int, TrackedWorkerState] = {}
        # For stale cleanup
        self._last_seen: dict[int, float] = {}
        self._stale_timeout = track_buffer * 0.1  # seconds
        self._next_fallback_id = 9000

    def reset(self):
        """Clear all active tracked worker states and last seen timestamps."""
        self._workers.clear()
        self._last_seen.clear()
        self._next_fallback_id = 9000
        logger.info("✓ WorkerTracker state cleared")

    def load(self) -> bool:
        try:
            # pyrefly: ignore [missing-import]
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            # Detect device
            # pyrefly: ignore [missing-import]
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                self._device_name = "cuda"
            else:
                self._device_name = "cpu"
            self._loaded = True
            logger.info(f"✓ YOLO tracker loaded: {self.model_path} on {self._device_name}")
            return True
        except Exception as e:
            self._load_error = str(e)
            self._loaded = False
            logger.error(f"YOLO load failed: {e}")
            return False

    @property
    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "model": self.model_path,
            "device": self._device_name,
            "active_workers": len(self._workers),
            **({"error": self._load_error} if not self._loaded else {}),
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def track(self, frame: np.ndarray) -> list[TrackedWorkerState]:
        """Run YOLO + ByteTrack on a frame, return tracked workers."""
        if not self._loaded or self._model is None:
            return []

        try:
            results = self._model.track(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                device=self._device_name,
                imgsz=self.input_size,
                classes=[0],        # person only
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        except Exception as e:
            logger.error(f"Track inference error: {e}")
            return []

        now = datetime.now(timezone.utc)
        now_ts = time.time()
        seen_ids: set[int] = set()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                track_id_t = box.id
                # Only track objects with confirmed multi-frame ByteTrack tracking
                if track_id_t is None:
                    continue
                track_id = int(track_id_t.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [float(c) for c in box.xyxy[0]]

                bw = x2 - x1
                bh = y2 - y1

                # Physical human geometry filter (suppress non-human background pipes, machinery, horizontal objects)
                if bh < 40 or (bw / max(1.0, bh)) > 1.30 or (bw / max(1.0, bh)) < 0.15:
                    continue

                if track_id not in self._workers:
                    self._workers[track_id] = TrackedWorkerState(
                        worker_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        first_seen=now,
                        last_seen=now,
                    )
                else:
                    w = self._workers[track_id]
                    w.bbox = (x1, y1, x2, y2)
                    w.confidence = conf
                    w.last_seen = now
                    if w.first_seen:
                        w.tracking_duration = (now - w.first_seen).total_seconds()

                self._last_seen[track_id] = now_ts
                seen_ids.add(track_id)

        # Remove stale workers
        stale = [
            tid for tid, ts in self._last_seen.items()
            if (now_ts - ts) > self._stale_timeout
        ]
        for tid in stale:
            self._workers.pop(tid, None)
            self._last_seen.pop(tid, None)

        active_list = [self._workers[tid] for tid in seen_ids if tid in self._workers]
        # Sort by area descending and filter nested / heavily overlapping duplicate detections
        sorted_workers = sorted(active_list, key=lambda w: (w.bbox[2] - w.bbox[0]) * (w.bbox[3] - w.bbox[1]), reverse=True)
        filtered_workers = []
        for w in sorted_workers:
            x1, y1, x2, y2 = w.bbox
            area = max(1.0, (x2 - x1) * (y2 - y1))
            is_duplicate = False
            for kept in filtered_workers:
                kx1, ky1, kx2, ky2 = kept.bbox
                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if (inter_area / area) > 0.55:
                        is_duplicate = True
                        break
            if not is_duplicate:
                filtered_workers.append(w)

        return filtered_workers

    def get_all_workers(self) -> list[TrackedWorkerState]:
        return list(self._workers.values())

    def get_worker(self, worker_id: int) -> Optional[TrackedWorkerState]:
        return self._workers.get(worker_id)

    def clear(self):
        self._workers.clear()
        self._last_seen.clear()
