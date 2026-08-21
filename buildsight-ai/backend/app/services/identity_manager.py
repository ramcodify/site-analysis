"""BuildSight AI — Permanent Identity & Temporal Confirmation Manager

Bridges temporary ByteTrack track IDs and permanent registered Worker IDs (e.g., W001).
Maintains temporal voting windows, handles occlusion, prevents identity switching,
and merges historical safety/tracking statistics across track reconnects.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

from app.config import settings
from app.ai.face_recognition_service import face_recognition_service

logger = logging.getLogger(__name__)


@dataclass
class TrackIdentityState:
    track_id: int
    confirmed_worker_code: Optional[str] = None
    confirmed_worker_name: Optional[str] = None
    confirmation_score: float = 0.0
    is_confirmed: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # Rolling history of (worker_code, similarity, timestamp)
    match_history: deque = field(default_factory=lambda: deque(maxlen=settings.face_history_window))
    # Count of continuous frames without face detection
    frames_without_face: int = 0
    # Last detected face bounding box relative to frame [x1, y1, x2, y2]
    last_face_bbox: Optional[Tuple[float, float, float, float]] = None


class IdentityManager:
    """
    Manages real-time mapping between temporary ByteTrack Track IDs and
    permanent Registered Worker IDs (W001, W002...).
    """

    def __init__(
        self,
        match_threshold: Optional[float] = None,
        confirmation_frames: Optional[int] = None,
        history_window: Optional[int] = None,
    ):
        self.match_threshold = match_threshold or settings.face_match_threshold
        self.confirmation_frames = confirmation_frames or settings.face_confirmation_frames
        self.history_window = history_window or settings.face_history_window

        # Active track states: track_id -> TrackIdentityState
        self._tracks: Dict[int, TrackIdentityState] = {}

    def reset(self):
        """Clear active track states (e.g. on stream restart)."""
        self._tracks.clear()

    def cleanup_stale_tracks(self, active_track_ids: set[int]):
        """Remove tracks that are no longer active."""
        current_ids = set(self._tracks.keys())
        stale_ids = current_ids - active_track_ids
        for tid in stale_ids:
            del self._tracks[tid]

    def update_track_face(
        self,
        track_id: int,
        face_crop_or_image,
        raw_face_data: Optional[Any] = None,
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Process face detection for a track in the current frame and update identity.

        Returns dict:
          - permanent_worker_id: str | None (e.g. "W001")
          - worker_code: str | None
          - name: str | None
          - identity_status: "REGISTERED" | "UNKNOWN" | "UNCERTAIN"
          - recognition_confidence: float | None
          - face_bbox: tuple | None
        """
        now = time.time()
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackIdentityState(track_id=track_id, first_seen=now, last_seen=now)

        state = self._tracks[track_id]
        state.last_seen = now
        if face_bbox:
            state.last_face_bbox = face_bbox

        # If image or crop is provided, extract embedding and match
        embedding = None
        match_code = None
        match_sim = 0.0
        match_name = None

        if face_crop_or_image is not None and face_crop_or_image.size > 0:
            embedding = face_recognition_service.extract_embedding(face_crop_or_image, raw_face=raw_face_data)

        if embedding is not None:
            match_code, match_sim, match_name = face_recognition_service.match_worker(
                embedding, threshold=self.match_threshold
            )
            state.match_history.append((match_code, match_sim, now, match_name))
            state.frames_without_face = 0
        else:
            state.frames_without_face += 1

        # ── Temporal Confirmation Logic ──────────────────────────────
        self._evaluate_temporal_confirmation(state)

        # Build output structure
        if state.is_confirmed and state.confirmed_worker_code:
            return {
                "permanent_worker_id": state.confirmed_worker_code,
                "worker_code": state.confirmed_worker_code,
                "name": state.confirmed_worker_name or "Registered Worker",
                "identity_status": "REGISTERED",
                "recognition_confidence": round(state.confirmation_score, 2),
                "face_bbox": state.last_face_bbox,
            }
        elif len(state.match_history) >= 2 and any(m[0] is not None for m in state.match_history):
            # We have seen potential matches but not reached confirmation threshold yet
            best_recent = max(state.match_history, key=lambda m: m[1])
            if best_recent[1] >= (self.match_threshold * 0.85):
                return {
                    "permanent_worker_id": None,
                    "worker_code": None,
                    "name": "Checking Identity...",
                    "identity_status": "UNCERTAIN",
                    "recognition_confidence": round(best_recent[1], 2),
                    "face_bbox": state.last_face_bbox,
                }

        # Default: Unknown worker
        return {
            "permanent_worker_id": None,
            "worker_code": None,
            "name": f"Unknown Worker (Track #{track_id})",
            "identity_status": "UNKNOWN",
            "recognition_confidence": None,
            "face_bbox": state.last_face_bbox,
        }

    def _evaluate_temporal_confirmation(self, state: TrackIdentityState):
        """
        Evaluate recent match history to confirm permanent identity.
        Requires at least `confirmation_frames` matches for the same worker_code.
        """
        if not state.match_history:
            return

        # Count occurrences and aggregate similarity per worker_code
        candidate_counts: Dict[str, int] = {}
        candidate_sims: Dict[str, List[float]] = {}
        candidate_names: Dict[str, str] = {}

        for code, sim, _, name in state.match_history:
            if code is not None and sim >= self.match_threshold:
                candidate_counts[code] = candidate_counts.get(code, 0) + 1
                candidate_sims.setdefault(code, []).append(sim)
                if name:
                    candidate_names[code] = name

        if not candidate_counts:
            return

        # Find worker with most matches in the window
        best_code, count = max(candidate_counts.items(), key=lambda item: item[1])
        avg_sim = sum(candidate_sims[best_code]) / len(candidate_sims[best_code])

        if count >= self.confirmation_frames and avg_sim >= self.match_threshold:
            # Prevent assigning the same confirmed worker_code to two distinct active tracks
            # unless this track has significantly higher evidence
            conflict = False
            for other_tid, other_state in self._tracks.items():
                if other_tid != state.track_id and other_state.confirmed_worker_code == best_code:
                    # Conflict found: keep the one with higher confirmation score
                    if other_state.confirmation_score > avg_sim:
                        conflict = True
                        break

            if not conflict:
                if state.confirmed_worker_code != best_code:
                    logger.info(
                        f"✓ IDENTITY CONFIRMED: Track #{state.track_id} → {best_code} "
                        f"({candidate_names.get(best_code, '')}) [avg_sim={avg_sim:.2f}, votes={count}]"
                    )
                state.confirmed_worker_code = best_code
                state.confirmed_worker_name = candidate_names.get(best_code, best_code)
                state.confirmation_score = avg_sim
                state.is_confirmed = True

    def manual_link_identity(self, track_id: int, worker_code: str, worker_name: str = ""):
        """Allow administrator to manually confirm / override an identity."""
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackIdentityState(track_id=track_id)
        state = self._tracks[track_id]
        state.confirmed_worker_code = worker_code
        state.confirmed_worker_name = worker_name or worker_code
        state.confirmation_score = 1.0
        state.is_confirmed = True
        logger.info(f"Manual identity link: Track #{track_id} → {worker_code} ({worker_name})")


# Global singleton instance
identity_manager = IdentityManager()
