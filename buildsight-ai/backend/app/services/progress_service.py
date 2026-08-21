"""BuildSight AI — Progress Service

Manages construction progress tracking and history.
"""

import logging
from datetime import datetime, timezone
from app.ai.progress_analyzer import ProgressAnalyzer

logger = logging.getLogger(__name__)


class ProgressService:
    """Service for tracking and querying construction progress."""

    def __init__(self, analyzer: ProgressAnalyzer):
        self.analyzer = analyzer
        self._history: list[dict] = []

    def record_progress(self, source_id: str = "webcam"):
        """Record current progress to history."""
        result = self.analyzer.analyze()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_id": source_id,
            "current_stage": result.current_stage,
            "stage_confidence": result.stage_confidence,
            "stage_completion": result.stage_completion_percentage,
            "overall_progress": result.overall_progress_percentage,
            "progress_status": result.progress_status,
        }
        self._history.append(entry)
        return entry

    def get_history(self, limit: int = 100) -> list[dict]:
        """Get progress history."""
        return self._history[-limit:]

    def clear_history(self):
        """Clear all history."""
        self._history.clear()
