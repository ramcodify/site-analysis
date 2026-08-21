"""BuildSight AI — Delay Prediction Service

Extracts project features from MongoDB, calls DelayPredictor, and persists forecasts.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, Any, Optional

from app.ai.delay_predictor import DelayPredictor
from app.database.mongodb import get_db

logger = logging.getLogger(__name__)


class DelayService:
    """Service integrating MongoDB real-time telemetry with the Delay Prediction ML Model."""

    def __init__(self, predictor: Optional[DelayPredictor] = None):
        self.predictor = predictor or DelayPredictor()
        self.predictor.load()

    def get_latest_prediction(self, planned_progress_override: Optional[float] = None) -> Dict[str, Any]:
        """Compute delay prediction using real state queried from MongoDB."""
        db = get_db()

        # 1. Query latest progress record
        latest_prog = db["progress_records"].find_one({}, sort=[("timestamp", -1)])
        actual_progress = latest_prog.get("overall_progress_percentage", 0.0) if latest_prog else 0.0
        current_stage = latest_prog.get("current_stage", "Site Preparation") if latest_prog else "Site Preparation"

        from app.ai.progress_analyzer import CONSTRUCTION_STAGES
        stage_idx = CONSTRUCTION_STAGES.index(current_stage) if current_stage in CONSTRUCTION_STAGES else 0

        # Default planned progress trajectory if not provided
        planned_progress = planned_progress_override if planned_progress_override is not None else actual_progress

        # 2. Query active workers
        active_workers = db["workers"].count_documents({"is_live": True})
        if active_workers == 0:
            active_workers = max(1, db["workers"].count_documents({}))

        # 3. Query safety violations
        total_viols = db["violations"].count_documents({})
        repeated_viols = db["violations"].count_documents({"status": "OPEN"})
        safety_interruptions = db["violations"].count_documents({"severity": "HIGH"})

        # Run prediction
        result = self.predictor.predict(
            planned_progress_pct=round(float(planned_progress), 1),
            actual_progress_pct=round(float(actual_progress), 1),
            current_stage_idx=stage_idx,
            stage_elapsed_days=18.0,
            planned_stage_days=20.0,
            active_worker_count=active_workers,
            total_violations=total_viols,
            repeated_violations=repeated_viols,
            safety_interruptions=safety_interruptions,
        )

        # Persist prediction record in MongoDB
        try:
            db["delay_predictions"].insert_one({
                **result,
                "timestamp": datetime.now(timezone.utc),
            })
        except Exception as e:
            logger.debug(f"Could not persist delay prediction: {e}")

        return result


delay_service = DelayService()
