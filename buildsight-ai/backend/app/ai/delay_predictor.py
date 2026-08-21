"""BuildSight AI — Real Construction Delay Predictor & Explainability Engine

Loads the trained GradientBoosting regressor & calibrated classifier to predict:
  - Delay probability (0.0 to 1.0)
  - Expected delay duration (days)
  - Predicted project completion date
  - Feature contribution breakdown and human-readable explainability
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
DEFAULT_DELAY_MODEL = MODELS_DIR / "delay_model.joblib"

FEATURE_NAMES = [
    "planned_progress_pct",
    "actual_progress_pct",
    "progress_variance",
    "current_stage_idx",
    "stage_elapsed_days",
    "planned_stage_days",
    "active_worker_count",
    "total_violations",
    "repeated_violations",
    "safety_interruptions",
]

FEATURE_DESCRIPTIONS = {
    "planned_progress_pct": "Planned progress schedule target",
    "actual_progress_pct": "Current measured progress completion",
    "progress_variance": "Schedule deviation (Actual % - Planned %)",
    "current_stage_idx": "Current active construction stage index",
    "stage_elapsed_days": "Observed duration in current construction stage",
    "planned_stage_days": "Baseline planned duration for current stage",
    "active_worker_count": "Active workers present on site",
    "total_violations": "Total confirmed safety violations logged",
    "repeated_violations": "Persistent/repeated worker safety violations",
    "safety_interruptions": "Critical safety hazard work stoppages",
}


class DelayPredictor:
    """Production delay prediction and feature explainability engine."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(DEFAULT_DELAY_MODEL)
        self.bundle = None
        self.loaded = False
        self._load_error = ""

    def load(self) -> bool:
        if not Path(self.model_path).exists():
            self._load_error = f"Delay model file not found at {self.model_path}"
            logger.warning(self._load_error)
            return False
        try:
            self.bundle = joblib.load(self.model_path)
            self.loaded = True
            logger.info(f"✓ Construction Delay Predictor loaded from {self.model_path}")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load delay model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def predict(
        self,
        planned_progress_pct: float,
        actual_progress_pct: float,
        current_stage_idx: int,
        stage_elapsed_days: float,
        planned_stage_days: float,
        active_worker_count: int,
        total_violations: int,
        repeated_violations: int,
        safety_interruptions: int,
        project_start_date: Optional[datetime] = None,
        planned_duration_days: float = 180.0,
    ) -> Dict[str, Any]:
        """
        Execute model inference to predict delay days, probability, completion date,
        and provide feature-level explanations.
        """
        if not self.loaded:
            success = self.load()
            if not success:
                raise RuntimeError(f"Delay model is not loaded: {self._load_error}")

        progress_variance = round(float(actual_progress_pct - planned_progress_pct), 1)

        input_vector = [
            float(planned_progress_pct),
            float(actual_progress_pct),
            float(progress_variance),
            int(current_stage_idx),
            float(stage_elapsed_days),
            float(planned_stage_days),
            int(active_worker_count),
            int(total_violations),
            int(repeated_violations),
            int(safety_interruptions),
        ]

        X = np.array([input_vector])

        regressor = self.bundle["regressor"]
        classifier = self.bundle["classifier"]

        pred_days = float(regressor.predict(X)[0])
        pred_days = max(0.0, round(pred_days, 1))

        probs = classifier.predict_proba(X)[0]
        delay_prob = float(probs[1]) if len(probs) > 1 else 0.5
        delay_prob = round(float(delay_prob), 2)

        # Baseline dates
        start_date = project_start_date or (datetime.now(timezone.utc) - timedelta(days=30))
        planned_completion = start_date + timedelta(days=planned_duration_days)
        predicted_completion = planned_completion + timedelta(days=pred_days)

        # Feature contributors
        importances = self.bundle.get("feature_importances", [])
        top_contributors = []
        for item in importances[:4]:
            feat_name = item["feature"]
            desc = FEATURE_DESCRIPTIONS.get(feat_name, feat_name)
            top_contributors.append({
                "feature": feat_name,
                "importance": item["importance"],
                "description": desc,
            })

        # Structured diagnostic explanation
        explanations = []
        if progress_variance < -5.0:
            explanations.append(f"Actual progress ({actual_progress_pct:.1f}%) is lagging behind planned target ({planned_progress_pct:.1f}%) by {abs(progress_variance):.1f}%.")
        elif progress_variance > 5.0:
            explanations.append(f"Actual progress ({actual_progress_pct:.1f}%) is ahead of schedule by {progress_variance:.1f}%.")

        if stage_elapsed_days > planned_stage_days:
            explanations.append(f"Current stage duration ({stage_elapsed_days:.0f} days) has exceeded the planned baseline ({planned_stage_days:.0f} days).")

        if total_violations >= 5:
            explanations.append(f"High cumulative safety violations ({total_violations}) and safety stoppages ({safety_interruptions}) are introducing operational friction.")

        if active_worker_count < 8:
            explanations.append(f"On-site active worker density ({active_worker_count}) is below recommended crew capacity.")

        if not explanations:
            explanations.append("Project execution is operating within acceptable tolerances of the planned schedule baseline.")

        return {
            "planned_progress_pct": planned_progress_pct,
            "actual_progress_pct": actual_progress_pct,
            "progress_variance_pct": progress_variance,
            "delay_probability": delay_prob,
            "is_delay_predicted": bool(delay_prob >= 0.50 or pred_days >= 3.0),
            "predicted_delay_days": pred_days,
            "planned_completion_date": planned_completion.strftime("%Y-%m-%d"),
            "predicted_completion_date": predicted_completion.strftime("%Y-%m-%d"),
            "model_confidence": round(float(np.max(probs)), 2),
            "model_type": "GradientBoosting Ensemble",
            "top_contributors": top_contributors,
            "explanations": explanations,
            "input_features": dict(zip(FEATURE_NAMES, input_vector)),
        }


delay_predictor = DelayPredictor()
