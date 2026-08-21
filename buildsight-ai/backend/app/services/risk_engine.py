"""BuildSight AI — Risk Engine (Phase 10)

Weighted rule-based risk scoring for tracked workers.
Score: 0-100, Risk Level: SAFE | LOW | MEDIUM | HIGH | CRITICAL
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

RISK_LEVELS = [
    (0,  20,  "SAFE"),
    (21, 40,  "LOW"),
    (41, 60,  "MEDIUM"),
    (61, 80,  "HIGH"),
    (81, 100, "CRITICAL"),
]

DEFAULT_WEIGHTS = {
    "no_helmet":      35.0,
    "no_vest":        20.0,
    "danger_zone":    30.0,
    "unsafe_activity": 25.0,
    "violation_count": 5.0,   # per violation (capped at 15)
}


class RiskEngine:
    """Weighted risk scoring engine."""

    def __init__(self):
        self.weights = dict(DEFAULT_WEIGHTS)

    def update_worker_risk(
        self,
        worker,
        in_danger_zone: bool = False,
        unsafe_activity: bool = False,
    ):
        """Compute and attach risk_score / risk_level / risk_factors to worker."""
        score = 0.0
        factors = []

        # Helmet
        if worker.helmet is False:
            score += self.weights["no_helmet"]
            factors.append("No helmet detected")
        # Vest
        if worker.vest is False:
            score += self.weights["no_vest"]
            factors.append("No safety vest detected")
        # Danger zone
        if in_danger_zone:
            score += self.weights["danger_zone"]
            factors.append("Worker in danger zone")
        # Unsafe activity
        if unsafe_activity:
            score += self.weights["unsafe_activity"]
            factors.append("Unsafe activity detected")
        # Violation count penalty (capped)
        if worker.violation_count > 0:
            penalty = min(worker.violation_count * self.weights["violation_count"], 15.0)
            score += penalty
            if worker.violation_count > 1:
                factors.append(f"{worker.violation_count} prior violations")

        score = min(100.0, max(0.0, score))
        worker.risk_score = round(score, 1)
        worker.risk_level = self._score_to_level(score)
        worker.risk_factors = factors

    def get_risk_distribution(self, workers) -> dict:
        dist = {"safe": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
        for w in workers:
            level = getattr(w, "risk_level", "SAFE").lower()
            if level in dist:
                dist[level] += 1
        return dist

    @staticmethod
    def _score_to_level(score: float) -> str:
        for lo, hi, level in RISK_LEVELS:
            if lo <= score <= hi:
                return level
        return "CRITICAL"
