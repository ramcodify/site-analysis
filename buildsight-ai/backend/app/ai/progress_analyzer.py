"""BuildSight AI — Research-Grade 9-Stage Construction Progress Analyzer

Tracks construction progress using a locally trained deep neural network classifier:
  1. Site Preparation (Weight: 5%)
  2. Excavation (Weight: 10%)
  3. Foundation (Weight: 15%)
  4. Structural Work (Weight: 20%)
  5. Brickwork (Weight: 15%)
  6. Roofing (Weight: 10%)
  7. Plastering (Weight: 10%)
  8. Electrical and Plumbing (Weight: 10%)
  9. Finishing (Weight: 5%)

Supports real CNN model inference, temporal softmax smoothing, stage confidence scoring,
probability distribution breakdown, and dynamic weighted completion estimation.
"""

import os
import logging
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

from app.schemas.models import ProgressResult

logger = logging.getLogger(__name__)

CONSTRUCTION_STAGES = [
    "Site Preparation",
    "Excavation",
    "Foundation",
    "Structural Work",
    "Brickwork",
    "Roofing",
    "Plastering",
    "Electrical and Plumbing",
    "Finishing",
]

STAGE_WEIGHTS = {
    "Site Preparation":        5.0,
    "Excavation":              10.0,
    "Foundation":              15.0,
    "Structural Work":         20.0,
    "Brickwork":               15.0,
    "Roofing":                 10.0,
    "Plastering":              10.0,
    "Electrical and Plumbing": 10.0,
    "Finishing":                5.0,
}


class ConstructionStageClassifier(nn.Module):
    """Convolutional neural network for 9-stage construction stage classification."""

    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)


class ProgressAnalyzer:
    """Construction progress tracker powered by a real local 9-stage CNN classifier."""

    def __init__(self, model_path: str = ""):
        if not model_path:
            default_path = Path(__file__).resolve().parents[2] / "data" / "models" / "progress_model.pth"
            if default_path.exists():
                model_path = str(default_path)

        self.model_path = model_path
        self._model = None
        self._loaded = False
        self._load_error = ""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tracking state
        self._current_stage_index: int = 0
        self._stage_completions: Dict[str, float] = {s: 0.0 for s in CONSTRUCTION_STAGES}
        self._stage_probabilities: Dict[str, float] = {s: (1.0 if i == 0 else 0.0) for i, s in enumerate(CONSTRUCTION_STAGES)}
        self._prediction_history: deque = deque(maxlen=10)

    def load(self) -> bool:
        if not self.model_path or not Path(self.model_path).exists():
            default_pth = Path(__file__).resolve().parents[2] / "data" / "models" / "progress_model.pth"
            if default_pth.exists():
                self.model_path = str(default_pth)
            else:
                self._load_error = "Progress model weights not found — manual tracking mode"
                logger.info(f"ℹ Progress analyzer: {self._load_error}")
                return False

        try:
            model = ConstructionStageClassifier(num_classes=len(CONSTRUCTION_STAGES))
            state_dict = torch.load(self.model_path, map_location=self._device)
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()
            self._model = model
            self._loaded = True
            logger.info(f"✓ Progress stage model loaded ({self._device}): {self.model_path}")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Progress model load failed: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "model": self.model_path if self._loaded else None,
            "mode": "ai" if self._loaded else "manual",
            "stages": len(CONSTRUCTION_STAGES),
            **({"error": self._load_error} if not self._loaded else {}),
        }

    def analyze(self, frame=None, context: Optional[dict] = None) -> ProgressResult:
        """Return construction progress via local CNN inference or manual baseline."""
        if self._loaded and frame is not None:
            return self._predict_with_model(frame)
        return self._manual_progress()

    def _predict_with_model(self, frame: np.ndarray) -> ProgressResult:
        try:
            # Preprocess frame (BGR -> RGB, resize 128x128, normalize)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb).resize((128, 128))
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Temporal smoothing over sliding window
            self._prediction_history.append(probs)
            smoothed_probs = np.mean(self._prediction_history, axis=0)

            stage_idx = int(np.argmax(smoothed_probs))
            confidence = float(smoothed_probs[stage_idx])

            # Update probability distribution
            for i, stage in enumerate(CONSTRUCTION_STAGES):
                self._stage_probabilities[stage] = round(float(smoothed_probs[i]), 3)

            stage_name = CONSTRUCTION_STAGES[min(stage_idx, len(CONSTRUCTION_STAGES) - 1)]
            self._current_stage_index = min(stage_idx, len(CONSTRUCTION_STAGES) - 1)

            # Auto-update stage completions based on predicted stage
            for i, s in enumerate(CONSTRUCTION_STAGES):
                if i < self._current_stage_index:
                    self._stage_completions[s] = 100.0
                elif i == self._current_stage_index:
                    self._stage_completions[s] = round(float(confidence * 100.0), 1)

            overall_pct = self._calc_overall()

            return ProgressResult(
                current_stage=stage_name,
                stage_confidence=round(confidence, 2),
                stage_completion_percentage=self._stage_completions.get(stage_name, 50.0),
                overall_progress_percentage=overall_pct,
                progress_status="ON_TRACK" if confidence >= 0.50 else "AHEAD",
                is_model_prediction=True,
            )
        except Exception as e:
            logger.error(f"Progress prediction error: {e}")
            return self._manual_progress()

    def predict_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify a single standalone image and return complete stage probability breakdown."""
        if not self._loaded or self._model is None:
            self.load()
        if not self._loaded or self._model is None:
            return {
                "success": False,
                "error": "Progress model not loaded",
                "predicted_stage": CONSTRUCTION_STAGES[self._current_stage_index],
                "confidence": 1.0,
                "probabilities": self._stage_probabilities,
                "overall_progress_percentage": self._calc_overall(),
                "stages": self.get_stage_details(),
            }

        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(image).resize((128, 128))
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            stage_idx = int(np.argmax(probs))
            confidence = float(probs[stage_idx])

            probs_dict = {
                stage: round(float(probs[i]), 4)
                for i, stage in enumerate(CONSTRUCTION_STAGES)
            }
            self._stage_probabilities = probs_dict
            stage_name = CONSTRUCTION_STAGES[stage_idx]
            self._current_stage_index = stage_idx

            for i, s in enumerate(CONSTRUCTION_STAGES):
                if i < stage_idx:
                    self._stage_completions[s] = 100.0
                elif i == stage_idx:
                    self._stage_completions[s] = round(confidence * 100.0, 1)

            return {
                "success": True,
                "predicted_stage": stage_name,
                "stage_index": stage_idx,
                "confidence": round(confidence, 4),
                "probabilities": probs_dict,
                "overall_progress_percentage": self._calc_overall(),
                "stages": self.get_stage_details(),
            }
        except Exception as e:
            logger.error(f"Image progress analysis error: {e}")
            return {"success": False, "error": str(e)}

    def _manual_progress(self) -> ProgressResult:
        stage_name = CONSTRUCTION_STAGES[self._current_stage_index]
        stage_completion = self._stage_completions.get(stage_name, 0.0)

        total_weight = sum(STAGE_WEIGHTS.values())
        weighted_progress = 0.0
        for i, s in enumerate(CONSTRUCTION_STAGES):
            if i < self._current_stage_index:
                weighted_progress += STAGE_WEIGHTS[s] / total_weight * 100
            elif i == self._current_stage_index:
                weighted_progress += STAGE_WEIGHTS[s] / total_weight * stage_completion

        return ProgressResult(
            current_stage=stage_name,
            stage_confidence=1.0,
            stage_completion_percentage=stage_completion,
            overall_progress_percentage=round(min(100.0, weighted_progress), 1),
            progress_status="ON_TRACK",
            is_model_prediction=False,
        )

    def _calc_overall(self) -> float:
        total_weight = sum(STAGE_WEIGHTS.values())
        weighted = 0.0
        for i, s in enumerate(CONSTRUCTION_STAGES):
            if i < self._current_stage_index:
                weighted += STAGE_WEIGHTS[s] / total_weight * 100
            elif i == self._current_stage_index:
                weighted += STAGE_WEIGHTS[s] / total_weight * self._stage_completions.get(s, 50.0)
        return round(min(100.0, weighted), 1)

    def set_current_stage(self, stage_index: int):
        """Manually set current construction stage."""
        if 0 <= stage_index < len(CONSTRUCTION_STAGES):
            for i in range(stage_index):
                self._stage_completions[CONSTRUCTION_STAGES[i]] = 100.0
            self._current_stage_index = stage_index
            logger.info(f"Stage set: {CONSTRUCTION_STAGES[stage_index]}")

    def set_stage_completion(self, stage_name: str, completion: float):
        """Set completion % for a specific stage."""
        if stage_name in self._stage_completions:
            self._stage_completions[stage_name] = max(0.0, min(100.0, completion))

    def get_stage_details(self) -> List[Dict[str, Any]]:
        """Return all 9 stages with their completion status and probability."""
        result = []
        for i, stage in enumerate(CONSTRUCTION_STAGES):
            if i < self._current_stage_index:
                status = "completed"
                completion = 100.0
            elif i == self._current_stage_index:
                status = "current"
                completion = self._stage_completions.get(stage, 50.0)
            else:
                status = "pending"
                completion = 0.0
            result.append({
                "index": i,
                "name": stage,
                "weight": STAGE_WEIGHTS[stage],
                "completion": completion,
                "status": status,
                "probability": self._stage_probabilities.get(stage, 0.0),
            })
        return result
