"""BuildSight AI — YOLO11 Worker Detection Module

Uses Ultralytics YOLO for real-time person/worker detection.
Filters detections to person class only (COCO class 0).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WorkerDetector:
    """YOLO-based worker/person detector.

    Loads a YOLO model and runs inference on frames,
    returning only person-class detections.
    """

    # COCO class 0 = person
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.5,
        iou: float = 0.45,
        device: str = "cpu",
        input_size: int = 640,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.input_size = input_size
        self.model = None
        self._loaded = False
        self._load_error: Optional[str] = None

    def load(self) -> bool:
        """Load the YOLO model. Returns True on success."""
        try:
            # pyrefly: ignore [missing-import]
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self.model_path} on device: {self.device}")
            self.model = YOLO(self.model_path)
            # Warm up with a dummy frame
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, device=self.device)
            self._loaded = True
            logger.info(f"✓ YOLO model loaded successfully: {self.model_path}")
            return True
        except Exception as e:
            self._load_error = str(e)
            self._loaded = False
            logger.error(f"✗ Failed to load YOLO model '{self.model_path}': {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> dict:
        if self._loaded:
            return {"loaded": True, "model": self.model_path, "device": self.device}
        return {"loaded": False, "error": self._load_error or "Not initialized"}

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run detection on a frame. Returns list of person detections."""
        if not self._loaded or self.model is None:
            return []

        try:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.input_size,
                device=self.device,
                classes=[self.PERSON_CLASS_ID],  # Only detect persons
                verbose=False,
            )

            detections = []
            timestamp = datetime.now(timezone.utc)

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    detections.append(DetectionResult(
                        class_id=cls_id,
                        class_name="person",
                        confidence=conf,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        timestamp=timestamp,
                    ))

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def detect_with_tracking_input(self, frame: np.ndarray):
        """Run detection and return raw YOLO results for ByteTrack integration.

        Returns the ultralytics Results object which contains tracking-ready data.
        """
        if not self._loaded or self.model is None:
            return None

        try:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.input_size,
                device=self.device,
                classes=[self.PERSON_CLASS_ID],
                verbose=False,
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None
