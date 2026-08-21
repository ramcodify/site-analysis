"""BuildSight AI — Multi-Class PPE Detection Module

Provides 5-class PPE compliance detection:
  - Class 0: Person
  - Class 1: Helmet / Hardhat
  - Class 2: Safety Vest
  - Class 3: Gloves
  - Class 4: Face Mask

Includes multi-model YOLO ensemble + spatial-anatomical body binding +
HSV visual color confirmation (fluorescent orange / neon yellow-green safety vests
and hardhat geometry) + temporal stability smoothing across video frames.
"""

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2 
import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["person", "helmet", "safety_vest", "gloves", "face_mask"]

# Spatial anatomical association guidelines (relative to worker bounding box height)
ANATOMICAL_REGIONS = {
    "helmet": {"y_min": -0.25, "y_max": 0.45, "x_pad": 0.30},
    "face_mask": {"y_min": 0.05, "y_max": 0.45, "x_pad": 0.25},
    "safety_vest": {"y_min": 0.12, "y_max": 0.85, "x_pad": 0.35},
    "gloves": {"y_min": 0.40, "y_max": 1.05, "x_pad": 0.45},
}


@dataclass
class PPEDetection:
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float


@dataclass
class DetailedPPEResult:
    worker_id: int
    helmet: Dict[str, Any]
    safety_vest: Dict[str, Any]
    gloves: Dict[str, Any]
    face_mask: Dict[str, Any]
    missing_ppe: List[str]
    compliance_status: str
    ppe_compliance: float
    model_available: bool = True


@dataclass
class WorkerPPEHistory:
    worker_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_update: float = 0.0


class PPEDetector:
    """Multi-Class PPE Detection & Anatomical Association Engine."""

    def __init__(
        self,
        model_path: str = "",
        confidence_threshold: float = 0.25,
        history_window_seconds: float = 2.0,
        missing_confirmation_ratio: float = 0.60,
        present_confirmation_ratio: float = 0.60,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.history_window_seconds = history_window_seconds
        self.missing_confirmation_ratio = missing_confirmation_ratio
        self.present_confirmation_ratio = present_confirmation_ratio

        self._model = None
        self._aux_model = None
        self._loaded = False
        self._load_error = ""
        self._history: Dict[int, WorkerPPEHistory] = {}

    def load(self) -> bool:
        """Load trained multi-class YOLO model and auxiliary hardhat detector."""
        models_dir = Path(__file__).resolve().parents[2] / "data" / "models"
        candidates = [
            Path(self.model_path) if self.model_path else None,
            models_dir / "ppe_model.pt",
            models_dir / "runs" / "ppe_yolo" / "weights" / "best.pt",
            models_dir / "Hansung-Cho_yolov8-ppe-detection.pt",
        ]

        selected_path = None
        for c in candidates:
            if c and c.exists() and c.is_file():
                selected_path = c
                break

        if selected_path is None:
            self._load_error = "No multi-class PPE model found in data/models/"
            logger.warning(self._load_error)
            return False

        try:
            # pyrefly: ignore [missing-import]
            from ultralytics import YOLO
            self._model = YOLO(str(selected_path))
            self._loaded = True
            self.model_path = str(selected_path)
            logger.info(f"✓ Multi-class PPE model loaded: {selected_path}")

            # Auxiliary Hardhat Model
            aux_path = models_dir / "best.pt"
            if aux_path.exists():
                try:
                    self._aux_model = YOLO(str(aux_path))
                    logger.info(f"✓ Auxiliary hardhat model loaded: {aux_path}")
                except Exception as e:
                    logger.debug(f"Aux model load note: {e}")

            return True
        except Exception as e:
            self._load_error = str(e)
            self._loaded = False
            logger.error(f"Failed to load PPE model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "model": self.model_path if self._loaded else None,
            "classes": CLASS_NAMES if self._loaded else [],
            **({"error": self._load_error} if not self._loaded else {}),
        }

    # ── Inference & Association ──────────────────────────────────

    def detect_raw_ppe(self, frame: np.ndarray) -> List[PPEDetection]:
        """Run YOLO models on frame to detect all raw visible PPE objects."""
        if not self._loaded or self._model is None or frame is None or frame.size == 0:
            return []

        detections: List[PPEDetection] = []
        try:
            # 1. Primary multi-class model (with explicit negative class discrimination)
            results = self._model(
                frame,
                conf=0.20,  # Low threshold to capture negative classes reliably
                verbose=False,
                device="cpu",
            )
            if results and len(results) > 0 and results[0].boxes is not None:
                names = self._model.names
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                    raw_cname = names.get(cls_id, "").lower()
                    cname = raw_cname.replace("-", "_").replace(" ", "_")

                    # Check for negative classes (e.g., no_helmet, no_hardhat, no_vest, no_mask)
                    is_negative = "no_" in cname or "no-" in raw_cname or "none" in cname

                    if is_negative:
                        if "helmet" in cname or "hardhat" in cname:
                            detections.append(PPEDetection(class_id=-1, class_name="no_helmet", bbox=(x1, y1, x2, y2), confidence=conf))
                        elif "vest" in cname:
                            detections.append(PPEDetection(class_id=-2, class_name="no_safety_vest", bbox=(x1, y1, x2, y2), confidence=conf))
                        elif "glove" in cname:
                            detections.append(PPEDetection(class_id=-3, class_name="no_gloves", bbox=(x1, y1, x2, y2), confidence=conf))
                        elif "mask" in cname:
                            detections.append(PPEDetection(class_id=-4, class_name="no_face_mask", bbox=(x1, y1, x2, y2), confidence=conf))
                        continue

                    # Positive detections with balanced confidence gating
                    std_name = None
                    std_id = None

                    if ("helmet" in cname or "hardhat" in cname) and conf >= 0.30:
                        std_name = "helmet"
                        std_id = 1
                    elif ("vest" in cname or "jacket" in cname) and conf >= 0.30:
                        std_name = "safety_vest"
                        std_id = 2
                    elif "glove" in cname and conf >= 0.30:
                        std_name = "gloves"
                        std_id = 3
                    elif "mask" in cname and conf >= 0.30:
                        std_name = "face_mask"
                        std_id = 4
                    elif ("person" in cname or "human" in cname) and conf >= 0.35:
                        std_name = "person"
                        std_id = 0

                    if std_name is not None and std_id is not None:
                        detections.append(PPEDetection(
                            class_id=std_id,
                            class_name=std_name,
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                        ))

            # 2. Auxiliary model — Enriches PPE coverage on distant and dark video streams
            if self._aux_model is not None:
                try:
                    aux_res = self._aux_model(
                        frame,
                        conf=0.30,
                        verbose=False,
                        device="cpu",
                    )
                    if aux_res and len(aux_res) > 0 and aux_res[0].boxes is not None:
                        for box in aux_res[0].boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            cname = self._aux_model.names.get(cls_id, "").lower()
                            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                            std_name = None
                            std_id = None
                            if ("helmet" in cname or "hardhat" in cname) and conf >= 0.30:
                                std_name = "helmet"
                                std_id = 1
                            elif ("vest" in cname or "jacket" in cname) and conf >= 0.30:
                                std_name = "safety_vest"
                                std_id = 2
                            elif "glove" in cname and "no" not in cname and conf >= 0.30:
                                std_name = "gloves"
                                std_id = 3
                            elif "mask" in cname and "no" not in cname and conf >= 0.30:
                                std_name = "face_mask"
                                std_id = 4

                            if std_name is not None and std_id is not None:
                                detections.append(PPEDetection(
                                    class_id=std_id,
                                    class_name=std_name,
                                    bbox=(x1, y1, x2, y2),
                                    confidence=conf,
                                ))
                except Exception as e:
                    logger.debug(f"Aux detection note: {e}")

            return detections
        except Exception as e:
            logger.debug(f"Raw PPE detection error: {e}")
            return []

    def associate_ppe_to_worker(
        self,
        worker_bbox: Tuple[float, float, float, float],
        ppe_detections: List[PPEDetection],
        frame_shape: Tuple[int, int],
        frame: Optional[np.ndarray] = None,
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Anatomical spatial association with strict negative class suppression:
        Associates neural network PPE object detections to the worker
        and actively suppresses false positives when kerchiefs, bandanas, soft caps, or cloth are tied on the head.
        """
        wx1, wy1, wx2, wy2 = [float(v) for v in worker_bbox]
        ww = max(1.0, wx2 - wx1)
        wh = max(1.0, wy2 - wy1)

        results = {
            "helmet": {"detected": False, "confidence": 0.0},
            "safety_vest": {"detected": False, "confidence": 0.0},
            "gloves": {"detected": False, "confidence": 0.0},
            "face_mask": {"detected": False, "confidence": 0.0},
        }

        negatives = {
            "no_helmet": 0.0,
            "no_safety_vest": 0.0,
            "no_gloves": 0.0,
            "no_face_mask": 0.0,
        }

        if ww <= 0 or wh <= 0:
            return results

        wcx = (wx1 + wx2) / 2.0
        candidate_helmet_bbox = None

        for det in ppe_detections:
            dx1, dy1, dx2, dy2 = det.bbox
            dcx = (dx1 + dx2) / 2.0
            dcy = (dy1 + dy2) / 2.0

            # Handle negative detections (e.g. no_helmet on head region)
            if det.class_name in negatives:
                h_dist_ratio = abs(dcx - wcx) / max(1.0, ww)
                if h_dist_ratio <= 0.65:
                    negatives[det.class_name] = max(negatives[det.class_name], det.confidence)
                continue

            item = det.class_name
            if item not in ANATOMICAL_REGIONS:
                continue

            region = ANATOMICAL_REGIONS[item]
            exp_y1 = wy1 + region["y_min"] * wh
            exp_y2 = wy1 + region["y_max"] * wh

            max_h_offset = {
                "helmet": 0.45,
                "face_mask": 0.45,
                "safety_vest": 0.60,
                "gloves": 0.85,
            }.get(item, 0.50)

            h_dist_ratio = abs(dcx - wcx) / max(1.0, ww)
            is_vertically_aligned = (exp_y1 <= dcy <= exp_y2)
            is_horizontally_aligned = h_dist_ratio <= max_h_offset

            if is_vertically_aligned and is_horizontally_aligned:
                if det.confidence > results[item]["confidence"]:
                    results[item]["detected"] = True
                    results[item]["confidence"] = round(det.confidence, 2)
                    if item == "helmet":
                        candidate_helmet_bbox = (dx1, dy1, dx2, dy2)
                    elif item == "safety_vest":
                        candidate_vest_bbox = (dx1, dy1, dx2, dy2)

        # ── Strict Rigid Hardhat vs Baseball Cap / Kerchief / Soft Cap Verification ──
        if results["helmet"]["detected"] and candidate_helmet_bbox is not None:
            hx1, hy1, hx2, hy2 = candidate_helmet_bbox
            helmet_w = max(1.0, hx2 - hx1)
            helmet_h = max(1.0, hy2 - hy1)

            # 1. Aspect ratio check (industrial hardhats are dome-shaped)
            aspect = helmet_w / helmet_h
            if aspect < 0.50 or aspect > 3.2:
                results["helmet"]["detected"] = False
                results["helmet"]["confidence"] = 0.0

            # 2. Geometric clearance relative to face (if frontal face detected)
            if results["helmet"]["detected"] and face_bbox is not None:
                fx1, fy1, fx2, fy2 = face_bbox
                face_w = max(1.0, fx2 - fx1)
                face_h = max(1.0, fy2 - fy1)

                rise_above_face = fy1 - hy1
                # A true rigid industrial hardhat rises above the brow
                # A tied cloth/kerchief sits directly flush against the forehead
                if rise_above_face < 0.10 * face_h:
                    logger.info(f"🚫 Cap/Cloth detected on worker: low crown height ({rise_above_face:.1f}px < {0.10*face_h:.1f}px)")
                    results["helmet"]["detected"] = False
                    results["helmet"]["confidence"] = 0.0

            # 3. Visual Color, Reflectance, and Dark Cap Suppression
            if results["helmet"]["detected"] and frame is not None and frame.size > 0 and not np.all(frame == 0):
                fh, fw = frame.shape[:2]
                ix1 = max(0, min(fw - 1, int(hx1)))
                iy1 = max(0, min(fh - 1, int(hy1)))
                ix2 = max(0, min(fw, int(hx2)))
                iy2 = max(0, min(fh, int(hy2)))

                if (ix2 - ix1) > 10 and (iy2 - iy1) > 10:
                    crop = frame[iy1:iy2, ix1:ix2]
                    try:
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        h, s, v = cv2.split(hsv)
                        avg_v = float(np.mean(v))
                        avg_s = float(np.mean(s))
                        avg_h = float(np.mean(h))

                        # Dark baseball caps (black, dark navy, dark grey, dark brown)
                        if avg_v < 40:
                            logger.info(f"🚫 Dark Cap detected on worker: brightness too low for safety hardhat (V={avg_v:.1f})")
                            results["helmet"]["detected"] = False
                            results["helmet"]["confidence"] = 0.0
                        else:
                            # Certified safety hardhat standard colors (illumination-invariant):
                            is_yellow = (14 <= avg_h <= 48) and (avg_s >= 25) and (avg_v >= 45)
                            is_orange = (4 <= avg_h <= 20) and (avg_s >= 35) and (avg_v >= 45)
                            is_blue = (90 <= avg_h <= 140) and (avg_s >= 25) and (avg_v >= 40)
                            is_red = ((avg_h <= 10) or (avg_h >= 160)) and (avg_s >= 35) and (avg_v >= 40)
                            is_white = (avg_s <= 55) and (avg_v >= 70)
                            is_green = (42 <= avg_h <= 88) and (avg_s >= 25) and (avg_v >= 45)

                            if not (is_yellow or is_orange or is_blue or is_red or is_white or is_green):
                                if avg_v < 60 or (avg_s > 60 and (20 <= avg_h <= 90)):
                                    results["helmet"]["detected"] = False
                                    results["helmet"]["confidence"] = 0.0
                    except Exception:
                        pass

        # ── High-Visibility Safety Vest Illumination-Invariant Verification ──
        if results["safety_vest"]["detected"] and candidate_vest_bbox is not None and frame is not None and frame.size > 0 and not np.all(frame == 0):
            vx1, vy1, vx2, vy2 = candidate_vest_bbox
            fh, fw = frame.shape[:2]
            ix1 = max(0, min(fw - 1, int(vx1)))
            iy1 = max(0, min(fh - 1, int(vy1)))
            ix2 = max(0, min(fw, int(vx2)))
            iy2 = max(0, min(fh, int(vy2)))

            if (ix2 - ix1) > 15 and (iy2 - iy1) > 15:
                vcrop = frame[iy1:iy2, ix1:ix2]
                try:
                    vhsv = cv2.cvtColor(vcrop, cv2.COLOR_BGR2HSV)
                    vh, vs, vv = cv2.split(vhsv)
                    
                    # High-Vis Lime/Yellow & Safety Orange in varied ambient lighting
                    yellow_mask = (vh >= 24) & (vh <= 88) & (vs >= 25) & (vv >= 35)
                    orange_mask = (vh >= 4) & (vh <= 26) & (vs >= 30) & (vv >= 35)
                    hivis_pixels = np.count_nonzero(yellow_mask | orange_mask)
                    total_pixels = max(1, (ix2 - ix1) * (iy2 - iy1))
                    hivis_ratio = hivis_pixels / total_pixels

                    if hivis_ratio < 0.04 and results["safety_vest"]["confidence"] < 0.45:
                        logger.info(f"🚫 Casual clothing detected (non-vest): fluorescent ratio {hivis_ratio:.1%}")
                        results["safety_vest"]["detected"] = False
                        results["safety_vest"]["confidence"] = 0.0
                except Exception:
                    pass

        # Negative suppression: cancel if NO-Hardhat or NO-Safety Vest is explicitly detected with high confidence
        if negatives["no_helmet"] >= 0.40 and negatives["no_helmet"] > results["helmet"]["confidence"]:
            results["helmet"]["detected"] = False
            results["helmet"]["confidence"] = 0.0

        if negatives["no_safety_vest"] >= 0.40 and negatives["no_safety_vest"] > results["safety_vest"]["confidence"]:
            results["safety_vest"]["detected"] = False
            results["safety_vest"]["confidence"] = 0.0

        return results

    def detect_worker_ppe(
        self,
        frame: np.ndarray,
        worker_bbox: Tuple[float, float, float, float],
        worker_id: int,
        precomputed_ppe_detections: Optional[List[PPEDetection]] = None,
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> DetailedPPEResult:
        """
        Complete per-worker PPE compliance evaluation with temporal smoothing.
        """
        fh, fw = frame.shape[:2]

        if precomputed_ppe_detections is not None:
            raw_detections = precomputed_ppe_detections
        else:
            raw_detections = self.detect_raw_ppe(frame)

        # 1. Spatial anatomical association with visual confirmation and face geometric grounding
        current_frame_ppe = self.associate_ppe_to_worker(
            worker_bbox, raw_detections, (fh, fw), frame=frame, face_bbox=face_bbox
        )

        # 2. Temporal stability smoothing
        smoothed_ppe = self._apply_temporal_smoothing(worker_id, current_frame_ppe)

        # 3. Determine Missing PPE & Compliance Status
        missing_ppe = []
        required_items = ["helmet", "safety_vest", "gloves", "face_mask"]
        present_count = 0

        for item in required_items:
            if not smoothed_ppe[item]["detected"]:
                missing_ppe.append(item)
            else:
                present_count += 1

        ppe_percentage = (present_count / len(required_items)) * 100.0
        compliance_status = "FULLY_COMPLIANT" if len(missing_ppe) == 0 else "NON_COMPLIANT"

        return DetailedPPEResult(
            worker_id=worker_id,
            helmet=smoothed_ppe["helmet"],
            safety_vest=smoothed_ppe["safety_vest"],
            gloves=smoothed_ppe["gloves"],
            face_mask=smoothed_ppe["face_mask"],
            missing_ppe=missing_ppe,
            compliance_status=compliance_status,
            ppe_compliance=round(ppe_percentage, 1),
            model_available=self._loaded,
        )

    def _apply_temporal_smoothing(
        self,
        worker_id: int,
        raw_ppe: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Temporal voting across recent frames to eliminate flicker."""
        now = time.time()
        if worker_id not in self._history:
            self._history[worker_id] = WorkerPPEHistory(worker_id=worker_id)

        rec = self._history[worker_id]
        rec.history.append((raw_ppe, now))
        rec.last_update = now

        valid_frames = [item for item in rec.history if (now - item[1]) <= self.history_window_seconds]
        if not valid_frames:
            valid_frames = list(rec.history)[-1:]

        smoothed = {}
        for key in ["helmet", "safety_vest", "gloves", "face_mask"]:
            present_votes = sum(1 for f in valid_frames if f[0][key]["detected"])
            total_votes = len(valid_frames)
            ratio = present_votes / max(1, total_votes)

            is_present = ratio >= self.present_confirmation_ratio
            best_conf = max((f[0][key]["confidence"] for f in valid_frames if f[0][key]["detected"]), default=0.0)

            smoothed[key] = {
                "detected": is_present,
                "confidence": round(best_conf if is_present else (1.0 - ratio), 2),
            }

        return smoothed

    def detect(self, frame: np.ndarray, bbox: tuple, worker_id: int = 0):
        """Legacy compatibility wrapper."""
        res = self.detect_worker_ppe(frame, bbox, worker_id)
        class LegacyWrapper:
            helmet = res.helmet["detected"]
            vest = res.safety_vest["detected"]
            gloves = res.gloves["detected"]
            boots = None
            ppe_compliance = res.ppe_compliance
            missing_ppe = res.missing_ppe
            compliance_status = res.compliance_status
            detailed = res
        return LegacyWrapper()


# Global singleton
ppe_detector = PPEDetector()
