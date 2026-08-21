"""BuildSight AI — Face Recognition Service (Modular Local Implementation)

Implements high-accuracy local face detection, alignment, embedding extraction,
quality verification, and biometric matching using OpenCV YuNet + SFace.

No external cloud APIs are used. All biometric data is computed and cached locally.
"""

import os
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

from app.config import settings

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """
    Modular local Face Recognition Service:
      - FaceDetector: YuNet (ONNX) - 5-landmark face detector
      - FaceRecognizer: SFace (ONNX) - 128-dim deep face embedding model
      - Cosine similarity matching against registered biometric templates
    """

    def __init__(
        self,
        detector_path: Optional[str] = None,
        recognizer_path: Optional[str] = None,
        match_threshold: float = 0.58,
    ):
        self.detector_path = detector_path or settings.face_detection_model_path
        self.recognizer_path = recognizer_path or settings.face_recognition_model_path
        self.match_threshold = match_threshold

        self._detector: Optional[Any] = None
        self._recognizer: Optional[Any] = None
        self._loaded = False
        self._load_error = ""

        # In-memory registered templates:
        # dict: worker_code -> { "name": str, "embeddings": np.ndarray [N, 128], "employee_number": str }
        self._registered_cache: Dict[str, Dict[str, Any]] = {}

    def load(self) -> bool:
        """Initialize YuNet detector and SFace recognizer."""
        models_dir = Path(__file__).resolve().parents[2] / "data" / "models"
        det_path = Path(self.detector_path)
        if not det_path.is_absolute():
            det_path = Path(__file__).resolve().parents[2] / self.detector_path

        rec_path = Path(self.recognizer_path)
        if not rec_path.is_absolute():
            rec_path = Path(__file__).resolve().parents[2] / self.recognizer_path

        # If not found at specific path, look in models_dir
        if not det_path.exists():
            det_path = models_dir / "face_detection_yunet_2023mar.onnx"
        if not rec_path.exists():
            rec_path = models_dir / "face_recognition_sface_2021dec.onnx"

        # Auto-download from OpenCV Zoo if missing
        if not rec_path.exists():
            try:
                import urllib.request
                logger.info("⬇ Downloading SFace face recognition model from OpenCV Zoo...")
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
                models_dir.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, str(rec_path))
                logger.info(f"✓ SFace model downloaded: {rec_path}")
            except Exception as e:
                logger.warning(f"SFace auto-download note: {e}")

        if not det_path.exists() or not rec_path.exists():
            self._load_error = f"Face model files not found (det={det_path.exists()}, rec={rec_path.exists()})"
            logger.warning(self._load_error)
            return False

        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(det_path),
                config="",
                input_size=(320, 320),
                score_threshold=settings.face_min_confidence,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )

            self._recognizer = cv2.FaceRecognizerSF.create(
                model=str(rec_path),
                config="",
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )

            self._loaded = True
            logger.info("✓ Face recognition service loaded (YuNet + SFace)")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load face recognition models: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "detector": "YuNet (2023mar.onnx)" if self._loaded else None,
            "recognizer": "SFace (2021dec.onnx)" if self._loaded else None,
            "registered_workers_cached": len(self._registered_cache),
            **({"error": self._load_error} if not self._loaded else {}),
        }

    # ── Face Detection ───────────────────────────────────────────

    def detect_faces(self, image: np.ndarray, conf_threshold: float = 0.55) -> List[Dict[str, Any]]:
        """
        Detect faces in image.
        Returns list of dicts with keys:
          - bbox: (x, y, w, h)
          - score: float
          - landmarks: 5 points [(x,y), ...] (right eye, left eye, nose tip, right mouth, left mouth)
          - raw: raw 15-float array from YuNet
        """
        if not self._loaded or self._detector is None or image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        self._detector.setScoreThreshold(conf_threshold)

        try:
            _, faces = self._detector.detect(image)
            if faces is None or len(faces) == 0:
                return []

            results = []
            for face in faces:
                x, y, fw, fh = map(int, face[:4])
                score = float(face[14])
                # 5 landmarks
                landmarks = [(int(face[4 + 2 * i]), int(face[5 + 2 * i])) for i in range(5)]
                results.append({
                    "bbox": (max(0, x), max(0, y), fw, fh),
                    "score": score,
                    "landmarks": landmarks,
                    "raw": face,
                })
            return results
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return []

    # ── Face Alignment & Embedding Extraction ────────────────────

    def extract_embedding(
        self,
        image: np.ndarray,
        raw_face: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Align face and extract 128-dimensional L2-normalized feature embedding.
        If raw_face is omitted, runs face detection first to locate landmarks.
        """
        if not self._loaded or self._recognizer is None or image is None or image.size == 0:
            return None

        try:
            if raw_face is None:
                faces = self.detect_faces(image, conf_threshold=0.50)
                if not faces:
                    return None
                # Pick largest or highest score face
                raw_face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])["raw"]

            aligned_face = self._recognizer.alignCrop(image, raw_face)
            if aligned_face is None or aligned_face.size == 0:
                return None

            feature = self._recognizer.feature(aligned_face)
            if feature is None or feature.size == 0:
                return None

            # Flatten and normalize to unit vector
            feat_flat = feature.flatten().astype(np.float32)
            norm = np.linalg.norm(feat_flat)
            if norm > 1e-6:
                feat_flat = feat_flat / norm
            return feat_flat
        except Exception as e:
            logger.debug(f"Embedding extraction error: {e}")
            return None

    # ── Quality Verification for Registration ────────────────────

    def verify_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Verify face image quality for registration.
        Checks:
          - Face presence & count (strictly 1 face)
          - Face size (min 60x60 pixels)
          - Sharpness via Laplacian variance (> 35)
          - Brightness & illumination uniformity
        """
        issues = []
        if image is None or image.size == 0:
            return {
                "is_valid": False,
                "face_detected": False,
                "face_count": 0,
                "score": 0.0,
                "sharpness_score": 0.0,
                "brightness_score": 0.0,
                "size_adequate": False,
                "issues": ["No image data provided"],
                "face_bbox": None,
            }

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. Detect faces
        faces = self.detect_faces(image, conf_threshold=0.55)
        face_count = len(faces)

        if face_count == 0:
            issues.append("No face detected in image")
            return {
                "is_valid": False,
                "face_detected": False,
                "face_count": 0,
                "score": 0.0,
                "sharpness_score": 0.0,
                "brightness_score": 0.0,
                "size_adequate": False,
                "issues": issues,
                "face_bbox": None,
            }
        elif face_count > 1:
            issues.append(f"Multiple faces detected ({face_count}). Only 1 person allowed per photo.")

        best_face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        fx, fy, fw, fh = best_face["bbox"]

        # 2. Face Size
        size_adequate = (fw >= 50 and fh >= 50)
        if not size_adequate:
            issues.append(f"Face is too small ({fw}x{fh}px). Minimum required: 50x50px.")

        # 3. Sharpness Check (Laplacian Variance on face crop)
        face_crop_gray = gray[max(0, fy): min(h, fy + fh), max(0, fx): min(w, fx + fw)]
        if face_crop_gray.size > 0:
            sharpness = float(cv2.Laplacian(face_crop_gray, cv2.CV_64F).var())
        else:
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if sharpness < 30.0:
            issues.append(f"Image is too blurry (sharpness: {sharpness:.1f} < 30.0). Please hold still.")

        # 4. Brightness Check
        mean_brightness = float(np.mean(face_crop_gray if face_crop_gray.size > 0 else gray))
        if mean_brightness < 40.0:
            issues.append("Lighting is too dark. Increase ambient lighting.")
        elif mean_brightness > 225.0:
            issues.append("Lighting is overexposed. Reduce glare or bright background.")

        # Overall Score calculation (0.0 to 1.0)
        sharp_norm = min(1.0, sharpness / 150.0)
        bright_norm = 1.0 - abs(mean_brightness - 128.0) / 128.0
        size_norm = min(1.0, (fw * fh) / (120.0 * 120.0))
        conf_norm = best_face["score"]

        quality_score = float(0.35 * sharp_norm + 0.25 * bright_norm + 0.20 * size_norm + 0.20 * conf_norm)
        is_valid = len(issues) == 0 and quality_score >= 0.50

        return {
            "is_valid": is_valid,
            "face_detected": True,
            "face_count": face_count,
            "score": round(quality_score, 2),
            "sharpness_score": round(sharpness, 1),
            "brightness_score": round(mean_brightness, 1),
            "size_adequate": size_adequate,
            "issues": issues,
            "face_bbox": {
                "x1": float(fx),
                "y1": float(fy),
                "x2": float(fx + fw),
                "y2": float(fy + fh),
            },
        }

    # ── In-Memory Biometric Cache & Matching ──────────────────────

    def update_registered_cache(self, worker_code: str, name: str, employee_number: str, embeddings: List[List[float]]):
        """Update or register embeddings for a worker in cache."""
        if not embeddings:
            return
        arr = np.array(embeddings, dtype=np.float32)
        # Ensure row vectors are L2-normalized
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        arr = arr / norms

        self._registered_cache[worker_code] = {
            "name": name,
            "employee_number": employee_number,
            "embeddings": arr,
        }
        logger.info(f"Biometric cache updated for {worker_code} ({name}): {len(arr)} templates")

    def remove_from_cache(self, worker_code: str):
        self._registered_cache.pop(worker_code, None)

    def clear_cache(self):
        """Clear all registered biometric templates."""
        self._registered_cache.clear()
        logger.info("✓ Biometric cache cleared")

    def load_all_registered(self, workers_data: List[Dict[str, Any]]):
        """Load all registered workers from DB into memory cache."""
        self._registered_cache.clear()
        count = 0
        for w in workers_data:
            code = w.get("worker_code")
            name = w.get("name", "Worker")
            emp_no = w.get("employee_number", "")
            raw_embs = w.get("biometric_embeddings")
            if code and raw_embs:
                try:
                    if isinstance(raw_embs, str):
                        raw_embs = json.loads(raw_embs)
                    if isinstance(raw_embs, list) and len(raw_embs) > 0:
                        self.update_registered_cache(code, name, emp_no, raw_embs)
                        count += 1
                except Exception as e:
                    logger.error(f"Error loading biometric template for {code}: {e}")
        logger.info(f"✓ Loaded {count} registered workers into biometric cache")

    def match_worker(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Compare query embedding against all cached registered workers.
        Uses max cosine similarity across multi-sample registration templates.

        Returns:
          (worker_code, best_similarity, worker_name)
          If best_similarity < threshold:
          (None, best_similarity, None)
        """
        if query_embedding is None or len(self._registered_cache) == 0:
            return None, 0.0, None

        thresh = threshold if threshold is not None else self.match_threshold
        query_norm = query_embedding.flatten()
        norm_val = np.linalg.norm(query_norm)
        if norm_val > 1e-6:
            query_norm = query_norm / norm_val

        scores = []
        for code, info in self._registered_cache.items():
            ref_embs = info["embeddings"]  # [N, 128]
            # Matrix dot product against all templates for this worker
            sims = np.dot(ref_embs, query_norm)
            max_sim = float(np.max(sims))
            scores.append((max_sim, code, info["name"]))

        scores.sort(key=lambda s: s[0], reverse=True)
        best_similarity, best_worker_code, best_worker_name = scores[0]

        # Margin check if multiple candidates exist in database
        is_distinct = True
        if len(scores) > 1 and best_similarity < 0.62:
            second_sim = scores[1][0]
            if (best_similarity - second_sim) < 0.06:
                is_distinct = False  # Close similarity between two different registered profiles

        if best_similarity >= thresh and best_worker_code is not None and is_distinct:
            return best_worker_code, round(best_similarity, 3), best_worker_name
        else:
            return None, round(max(0.0, best_similarity), 3), None


# Global singleton instance
face_recognition_service = FaceRecognitionService()
