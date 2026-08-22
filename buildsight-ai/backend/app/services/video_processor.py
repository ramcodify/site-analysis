
"""BuildSight AI — Complete Video Processing Pipeline

Handles webcam frames, RTSP streams, and uploaded videos through the
full AI pipeline:
  YOLO11 Person Detection
       ↓
  ByteTrack (Temporary Track ID)
       ↓
  Face Detection & Alignment (YuNet)
       ↓
  Face Recognition & Biometric Matching (SFace)
       ↓
  Temporal Identity Confirmation (Permanent Worker ID: W001, W002...)
       ↓
  PPE Detection (Helmet & Vest)
       ↓
  Activity & Danger Zone Analysis
       ↓
  Risk Engine & Compliance Tracking
       ↓
  DB Persistence & Real-time WebSocket Broadcast
"""

from pathlib import Path
import os
import uuid
import asyncio
import base64
import time
import threading
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Optional
import logging

from app.config import settings
from app.ai.worker_tracker import WorkerTracker, TrackedWorkerState
from app.ai.face_recognition_service import face_recognition_service
from app.services.identity_manager import identity_manager
from app.ai.ppe_detector import PPEDetector
from app.ai.progress_analyzer import ProgressAnalyzer
from app.ai.activity_analyzer import ActivityAnalyzer
from app.services.compliance_engine import ComplianceEngine
from app.services.risk_engine import RiskEngine
from app.services.danger_zone_service import DangerZoneService
from app.services.websocket_manager import ws_manager
from app.database.repository import (
    RegisteredWorkerRepository, WorkerRepository, ViolationRepository, ProgressRepository
)
from app.schemas.models import (
    AnalyticsMessage, PerformanceData, WorkersSummary,
    RiskDistribution, SafetySummary, ProgressData, TrackedWorker, BoundingBox,
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Core video processing pipeline with permanent worker identity recognition."""

    def __init__(self):
        self.tracker = WorkerTracker(
            model_path=settings.yolo_model_path,
            confidence=settings.detection_confidence,
            iou=settings.detection_iou,
            device=settings.device,
            input_size=settings.model_input_size,
            track_buffer=settings.track_buffer,
            match_threshold=settings.match_threshold,
        )
        self.face_service = face_recognition_service
        self.identity_mgr = identity_manager
        self.ppe_detector = PPEDetector(model_path=settings.ppe_model_path)
        self.progress_analyzer = ProgressAnalyzer(model_path=settings.progress_model_path)
        self.activity_analyzer = ActivityAnalyzer(model_path=settings.activity_model_path)
        self.compliance_engine = ComplianceEngine()
        self.risk_engine = RiskEngine()
        self.danger_zone_service = DangerZoneService()

        # Frame queue (webcam/upload)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=settings.max_processing_queue)
        self._processing = False
        self._source: str = "none"
        self._process_task: Optional[asyncio.Task] = None

        # RTSP threading
        self._rtsp_thread: Optional[threading.Thread] = None
        self._rtsp_active = False
        self._rtsp_url: str = ""

        # Video file processing
        self._video_file_path: Optional[str] = None
        self._video_paused = False

        # Performance metrics
        self._fps_window_start: float = time.time()
        self._fps_frame_count: int = 0
        self._current_inference_fps: float = 0.0
        self._current_latency_ms: float = 0.0

        # DB snapshot throttle (save every N seconds)
        self._last_db_save: float = 0.0
        self._db_save_interval: float = getattr(settings, "worker_snapshot_interval_seconds", 5.0)

        # Progress record throttle
        self._last_progress_save: float = 0.0
        self._progress_save_interval: float = getattr(settings, "progress_persist_interval_seconds", 30.0)

    # ── Initialization ───────────────────────────────────────────

    def initialize(self) -> dict:
        status = {}
        # 1. YOLO Tracker
        self.tracker.load()
        status["yolo_tracker"] = self.tracker.status

        # 2. Face Recognition & Biometric Cache
        self.face_service.load()
        try:
            raw_workers = RegisteredWorkerRepository.get_all_raw_for_biometric_cache()
            self.face_service.load_all_registered(raw_workers)
        except Exception as e:
            logger.warning(f"Could not load registered workers from DB into cache: {e}")
        status["face_recognition"] = self.face_service.status

        # 3. PPE Detector
        self.ppe_detector.load()
        status["ppe_detector"] = self.ppe_detector.status

        # 4. Progress Analyzer
        self.progress_analyzer.load()
        status["progress_analyzer"] = self.progress_analyzer.status

        # 5. Activity Analyzer
        self.activity_analyzer.load()
        status["activity_analyzer"] = self.activity_analyzer.status

        # 6. Danger Zones
        try:
            from app.database.repository import DangerZoneRepository
            active_zones = DangerZoneRepository.get_active_zones()
            for z in active_zones:
                self.danger_zone_service.add_zone(z)
            logger.info(f"✓ Loaded {len(active_zones)} active danger zones into memory")
        except Exception as e:
            logger.debug(f"Could not load danger zones from DB: {e}")

        # 7. InternVL3 Scene Understanding
        try:
            from app.ai.scene_understanding import scene_understanding
            scene_understanding.load()
            status["scene_understanding"] = scene_understanding.status
        except Exception as e:
            logger.debug(f"Scene understanding load error: {e}")

        return status

    @property
    def model_status(self) -> dict:
        status_dict = {
            "yolo_tracker": self.tracker.status,
            "face_recognition": self.face_service.status,
            "ppe_detector": self.ppe_detector.status,
            "progress_analyzer": self.progress_analyzer.status,
            "activity_analyzer": self.activity_analyzer.status,
        }
        try:
            from app.ai.scene_understanding import scene_understanding
            status_dict["scene_understanding"] = scene_understanding.status
        except Exception:
            pass
        return status_dict

    @property
    def is_processing(self) -> bool:
        return self._processing

    # ── Webcam / Frame WebSocket Processing ─────────────────────

    async def start_processing(self, source: str = "webcam"):
        if self._processing:
            return
        self._source = source
        self._processing = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"Video processing started: {source}")

    async def stop_processing(self):
        self._processing = False
        self._rtsp_active = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.identity_mgr.reset()
        logger.info("Video processing stopped")

    async def submit_frame(self, frame_data: str, capture_fps: float = 0.0):
        if not self._processing:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait((frame_data, capture_fps, time.time()))
        except asyncio.QueueFull:
            pass

    async def _process_loop(self):
        while self._processing:
            try:
                frame_data, capture_fps, submit_time = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue
            try:
                start_time = time.time()
                frame = self._decode_frame(frame_data)
                if frame is None:
                    continue
                loop = asyncio.get_event_loop()
                tracked_workers = await loop.run_in_executor(
                    None, self._run_pipeline, frame
                )
                inference_time = time.time() - start_time
                self._update_performance(inference_time)
                analytics = self._build_analytics(tracked_workers, capture_fps)
                await ws_manager.broadcast_json(analytics.model_dump())
            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)

    # ── RTSP Stream ─────────────────────────────────────────────

    async def start_rtsp(self, rtsp_url: str, source_name: str = "rtsp"):
        if self._rtsp_active:
            await self.stop_rtsp()
        self._rtsp_url = rtsp_url
        self._source = source_name
        self._rtsp_active = True
        self._processing = True
        self._process_task = asyncio.create_task(self._process_loop())
        self._rtsp_thread = threading.Thread(
            target=self._rtsp_capture_thread,
            daemon=True,
        )
        self._rtsp_thread.start()
        logger.info(f"RTSP stream started: {rtsp_url}")

    async def stop_rtsp(self):
        self._rtsp_active = False
        await self.stop_processing()

    def _rtsp_capture_thread(self):
        cap = cv2.VideoCapture(self._rtsp_url)
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {self._rtsp_url}")
            return
        frame_interval = 1.0 / settings.default_processing_fps
        while self._rtsp_active:
            ret, frame = cap.read()
            if not ret:
                logger.warning("RTSP stream lost, reconnecting...")
                time.sleep(2)
                cap = cv2.VideoCapture(self._rtsp_url)
                continue
            encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]
            b64 = base64.b64encode(encoded.tobytes()).decode()
            asyncio.run_coroutine_threadsafe(
                self.submit_frame(b64), asyncio.get_event_loop()
            )
            time.sleep(frame_interval)
        cap.release()

    # ── Video File Processing ────────────────────────────────────

    async def start_video_file(self, file_path: str):
        self._video_file_path = file_path
        self._video_paused = False
        self._source = "upload"
        self._processing = True
        self._process_task = asyncio.create_task(self._process_loop())
        video_task = asyncio.create_task(self._video_file_loop(file_path))
        logger.info(f"Video file processing started: {file_path}")

    async def pause_video(self):
        self._video_paused = True

    async def resume_video(self):
        self._video_paused = False

    async def _video_file_loop(self, file_path: str):
        loop = asyncio.get_event_loop()
        cap = await loop.run_in_executor(None, cv2.VideoCapture, file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = 1.0 / min(fps, settings.default_processing_fps)

        while self._processing:
            if self._video_paused:
                await asyncio.sleep(0.1)
                continue
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                logger.info("Video file processing complete")
                await self.stop_processing()
                break
            encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]
            b64 = base64.b64encode(encoded.tobytes()).decode()
            await self.submit_frame(b64)
            await asyncio.sleep(frame_interval)
        cap.release()

    # ── Core AI Pipeline ─────────────────────────────────────────

    def _decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        try:
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return None

    def _run_pipeline(self, frame: np.ndarray) -> list[TrackedWorkerState]:
        """Full AI pipeline on a single frame."""
        now = time.time()
        fh, fw = frame.shape[:2]

        # 1. YOLO + ByteTrack
        tracked = self.tracker.track(frame)
        active_track_ids = {w.worker_id for w in tracked}
        self.identity_mgr.cleanup_stale_tracks(active_track_ids)

        # 1b. Raw multi-class PPE detection on full frame
        raw_ppe_detections = self.ppe_detector.detect_raw_ppe(frame) if self.ppe_detector.is_loaded else []

        # 2. Per-worker analysis
        for worker in tracked:
            # ── 2a. Face Detection & Permanent Identity Recognition ──
            x1, y1, x2, y2 = [int(v) for v in worker.bbox]
            bw = x2 - x1
            bh = y2 - y1

            # Head region: upper 45% of worker bbox with padding
            hy1 = max(0, y1 - int(bh * 0.08))
            hy2 = min(fh, y1 + int(bh * 0.45))
            hx1 = max(0, x1 - int(bw * 0.10))
            hx2 = min(fw, x2 + int(bw * 0.10))

            head_crop = frame[hy1:hy2, hx1:hx2]
            face_result = None
            abs_face_bbox = None
            raw_face_data = None

            if self.face_service.is_loaded and head_crop.size > 0:
                detected_faces = self.face_service.detect_faces(head_crop, conf_threshold=settings.face_min_confidence)
                if not detected_faces and bh > 20:
                    # Fallback: expand region down to upper body if hardhat was blocking upper head crop
                    upper_body = frame[y1:min(fh, y1 + int(bh * 0.65)), hx1:hx2]
                    if upper_body.size > 0:
                        detected_faces = self.face_service.detect_faces(upper_body, conf_threshold=settings.face_min_confidence)
                        if detected_faces:
                            best_f = max(detected_faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
                            raw_face_data = best_f["raw"]
                            fx, fy, fbw, fbh = best_f["bbox"]
                            abs_face_bbox = (float(hx1 + fx), float(y1 + fy), float(hx1 + fx + fbw), float(y1 + fy + fbh))
                            head_crop = upper_body
                if detected_faces and raw_face_data is None:
                    # Pick largest detected face in head region
                    best_f = max(detected_faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
                    raw_face_data = best_f["raw"]
                    fx, fy, fbw, fbh = best_f["bbox"]
                    # Map face coordinates back to full frame space
                    abs_face_bbox = (float(hx1 + fx), float(hy1 + fy), float(hx1 + fx + fbw), float(hy1 + fy + fbh))

            # Update identity manager (temporal voting, occlusion handling)
            id_info = self.identity_mgr.update_track_face(
                track_id=worker.worker_id,
                face_crop_or_image=head_crop if raw_face_data is not None else None,
                raw_face_data=raw_face_data,
                face_bbox=abs_face_bbox,
            )

            worker.permanent_worker_id = id_info["permanent_worker_id"]
            worker.worker_code = id_info["worker_code"]
            worker.name = id_info["name"]
            worker.identity_status = id_info["identity_status"]
            worker.recognition_confidence = id_info["recognition_confidence"]
            worker.face_bbox = id_info["face_bbox"]

            # Live face / head / upper-body optical photo capture (Guaranteed for ALL workers including 100% SAFE)
            crop_source = head_crop if (head_crop is not None and head_crop.size > 0) else frame[max(0, y1):min(fh, y1 + max(30, int(bh * 0.55))), max(0, x1):min(fw, x2)]
            if crop_source is not None and crop_source.size > 0:
                now_t = time.time()
                last_crop_t = getattr(worker, "_last_crop_time", 0.0)
                if not worker.face_crop_base64 or (now_t - last_crop_t > 2.0):
                    try:
                        ch, cw = crop_source.shape[:2]
                        if ch > 10 and cw > 10:
                            scaled_crop = cv2.resize(crop_source, (160, int(160 * ch / max(1, cw))))
                            _, enc_thumb = cv2.imencode(".jpg", scaled_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            worker.face_crop_base64 = f"data:image/jpeg;base64,{base64.b64encode(enc_thumb.tobytes()).decode()}"
                            
                            # Save physical JPEG image file for safe worker photo history
                            snap_fn = f"worker_{worker.worker_id}.jpg"
                            snap_disk_path = os.path.join("data", "snapshots", snap_fn)
                            os.makedirs(os.path.dirname(snap_disk_path), exist_ok=True)
                            cv2.imwrite(snap_disk_path, scaled_crop)
                            if not worker.photo_url:
                                worker.photo_url = f"/data/snapshots/{snap_fn}"
                            
                            worker._last_crop_time = now_t
                    except Exception as e:
                        logger.debug(f"Worker crop save error: {e}")

            # ── 2b. PPE detection (Helmet + Vest + Gloves + Mask) with anatomical association & rigid dome verification ──
            if self.ppe_detector.is_loaded:
                ppe_res = self.ppe_detector.detect_worker_ppe(
                    frame=frame,
                    worker_bbox=worker.bbox,
                    worker_id=worker.worker_id,
                    precomputed_ppe_detections=raw_ppe_detections,
                    face_bbox=worker.face_bbox,
                )
                worker.helmet = ppe_res.helmet["detected"]
                worker.vest = ppe_res.safety_vest["detected"]
                worker.gloves = ppe_res.gloves["detected"]
                worker.face_mask = ppe_res.face_mask["detected"]
                worker.missing_ppe = ppe_res.missing_ppe
                worker.compliance_status = ppe_res.compliance_status
                worker.ppe_compliance = ppe_res.ppe_compliance

            # ── 2c. Activity analysis ──
            self.activity_analyzer.update_worker(
                worker.worker_id, worker.bbox, frame_wh=(fw, fh)
            )
            act = self.activity_analyzer.analyze_worker(
                worker.worker_id, frame_wh=(fw, fh)
            )
            worker.activity = act.activity
            worker.activity_confidence = act.confidence
            unsafe_activity = act.is_unsafe

            # ── 2d. Danger zone check ──
            in_zone = self.danger_zone_service.check_worker_in_zone(
                worker.bbox, fw, fh
            )

            # ── 2e. Risk scoring ──
            self.risk_engine.update_worker_risk(
                worker,
                in_danger_zone=bool(in_zone),
                unsafe_activity=unsafe_activity,
            )

        # 3. Compliance engine & state-transition persistence
        for worker in tracked:
            violations = list(self.compliance_engine.analyze_worker(
                worker, source_id=self._source
            ))
            
            in_zone = self.danger_zone_service.check_worker_in_zone(
                worker.bbox, fw, fh
            )
            if in_zone:
                dz_viols = self.compliance_engine.add_danger_zone_violation(
                    worker, in_zone, source_id=self._source
                )
                if dz_viols:
                    violations.extend(dz_viols)

            worker_code = worker.worker_code
            worker_id = worker.worker_id

            if violations:
                worker.violation_count += len(violations)
                
                # Crop worker tracker photo as evidence (compressed & pruned to protect disk storage)
                evidence_rel_path = None
                b64_evidence = None
                try:
                    x1, y1, x2, y2 = [int(v) for v in worker.bbox]
                    pad_x = int(max(10, (x2 - x1) * 0.15))
                    pad_y = int(max(10, (y2 - y1) * 0.15))
                    crop_x1 = max(0, x1 - pad_x)
                    crop_y1 = max(0, y1 - pad_y)
                    crop_x2 = min(fw, x2 + pad_x)
                    crop_y2 = min(fh, y2 + pad_y)

                    crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop_img is not None and crop_img.size > 0:
                        # Resize large evidence crops to max 400px width for storage efficiency
                        if crop_img.shape[1] > 400:
                            scale = 400.0 / crop_img.shape[1]
                            crop_img = cv2.resize(crop_img, (400, max(1, int(crop_img.shape[0] * scale))), interpolation=cv2.INTER_AREA)

                        os.makedirs(settings.evidence_dir, exist_ok=True)
                        _, buf = cv2.imencode(".jpg", crop_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        b64_evidence = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                except Exception as crop_err:
                    logger.debug(f"Evidence crop error: {crop_err}")

                for viol in violations:
                    vid = viol.get("violation_id", str(uuid.uuid4()))
                    viol["violation_id"] = vid
                    
                    if b64_evidence and crop_img is not None and crop_img.size > 0:
                        ev_filename = f"violation_{vid}.jpg"
                        ev_disk_path = os.path.join(settings.evidence_dir, ev_filename)
                        try:
                            cv2.imwrite(ev_disk_path, crop_img)
                            viol["evidence_path"] = f"/data/evidence/{ev_filename}"
                            viol["evidence_url"] = f"/data/evidence/{ev_filename}"
                            # Prune storage so disk never fills up
                            self._prune_evidence_storage(max_files=150)
                        except Exception:
                            pass
                        viol["snapshot_base64"] = b64_evidence
                    elif worker.face_crop_base64 or worker.photo_url:
                        viol["snapshot_base64"] = worker.face_crop_base64
                        viol["evidence_path"] = worker.photo_url
                        viol["evidence_url"] = worker.photo_url

                    try:
                        # Close previous open violations for this worker in MongoDB
                        ViolationRepository.resolve_worker_violations(
                            worker_code=worker_code,
                            worker_id=worker_id,
                            except_violation_id=vid
                        )
                        # Upsert worker session
                        WorkerRepository.upsert_worker(
                            worker_id, self._source, worker_code=worker_code
                        )
                        # Save the new violation event once with worker photo attached
                        ViolationRepository.save_violation(
                            viol,
                            worker_db_id=worker_id,
                            worker_code=worker_code,
                        )
                    except Exception as e:
                        logger.debug(f"DB save violation error: {e}")
            elif worker.helmet is True and worker.vest is True:
                # Worker is fully compliant — resolve any open violations in MongoDB
                try:
                    ViolationRepository.resolve_worker_violations(
                        worker_code=worker_code,
                        worker_id=worker_id
                    )
                except Exception as e:
                    logger.debug(f"DB resolve violations error: {e}")

        # 4. Progress Analysis on Live Video Frame
        try:
            self._current_progress = self.progress_analyzer.analyze(frame=frame)
        except Exception as e:
            logger.debug(f"Progress frame analysis error: {e}")

        # 5. Throttled State-Driven DB persistence
        if (now - self._last_db_save) > self._db_save_interval:
            self._last_db_save = now
            self._save_workers_to_db(tracked)

        # 6. Throttled progress record
        if (now - self._last_progress_save) > self._progress_save_interval:
            self._last_progress_save = now
            self._save_progress_to_db()

        return tracked

    def _save_workers_to_db(self, tracked: list[TrackedWorkerState]):
        now = time.time()
        for worker in tracked:
            try:
                WorkerRepository.upsert_worker(
                    worker.worker_id, self._source, worker_code=worker.worker_code, photo_url=worker.photo_url
                )
                WorkerRepository.update_worker_duration(
                    worker.worker_id, worker.tracking_duration, worker_code=worker.worker_code, photo_url=worker.photo_url
                )
                
                # Check if state changed before inserting snapshot
                worker_key = worker.worker_code or f"track_{worker.worker_id}"
                current_state = (
                    worker.helmet,
                    worker.vest,
                    getattr(worker, "gloves", None),
                    getattr(worker, "face_mask", None),
                    worker.risk_level
                )
                last = getattr(self, "_last_snapshot_states", {}).get(worker_key)
                
                # Only insert snapshot if state changed or 60s elapsed
                if not last or last["state"] != current_state or (now - last["time"]) >= 60.0:
                    WorkerRepository.add_snapshot(
                        worker.worker_id,
                        worker.helmet,
                        worker.vest,
                        worker.risk_score,
                        worker.risk_level,
                        worker.bbox,
                        worker_code=worker.worker_code,
                        activity=worker.activity,
                        gloves=worker.gloves,
                        face_mask=worker.face_mask,
                        photo_url=worker.photo_url,
                    )
                    if not hasattr(self, "_last_snapshot_states"):
                        self._last_snapshot_states = {}
                    self._last_snapshot_states[worker_key] = {
                        "state": current_state,
                        "time": now
                    }
            except Exception as e:
                logger.debug(f"DB save worker error: {e}")

    def _save_progress_to_db(self):
        try:
            result = getattr(self, "_current_progress", None) or self.progress_analyzer.analyze()
            ProgressRepository.save({
                "source_id": self._source,
                "current_stage": result.current_stage,
                "stage_confidence": result.stage_confidence,
                "stage_completion": result.stage_completion_percentage,
                "overall_progress": result.overall_progress_percentage,
                "project_status": result.progress_status,
            })
        except Exception as e:
            logger.debug(f"DB save progress error: {e}")

    # ── Performance ──────────────────────────────────────────────

    def _update_performance(self, inference_time: float):
        self._fps_frame_count += 1
        self._current_latency_ms = inference_time * 1000
        now = time.time()
        elapsed = now - self._fps_window_start
        if elapsed >= 2.0:
            self._current_inference_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_window_start = now

    # ── Analytics Message Builder ────────────────────────────────

    def _build_analytics(
        self, tracked_workers: list[TrackedWorkerState], capture_fps: float
    ) -> AnalyticsMessage:
        risk_dist = self.risk_engine.get_risk_distribution(tracked_workers)

        # PPE compliance from real detections only
        ppe_checked = [w for w in tracked_workers if w.helmet is not None or w.vest is not None]
        if ppe_checked:
            total_items = compliant_items = 0
            for w in ppe_checked:
                if w.helmet is not None:
                    total_items += 1
                    compliant_items += int(bool(w.helmet))
                if w.vest is not None:
                    total_items += 1
                    compliant_items += int(bool(w.vest))
            ppe_compliance = (compliant_items / total_items * 100) if total_items > 0 else 0.0
        else:
            ppe_compliance = 0.0

        progress_result = getattr(self, "_current_progress", None) or self.progress_analyzer.analyze()

        # Active violations from memory (fast) + DB count
        active_violations = self.compliance_engine.active_violation_count

        registered_count = sum(1 for w in tracked_workers if w.identity_status == "REGISTERED")
        unknown_count = len(tracked_workers) - registered_count

        worker_list = [
            TrackedWorker(
                worker_id=w.worker_id,
                temporary_track_id=w.worker_id,
                permanent_worker_id=w.permanent_worker_id,
                worker_code=w.worker_code,
                name=w.name,
                identity_status=w.identity_status,
                recognition_confidence=w.recognition_confidence,
                face_bbox=BoundingBox(
                    x1=w.face_bbox[0], y1=w.face_bbox[1], x2=w.face_bbox[2], y2=w.face_bbox[3]
                ) if w.face_bbox else None,
                bbox=BoundingBox(x1=w.bbox[0], y1=w.bbox[1], x2=w.bbox[2], y2=w.bbox[3]),
                confidence=w.confidence,
                helmet=w.helmet,
                vest=w.vest,
                ppe_compliance=w.ppe_compliance,
                risk_score=w.risk_score,
                risk_level=w.risk_level,
                risk_factors=w.risk_factors,
                first_seen=w.first_seen,
                last_seen=w.last_seen,
                tracking_duration=w.tracking_duration,
                violation_count=w.violation_count,
                activity=w.activity,
                activity_confidence=w.activity_confidence,
            )
            for w in tracked_workers
        ]

        return AnalyticsMessage(
            type="analytics_update",
            timestamp=datetime.now(timezone.utc),
            source=self._source,
            performance=PerformanceData(
                capture_fps=round(capture_fps, 1),
                inference_fps=round(self._current_inference_fps, 1),
                latency_ms=round(self._current_latency_ms, 1),
            ),
            workers=WorkersSummary(
                active_count=len(tracked_workers),
                registered_count=registered_count,
                unknown_count=unknown_count,
                risk_distribution=RiskDistribution(**risk_dist),
            ),
            safety=SafetySummary(
                ppe_compliance_percentage=round(ppe_compliance, 1),
                active_violations=active_violations,
                total_violations=self.compliance_engine.total_violation_count,
            ),
            progress=ProgressData(
                current_stage=progress_result.current_stage,
                stage_confidence=progress_result.stage_confidence,
                stage_completion_percentage=progress_result.stage_completion_percentage,
                overall_progress_percentage=progress_result.overall_progress_percentage,
                progress_status=progress_result.progress_status,
            ),
            tracked_workers=worker_list,
            model_status=self.model_status,
        )

    # ── Dashboard / API helpers ───────────────────────────────────

    def get_dashboard_data(self) -> dict:
        workers = self.tracker.get_all_workers()
        risk_dist = self.risk_engine.get_risk_distribution(workers)
        progress = self.progress_analyzer.analyze()
        registered_count = sum(1 for w in workers if w.identity_status == "REGISTERED")

        # Query DB state to ensure dashboard displays real persistent project telemetry
        overall_progress = progress.overall_progress_percentage
        current_stage = progress.current_stage
        total_registered_in_db = 0
        active_viols_count = self.compliance_engine.active_violation_count

        try:
            from app.database.mongodb import get_db
            from app.database.collections import COLLECTION_PROGRESS_RECORDS, COLLECTION_REGISTERED_WORKERS, COLLECTION_VIOLATIONS
            # pyrefly: ignore [missing-import]
            import pymongo
            db = get_db()
            total_registered_in_db = db[COLLECTION_REGISTERED_WORKERS].count_documents({"active_status": "ACTIVE"})
            if active_viols_count == 0:
                active_viols_count = db[COLLECTION_VIOLATIONS].count_documents({"status": "OPEN"})
            if overall_progress == 0.0 or current_stage in ["Site Preparation", "Not Started"]:
                latest_p = db[COLLECTION_PROGRESS_RECORDS].find_one({}, sort=[("timestamp", pymongo.DESCENDING)])
                if latest_p:
                    overall_progress = latest_p.get("overall_progress_percentage", overall_progress)
                    current_stage = latest_p.get("current_stage", current_stage)
        except Exception:
            pass

        ppe_comp = self._calc_ppe_compliance(workers) if workers else 0.0

        return {
            "active_workers": len(workers),
            "registered_workers": max(registered_count, total_registered_in_db if not workers else registered_count),
            "unknown_workers": max(0, len(workers) - registered_count),
            "ppe_compliance": ppe_comp,
            "active_violations": active_viols_count,
            "high_risk_workers": risk_dist.get("high", 0) + risk_dist.get("critical", 0),
            "current_stage": current_stage,
            "overall_progress": overall_progress,
            "inference_fps": round(self._current_inference_fps, 1),
            "latency_ms": round(self._current_latency_ms, 1),
            "risk_distribution": risk_dist if any(risk_dist.values()) else {"safe": total_registered_in_db, "low": 0, "medium": 0, "high": 0, "critical": 0},
            "model_status": self.model_status,
        }

    def _prune_evidence_storage(self, max_files: int = 150):
        """Automatically keep disk storage clean by pruning oldest evidence files."""
        try:
            ev_dir = Path(settings.evidence_dir)
            if not ev_dir.exists():
                return
            files = sorted(ev_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
            if len(files) > max_files:
                to_delete = files[: len(files) - max_files]
                for f in to_delete:
                    try:
                        f.unlink()
                    except Exception:
                        pass
                logger.info(f"🧹 Pruned {len(to_delete)} old violation evidence images to protect storage")
        except Exception as e:
            logger.debug(f"Evidence prune note: {e}")

    def reset_live_tracking(self):
        """Clears all active in-memory trackers, identity mappings, and compliance episodes."""
        self.tracker.reset()
        self.identity_mgr.reset()
        self.compliance_engine.clear()
        logger.info("✓ Live tracking and tracker state reset to empty")

    def remove_worker_track(self, track_id: int):
        """Removes a specific worker track from in-memory tracker and identity state."""
        try:
            if hasattr(self.tracker, 'remove_track'):
                self.tracker.remove_track(track_id)
            if hasattr(self.identity_mgr, 'unbind_track'):
                self.identity_mgr.unbind_track(track_id)
            logger.info(f"✓ Removed live tracking state for worker #{track_id}")
        except Exception as e:
            logger.debug(f"Error removing track #{track_id}: {e}")

    @staticmethod
    def _calc_ppe_compliance(workers: list[TrackedWorkerState]) -> float:
        checked = [w for w in workers if w.helmet is not None or w.vest is not None]
        if not checked:
            return 0.0
        total = compliant = 0
        for w in checked:
            if w.helmet is not None:
                total += 1
                compliant += int(bool(w.helmet))
            if w.vest is not None:
                total += 1
                compliant += int(bool(w.vest))
        return round(compliant / total * 100, 1) if total else 0.0


# Global singleton
video_processor = VideoProcessor()
