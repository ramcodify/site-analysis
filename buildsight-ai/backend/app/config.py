"""BuildSight AI — Configuration (Pydantic Settings)"""

import os
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "BuildSight AI"
    debug: bool = False

    # Database (MongoDB)
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "buildsight_ai"
    database_url: str = "mongodb://localhost:27017/buildsight_ai"

    # AI Models
    yolo_model_path: str = "yolo11n.pt"
    ppe_model_path: str = ""
    activity_model_path: str = ""
    progress_model_path: str = ""
    face_detection_model_path: str = "data/models/face_detection_yunet_2023mar.onnx"
    face_recognition_model_path: str = "data/models/face_recognition_sface_2021dec.onnx"
    profile_images_dir: str = "data/profiles"
    evidence_dir: str = "data/evidence"

    # Face Recognition & Identity Matching
    face_match_threshold: float = 0.50
    face_confirmation_frames: int = 2
    face_history_window: int = 10
    face_min_confidence: float = 0.50

    # Detection
    detection_confidence: float = 0.50
    detection_iou: float = 0.45
    model_input_size: int = 640
    use_cuda: bool = False

    # Tracking
    track_buffer: int = 30
    match_threshold: float = 0.8

    # Processing & Persistence Intervals
    default_processing_fps: int = 10
    max_processing_queue: int = 2
    worker_snapshot_interval_seconds: float = 5.0
    progress_persist_interval_seconds: float = 30.0

    # Compliance
    violation_cooldown_seconds: int = 30

    # CORS
    cors_origins: str = "*"

    @property
    def device(self) -> str:
        if self.use_cuda:
            try:
                # pyrefly: ignore [missing-import]
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"

    model_config = {"env_file": ".env", "extra": "ignore", "frozen": False}


settings = Settings()
