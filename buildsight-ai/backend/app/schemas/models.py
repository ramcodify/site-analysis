"""BuildSight AI — Complete Pydantic Schemas"""

from datetime import datetime
from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────

class ViolationStatus(str, Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class ViolationSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ZoneType(str, Enum):
    RESTRICTED = "RESTRICTED"
    HAZARD = "HAZARD"
    EQUIPMENT = "EQUIPMENT"
    EDGE = "EDGE"


class IdentityStatus(str, Enum):
    REGISTERED = "REGISTERED"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"


# ── AI Output Models ──────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PPEResult(BaseModel):
    worker_id: int
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    gloves: Optional[bool] = None
    boots: Optional[bool] = None
    ppe_compliance: float = 0.0
    model_available: bool = False


class ActivityResult(BaseModel):
    worker_id: int
    activity: str = "Unknown"
    confidence: float = 0.0
    is_unsafe: bool = False
    model_available: bool = False


class ProgressResult(BaseModel):
    current_stage: str = "Not Started"
    stage_confidence: float = 0.0
    stage_completion_percentage: float = 0.0
    overall_progress_percentage: float = 0.0
    progress_status: str = "ON_TRACK"
    is_model_prediction: bool = False


# ── Registered Worker Models (No Raw Biometrics Exposed) ───────────

class RegisteredWorkerBase(BaseModel):
    name: str
    employee_number: str
    department: str
    role: str
    active_status: str = "ACTIVE"


class RegisteredWorkerCreate(RegisteredWorkerBase):
    worker_code: Optional[str] = None  # Auto-generated if omitted e.g. W001


class RegisteredWorkerUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    active_status: Optional[str] = None


class RegisteredWorkerResponse(RegisteredWorkerBase):
    id: Union[int, str]
    worker_code: str
    profile_image_path: Optional[str] = None
    registration_date: Optional[Union[datetime, str]] = None
    created_at: Optional[Union[datetime, str]] = None
    updated_at: Optional[Union[datetime, str]] = None
    total_embeddings: int = 0
    # Note: biometric_embeddings is NEVER included in API responses


class RegisteredWorkerDetail(RegisteredWorkerResponse):
    is_currently_active: bool = False
    current_track_id: Optional[int] = None
    total_violations_count: int = 0
    avg_ppe_compliance: float = 0.0
    latest_risk_score: float = 0.0
    latest_risk_level: str = "SAFE"
    last_recognized: Optional[Union[datetime, str]] = None


class QualityCheckResult(BaseModel):
    is_valid: bool
    face_detected: bool
    face_count: int = 0
    score: float = 0.0
    sharpness_score: float = 0.0
    brightness_score: float = 0.0
    size_adequate: bool = False
    issues: List[str] = Field(default_factory=list)
    face_bbox: Optional[BoundingBox] = None


class UnknownPersonLinkRequest(BaseModel):
    track_id: int
    worker_code: str
    notes: Optional[str] = None


# ── Analytics WebSocket Message ───────────────────────────────────

class PerformanceData(BaseModel):
    capture_fps: float = 0.0
    inference_fps: float = 0.0
    latency_ms: float = 0.0


class RiskDistribution(BaseModel):
    safe: int = 0
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


class WorkersSummary(BaseModel):
    active_count: int = 0
    registered_count: int = 0
    unknown_count: int = 0
    risk_distribution: RiskDistribution = Field(default_factory=RiskDistribution)


class SafetySummary(BaseModel):
    ppe_compliance_percentage: float = 0.0
    active_violations: int = 0
    total_violations: int = 0


class ProgressData(BaseModel):
    current_stage: str = "Not Started"
    stage_confidence: float = 0.0
    stage_completion_percentage: float = 0.0
    overall_progress_percentage: float = 0.0
    progress_status: str = "ON_TRACK"


class TrackedWorker(BaseModel):
    # Temporary ByteTrack track ID
    worker_id: int
    temporary_track_id: int
    # Permanent Identity
    permanent_worker_id: Optional[str] = None  # e.g. "W001" or None
    worker_code: Optional[str] = None         # "W001"
    name: Optional[str] = None                # "John Doe" or "Unknown Worker"
    identity_status: str = "UNKNOWN"          # "REGISTERED" | "UNKNOWN" | "UNCERTAIN"
    recognition_confidence: Optional[float] = None
    face_bbox: Optional[BoundingBox] = None
    # Spatial & Detection
    bbox: BoundingBox
    confidence: float
    # PPE & Compliance
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    gloves: Optional[bool] = None
    face_mask: Optional[bool] = None
    missing_ppe: List[str] = Field(default_factory=list)
    compliance_status: str = "UNKNOWN"
    ppe_compliance: float = 0.0
    # Risk & Status
    risk_score: float = 0.0
    risk_level: str = "SAFE"
    risk_factors: List[str] = Field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    tracking_duration: float = 0.0
    violation_count: int = 0
    activity: str = "Unknown"
    activity_confidence: float = 0.0


class AnalyticsMessage(BaseModel):
    type: str = "analytics_update"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "webcam"
    performance: PerformanceData = Field(default_factory=PerformanceData)
    workers: WorkersSummary = Field(default_factory=WorkersSummary)
    safety: SafetySummary = Field(default_factory=SafetySummary)
    progress: ProgressData = Field(default_factory=ProgressData)
    tracked_workers: List[TrackedWorker] = Field(default_factory=list)
    model_status: dict = Field(default_factory=dict)


# ── API Request/Response Models ───────────────────────────────────

class ViolationUpdate(BaseModel):
    status: Optional[ViolationStatus] = None
    resolution_notes: Optional[str] = None


class RTSPSourceCreate(BaseModel):
    name: str
    rtsp_url: str
    fps: float = 5.0


class DangerZoneCreate(BaseModel):
    name: str
    source_id: Optional[str] = None
    zone_type: ZoneType = ZoneType.RESTRICTED
    polygon_data: List[List[float]]   # [[x1,y1],[x2,y2],...]
    risk_weight: float = Field(default=30.0, ge=0, le=100)


class StageProgressUpdate(BaseModel):
    stage_index: int
    completion: float = Field(ge=0.0, le=100.0)


# ── Response shapes ───────────────────────────────────────────────

class WorkerResponse(BaseModel):
    worker_id: int
    track_id: int
    permanent_worker_id: Optional[str] = None
    worker_code: Optional[str] = None
    name: Optional[str] = None
    identity_status: str = "UNKNOWN"
    recognition_confidence: Optional[float] = None
    source_id: Optional[str] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    tracking_duration: float = 0.0
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    gloves: Optional[bool] = None
    face_mask: Optional[bool] = None
    missing_ppe: List[str] = Field(default_factory=list)
    compliance_status: str = "UNKNOWN"
    ppe_compliance: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "SAFE"
    risk_factors: List[str] = Field(default_factory=list)
    violation_count: int = 0
    is_live: bool = False


class StageDetail(BaseModel):
    index: int
    name: str
    weight: float
    completion: float
    status: str   # "completed" | "current" | "pending"
