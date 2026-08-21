/**
 * BuildSight AI — TypeScript Type Definitions (Complete with Permanent Identity)
 */

// ── Connection / Processing ───────────────────────────────────────

export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';
export type IdentityStatus = 'REGISTERED' | 'UNKNOWN' | 'UNCERTAIN';

// ── AI Analytics (WebSocket) ──────────────────────────────────────

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TrackedWorker {
  worker_id: number;
  temporary_track_id: number;
  permanent_worker_id?: string | null; // e.g. "W001"
  worker_code?: string | null;        // "W001"
  name?: string | null;               // "Alice Smith" or "Unknown Worker"
  identity_status: IdentityStatus;
  recognition_confidence?: number | null;
  face_bbox?: BoundingBox | null;
  bbox: BoundingBox;
  confidence: number;
  helmet: boolean | null;
  vest: boolean | null;
  gloves: boolean | null;
  face_mask: boolean | null;
  missing_ppe?: string[];
  compliance_status?: 'FULLY_COMPLIANT' | 'NON_COMPLIANT' | 'UNKNOWN';
  ppe_compliance: number;
  risk_score: number;
  risk_level: 'SAFE' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  risk_factors: string[];
  first_seen: string | null;
  last_seen: string | null;
  tracking_duration: number;
  violation_count: number;
  activity?: string;
  activity_confidence?: number;
}

export interface RiskDistribution {
  safe: number;
  low: number;
  medium: number;
  high: number;
  critical: number;
}

export interface WorkersSummary {
  active_count: number;
  registered_count?: number;
  unknown_count?: number;
  risk_distribution: RiskDistribution;
}

export interface PerformanceData {
  capture_fps: number;
  inference_fps: number;
  latency_ms: number;
}

export interface SafetySummary {
  ppe_compliance_percentage: number;
  active_violations: number;
  total_violations: number;
}

export interface ProgressData {
  current_stage: string;
  stage_confidence: number;
  stage_completion_percentage: number;
  overall_progress_percentage: number;
  progress_status: 'ON_TRACK' | 'AHEAD' | 'DELAYED';
}

export interface ModelStatusEntry {
  loaded: boolean;
  model?: string | null;
  detector?: string | null;
  recognizer?: string | null;
  device?: string;
  active_workers?: number;
  registered_workers_cached?: number;
  error?: string;
  mode?: string;
  classes?: string[];
}

export type ModelStatus = Record<string, ModelStatusEntry>;

export interface AnalyticsMessage {
  type: string;
  timestamp: string;
  source: string;
  performance: PerformanceData;
  workers: WorkersSummary;
  safety: SafetySummary;
  progress: ProgressData;
  tracked_workers: TrackedWorker[];
  model_status: ModelStatus;
}

// ── Registered Worker Models ──────────────────────────────────────

export interface RegisteredWorker {
  id: number;
  worker_code: string;
  name: string;
  employee_number: string;
  department: string;
  role: string;
  profile_image_path?: string | null;
  registration_date?: string | null;
  active_status: 'ACTIVE' | 'INACTIVE';
  created_at?: string | null;
  updated_at?: string | null;
  total_embeddings?: number;
}

export interface RegisteredWorkerDetail extends RegisteredWorker {
  is_currently_active: boolean;
  current_track_id?: number | null;
  total_violations_count: number;
  lifetime_tracking_duration: number;
  avg_ppe_compliance: number;
  latest_risk_score: number;
  latest_risk_level: 'SAFE' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  last_recognized?: string | null;
}

export interface QualityCheckResult {
  is_valid: boolean;
  face_detected: boolean;
  face_count: number;
  score: number;
  sharpness_score: number;
  brightness_score: number;
  size_adequate: boolean;
  issues: string[];
  face_bbox?: BoundingBox | null;
}

// ── REST API Response Types ───────────────────────────────────────

export interface WorkerResponse {
  worker_id: number;
  track_id: number;
  permanent_worker_id?: string | null;
  worker_code?: string | null;
  name?: string | null;
  identity_status?: IdentityStatus;
  recognition_confidence?: number | null;
  source_id: string | null;
  first_seen: string | null;
  last_seen: string | null;
  tracking_duration: number;
  helmet: boolean | null;
  vest: boolean | null;
  ppe_compliance: number;
  risk_score: number;
  risk_level: 'SAFE' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  risk_factors: string[];
  violation_count: number;
  photo_url?: string | null;
  face_crop_base64?: string | null;
  is_live: boolean;
}

export interface WorkerDetail extends WorkerResponse {
  violations: ViolationResponse[];
}

export interface ViolationResponse {
  id: number | string;
  violation_id: string;
  worker_id: number;
  permanent_worker_id?: string | null;
  worker_code?: string | null;
  worker_name?: string | null;
  employee_number?: string | null;
  source_id: string | null;
  violation_type: string;
  missing_items?: string[];
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  risk_score: number;
  status: 'OPEN' | 'ACKNOWLEDGED' | 'RESOLVED';
  timestamp: string | null;
  resolved_at?: string | null;
  duration_seconds: number;
  evidence_path: string | null;
  evidence_url?: string | null;
  snapshot_base64?: string | null;
  description: string | null;
}

export interface StageDetail {
  index: number;
  name: string;
  weight: number;
  completion: number;
  status: 'completed' | 'current' | 'pending';
  probability?: number;
}

export interface ProgressResponse {
  current_stage: string;
  stage_confidence: number;
  stage_completion_percentage: number;
  overall_progress_percentage: number;
  progress_status: 'ON_TRACK' | 'AHEAD' | 'DELAYED';
  is_model_prediction: boolean;
  stages: StageDetail[];
}

export interface ProgressHistoryEntry {
  id: number;
  timestamp: string;
  source_id: string | null;
  current_stage: string;
  stage_confidence: number;
  stage_completion: number;
  overall_progress: number;
  project_status: string;
}

export interface DangerZone {
  id: number;
  name: string;
  source_id: string | null;
  zone_type: 'RESTRICTED' | 'HAZARD' | 'EQUIPMENT' | 'EDGE';
  polygon_data: [number, number][];
  risk_weight: number;
  is_active: boolean;
}

// ── UI Settings ───────────────────────────────────────────────────

export interface OverlaySettings {
  showBoundingBoxes: boolean;
  showWorkerIds: boolean;
  showPPEStatus: boolean;
  showRiskLabels: boolean;
  showConfidence: boolean;
  showFaceBoxes?: boolean;
}
