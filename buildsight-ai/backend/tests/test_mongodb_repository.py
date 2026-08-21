"""BuildSight AI — MongoDB Comprehensive Test Suite

Verifies:
  1. MongoDB connection & health check
  2. Worker registration with multi-sample biometric embeddings
  3. Duplicate worker code & employee number rejection
  4. Worker lookup (biometric templates hidden from public responses)
  5. Face recognition identity mapping
  6. Temporary Track ID to permanent Worker ID mapping
  7. Violation creation & persistence
  8. Duplicate violation cooldown & update logic
  9. Worker PPE snapshots & historical compliance tracking
  10. Construction progress record creation & history retrieval
  11. Real-time Dashboard and Safety Analytics aggregations
  12. Invalid ObjectId validation and error handling
  13. Danger zone and Video Source persistence
"""

import unittest
import uuid
import numpy as np
from fastapi import HTTPException

from app.database.mongodb import get_db, init_db, is_mongo_connected
from app.database.utils import to_object_id, is_valid_object_id, serialize_mongo_doc
from app.database.repository import (
    RegisteredWorkerRepository,
    WorkerRepository,
    ViolationRepository,
    ProgressRepository,
    DangerZoneRepository,
    VideoSourceRepository,
    AnalyticsAggregationRepository,
)


class TestMongoDBArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.db = get_db()

    def test_01_mongodb_connection_and_health(self):
        """Test 1: MongoDB connection ping and health check status."""
        self.assertTrue(is_mongo_connected())
        db = get_db()
        pong = db.command("ping")
        self.assertEqual(pong.get("ok"), 1.0)

    def test_02_worker_registration(self):
        """Test 2: Register a worker with multi-sample 128-d biometric embeddings."""
        unique_emp = f"EMP-TEST-{uuid.uuid4().hex[:6]}"
        code = RegisteredWorkerRepository.get_next_worker_code()
        emb1 = [float(x) for x in np.random.randn(128)]
        emb2 = [float(x) for x in np.random.randn(128)]

        w = RegisteredWorkerRepository.create(
            name="John Site Lead",
            employee_number=unique_emp,
            department="Civil Engineering",
            role="Supervisor",
            embeddings=[emb1, emb2],
            worker_code=code,
            profile_image_path="/api/profiles/test.jpg"
        )
        self.assertEqual(w["worker_code"], code)
        self.assertEqual(w["name"], "John Site Lead")
        self.assertEqual(w["total_embeddings"], 2)

    def test_03_duplicate_worker_rejection(self):
        """Test 3: Reject duplicate worker code and employee number."""
        unique_emp = f"EMP-DUP-{uuid.uuid4().hex[:6]}"
        code = RegisteredWorkerRepository.get_next_worker_code()
        emb = [float(x) for x in np.random.randn(128)]

        RegisteredWorkerRepository.create(
            name="Original Worker",
            employee_number=unique_emp,
            department="Electrical",
            role="Technician",
            embeddings=[emb],
            worker_code=code,
        )

        # Attempt duplicate with same employee number
        with self.assertRaises(ValueError):
            RegisteredWorkerRepository.create(
                name="Duplicate Worker",
                employee_number=unique_emp,
                department="Electrical",
                role="Technician",
                embeddings=[emb],
            )

    def test_04_worker_lookup_and_biometric_isolation(self):
        """Test 4: Verify worker lookup never exposes raw 128-d embeddings to public API."""
        unique_emp = f"EMP-SEC-{uuid.uuid4().hex[:6]}"
        code = RegisteredWorkerRepository.get_next_worker_code()
        emb = [float(x) for x in np.random.randn(128)]

        RegisteredWorkerRepository.create(
            name="Secure Worker",
            employee_number=unique_emp,
            department="Safety",
            role="Officer",
            embeddings=[emb],
            worker_code=code,
        )

        w = RegisteredWorkerRepository.get_by_code(code)
        self.assertIsNotNone(w)
        self.assertNotIn("biometric_embeddings", w)
        self.assertEqual(w["total_embeddings"], 1)

    def test_05_face_recognition_biometric_cache_query(self):
        """Test 5: Verify backend startup cache method retrieves raw embeddings."""
        cache_data = RegisteredWorkerRepository.get_all_raw_for_biometric_cache()
        self.assertIsInstance(cache_data, list)
        for item in cache_data:
            self.assertIn("worker_code", item)
            self.assertIn("biometric_embeddings", item)

    def test_06_track_id_to_permanent_worker_mapping(self):
        """Test 6: Map temporary ByteTrack tracking session to permanent worker ID."""
        track_id = 8812
        w = WorkerRepository.upsert_worker(track_id=track_id, source_id="webcam_main", worker_code="W001")
        self.assertEqual(w["track_id"], track_id)
        self.assertEqual(w["worker_code"], "W001")
        self.assertEqual(w["identity_status"], "REGISTERED")

        # Update duration
        WorkerRepository.update_worker_duration(track_id, 35.0, worker_code="W001")
        detail = WorkerRepository.get_worker_detail(track_id)
        self.assertEqual(detail["tracking_duration"], 35.0)

    def test_07_violation_creation(self):
        """Test 7: Create and persist safety violations in MongoDB."""
        vid = f"viol-test-{uuid.uuid4().hex[:8]}"
        doc = {
            "violation_id": vid,
            "worker_id": 8812,
            "source_id": "webcam_main",
            "violation_type": "MISSING_HELMET",
            "severity": "HIGH",
            "risk_score": 75.0,
            "status": "OPEN",
            "duration_seconds": 1.5,
            "description": "Hardhat missing in active work area",
        }
        ViolationRepository.save_violation(doc, worker_code="W001")

        all_viols = ViolationRepository.get_all()
        found = next((v for v in all_viols if v["violation_id"] == vid), None)
        self.assertIsNotNone(found)
        self.assertEqual(found["violation_type"], "MISSING_HELMET")
        self.assertEqual(found["worker_code"], "W001")

    def test_08_duplicate_violation_prevention_and_cooldown(self):
        """Test 8: Ensure repeated violation updates existing record duration rather than duplicating."""
        vid = f"viol-cool-{uuid.uuid4().hex[:8]}"
        doc = {
            "violation_id": vid,
            "worker_id": 8812,
            "violation_type": "MISSING_SAFETY_VEST",
            "severity": "MEDIUM",
            "risk_score": 50.0,
            "status": "OPEN",
            "duration_seconds": 2.0,
        }
        ViolationRepository.save_violation(doc, worker_code="W001")

        # Save again with extended duration
        doc["duration_seconds"] = 5.5
        ViolationRepository.save_violation(doc, worker_code="W001")

        all_viols = [v for v in ViolationRepository.get_all() if v["violation_id"] == vid]
        self.assertEqual(len(all_viols), 1)
        self.assertEqual(all_viols[0]["duration_seconds"], 5.5)

    def test_09_worker_ppe_snapshots_and_history(self):
        """Test 9: Store and aggregate worker PPE snapshots in MongoDB."""
        WorkerRepository.add_snapshot(
            worker_id=8812,
            helmet=True,
            vest=True,
            risk_score=20.0,
            risk_level="LOW",
            bbox=(50, 50, 150, 250),
            worker_code="W001",
            activity="Inspecting",
            gloves=True,
            face_mask=True,
            ppe_compliance=100.0,
        )

        stats = RegisteredWorkerRepository.get_historical_stats("W001")
        self.assertIn("avg_ppe_compliance", stats)
        self.assertIn("lifetime_tracking_duration", stats)

    def test_10_progress_record_creation_and_history(self):
        """Test 10: Persist and retrieve construction progress records."""
        ProgressRepository.save({
            "source_id": "cctv_01",
            "current_stage": "Structural Framing",
            "stage_confidence": 0.92,
            "stage_completion": 65.0,
            "overall_progress": 48.0,
            "project_status": "ON_TRACK",
        })

        history = ProgressRepository.get_history(limit=5)
        self.assertTrue(len(history) > 0)
        self.assertEqual(history[0]["current_stage"], "Structural Framing")

    def test_11_dashboard_and_safety_aggregations(self):
        """Test 11: Execute MongoDB aggregation pipelines for Dashboard and Safety metrics."""
        dash = AnalyticsAggregationRepository.get_dashboard_metrics()
        self.assertIn("active_violations", dash)
        self.assertIn("risk_distribution", dash)
        self.assertIn("average_ppe_compliance", dash)

        safety = AnalyticsAggregationRepository.get_safety_analytics_data()
        self.assertIn("violation_types", safety)
        self.assertIn("severities", safety)

    def test_12_objectid_handling_and_validation(self):
        """Test 12: Validate MongoDB ObjectId conversions and error responses."""
        valid_hex = "507f1f77bcf86cd799439011"
        self.assertTrue(is_valid_object_id(valid_hex))
        oid = to_object_id(valid_hex)
        self.assertEqual(str(oid), valid_hex)

        # Invalid ObjectId
        invalid_hex = "not-a-valid-id"
        self.assertFalse(is_valid_object_id(invalid_hex))
        with self.assertRaises(HTTPException) as ctx:
            to_object_id(invalid_hex)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_13_danger_zones_and_video_sources(self):
        """Test 13: CRUD for danger zones and sanitized video sources."""
        zone_id = f"zone-{uuid.uuid4().hex[:6]}"
        DangerZoneRepository.save({
            "zone_id": zone_id,
            "name": "Crane Swing Radius",
            "zone_type": "RESTRICTED",
            "polygon_data": [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]],
            "risk_weight": 40.0,
        })
        zones = DangerZoneRepository.get_all()
        self.assertTrue(any(z["zone_id"] == zone_id for z in zones))

        # Video source with RTSP credentials sanitization
        src_id = f"src-{uuid.uuid4().hex[:6]}"
        VideoSourceRepository.save({
            "source_id": src_id,
            "name": "North Gate CCTV",
            "source_type": "RTSP",
            "configuration": {"rtsp_url": "rtsp://admin:secretPass123@192.168.1.50:554/h264"},
            "status": "ACTIVE",
        })
        sources = VideoSourceRepository.get_all()
        saved_src = next((s for s in sources if s["source_id"] == src_id), None)
        self.assertIsNotNone(saved_src)
        # Verify secret password is redacted
        self.assertNotIn("secretPass123", saved_src["configuration"]["rtsp_url"])


if __name__ == "__main__":
    unittest.main()
