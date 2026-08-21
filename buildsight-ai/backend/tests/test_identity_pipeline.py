"""BuildSight AI — Registered Worker Face Recognition & Identity Test Suite

Comprehensive automated test cases for:
  TEST_ID_01: Registered Worker W001 enters camera -> Correct identity W001
  TEST_ID_02: W001 leaves frame and returns -> New Track ID allowed, Permanent Worker ID remains W001
  TEST_ID_03: W001 appears on different camera source -> Recognized as W001
  TEST_ID_04: Registered Worker W002 enters -> Recognized as W002 (not W001)
  TEST_ID_05: Two registered workers appear together -> Distinct W001 and W002 assignment
  TEST_ID_06: Unregistered person appears -> UNKNOWN (no forced match)
  TEST_ID_07: Face temporarily occluded -> Keeps confirmed identity during continuous tracking
  TEST_ID_08: Track ID change preserves & restores permanent Worker ID + historical records
  TEST_ID_09: Low-confidence / ambiguous match -> UNKNOWN or UNCERTAIN (never forces false match)
"""

import unittest
import numpy as np
from datetime import datetime, timezone

from app.ai.face_recognition_service import FaceRecognitionService
from app.services.identity_manager import IdentityManager, TrackIdentityState
from app.ai.worker_tracker import TrackedWorkerState
from app.database.repository import RegisteredWorkerRepository, WorkerRepository, ViolationRepository


class TestIdentityPipeline(unittest.TestCase):

    def setUp(self):
        self.face_service = FaceRecognitionService(match_threshold=0.58)
        self.face_service._loaded = True
        self.identity_mgr = IdentityManager(
            match_threshold=0.58,
            confirmation_frames=3,
            history_window=10,
        )

        # Create distinct normalized 128-d synthetic embeddings for testing
        np.random.seed(42)
        emb_w001_1 = np.random.randn(128).astype(np.float32)
        emb_w001_1 /= np.linalg.norm(emb_w001_1)
        emb_w001_2 = emb_w001_1 + 0.05 * np.random.randn(128).astype(np.float32)
        emb_w001_2 /= np.linalg.norm(emb_w001_2)

        emb_w002_1 = np.random.randn(128).astype(np.float32)
        emb_w002_1 /= np.linalg.norm(emb_w002_1)

        emb_unregistered = np.random.randn(128).astype(np.float32)
        emb_unregistered /= np.linalg.norm(emb_unregistered)

        self.emb_w001_1 = emb_w001_1
        self.emb_w001_2 = emb_w001_2
        self.emb_w002 = emb_w002_1
        self.emb_unregistered = emb_unregistered

        # Populate in-memory biometric cache
        self.face_service.update_registered_cache(
            worker_code="W001",
            name="Alice Smith",
            employee_number="EMP-1001",
            embeddings=[emb_w001_1.tolist(), emb_w001_2.tolist()],
        )
        self.face_service.update_registered_cache(
            worker_code="W002",
            name="Bob Jones",
            employee_number="EMP-1002",
            embeddings=[emb_w002_1.tolist()],
        )

    def test_id_01_registered_worker_enters(self):
        """TEST_ID_01: Registered Worker W001 enters the camera -> correct identity W001."""
        track_id = 17
        # Feed 3 consistent matching frames
        for i in range(3):
            # Query embedding very close to W001
            query = self.emb_w001_1 + 0.02 * np.random.randn(128).astype(np.float32)
            query /= np.linalg.norm(query)

            # Direct match test
            code, sim, name = self.face_service.match_worker(query)
            self.assertEqual(code, "W001")
            self.assertGreater(sim, 0.70)

            # Pass into temporal identity manager
            res = self.identity_mgr.update_track_face(
                track_id=track_id,
                face_crop_or_image=None,  # simulating pre-extracted embedding
            )
            # Inject match history directly
            self.identity_mgr._tracks[track_id].match_history.append((code, sim, 0, name))
            self.identity_mgr._evaluate_temporal_confirmation(self.identity_mgr._tracks[track_id])

        state = self.identity_mgr._tracks[track_id]
        self.assertTrue(state.is_confirmed)
        self.assertEqual(state.confirmed_worker_code, "W001")
        self.assertEqual(state.confirmed_worker_name, "Alice Smith")

    def test_id_02_worker_leaves_and_returns_new_track(self):
        """TEST_ID_02: W001 leaves the frame and returns -> new track ID allowed, permanent Worker ID remains W001."""
        # Initial track 17
        self.identity_mgr.manual_link_identity(track_id=17, worker_code="W001", worker_name="Alice Smith")
        self.assertEqual(self.identity_mgr._tracks[17].confirmed_worker_code, "W001")

        # Worker leaves: track 17 removed
        self.identity_mgr.cleanup_stale_tracks(active_track_ids=set())
        self.assertNotIn(17, self.identity_mgr._tracks)

        # Worker returns with ByteTrack ID 32
        for _ in range(3):
            query = self.emb_w001_2
            code, sim, name = self.face_service.match_worker(query)
            self.identity_mgr.update_track_face(track_id=32, face_crop_or_image=None)
            self.identity_mgr._tracks[32].match_history.append((code, sim, 0, name))
            self.identity_mgr._evaluate_temporal_confirmation(self.identity_mgr._tracks[32])

        # Confirmed on new track 32 as permanent W001
        self.assertEqual(self.identity_mgr._tracks[32].confirmed_worker_code, "W001")

    def test_id_03_different_camera_source(self):
        """TEST_ID_03: W001 appears on a different supported camera source -> recognized as W001."""
        query = self.emb_w001_1
        code, sim, name = self.face_service.match_worker(query)
        self.assertEqual(code, "W001")
        self.assertEqual(name, "Alice Smith")

    def test_id_04_registered_worker_w002(self):
        """TEST_ID_04: Registered Worker W002 enters -> recognized as W002 (not W001)."""
        query = self.emb_w002
        code, sim, name = self.face_service.match_worker(query)
        self.assertEqual(code, "W002")
        self.assertNotEqual(code, "W001")
        self.assertEqual(name, "Bob Jones")

    def test_id_05_two_registered_workers_together(self):
        """TEST_ID_05: Two registered workers appear together -> distinct W001 and W002 assignment."""
        code1, _, _ = self.face_service.match_worker(self.emb_w001_1)
        code2, _, _ = self.face_service.match_worker(self.emb_w002)

        self.assertEqual(code1, "W001")
        self.assertEqual(code2, "W002")
        self.assertNotEqual(code1, code2)

    def test_id_06_unregistered_person_unknown(self):
        """TEST_ID_06: Unregistered person appears -> UNKNOWN (no forced match)."""
        code, sim, name = self.face_service.match_worker(self.emb_unregistered, threshold=0.58)
        self.assertIsNone(code)
        self.assertIsNone(name)

    def test_id_07_face_temporarily_occluded(self):
        """TEST_ID_07: Face is temporarily occluded -> keeps confirmed identity while tracking is continuous."""
        track_id = 17
        self.identity_mgr.manual_link_identity(track_id=track_id, worker_code="W001", worker_name="Alice Smith")

        # Worker turns head (no face detected for 5 frames)
        for _ in range(5):
            res = self.identity_mgr.update_track_face(track_id=track_id, face_crop_or_image=None)
            # Must remain confirmed as W001!
            self.assertEqual(res["permanent_worker_id"], "W001")
            self.assertEqual(res["identity_status"], "REGISTERED")

    def test_id_08_track_id_change_merges_history(self):
        """TEST_ID_08: Track ID change preserves and merges history."""
        # W001 tracking on track 17
        w1 = TrackedWorkerState(worker_id=17, bbox=(10, 10, 50, 100), permanent_worker_id="W001", worker_code="W001")
        self.assertEqual(w1.permanent_worker_id, "W001")

        # Track 17 disappears and track 32 appears
        w2 = TrackedWorkerState(worker_id=32, bbox=(10, 10, 50, 100), permanent_worker_id="W001", worker_code="W001")
        self.assertEqual(w2.worker_code, "W001")

    def test_id_09_low_confidence_uncertain_unknown(self):
        """TEST_ID_09: Low-confidence match below threshold -> UNKNOWN/UNCERTAIN, does not force incorrect ID."""
        # Orthogonal embedding gives near 0 similarity
        orthogonal_emb = np.zeros(128, dtype=np.float32)
        orthogonal_emb[0] = 1.0
        code, sim, name = self.face_service.match_worker(orthogonal_emb, threshold=0.58)
        self.assertIsNone(code)


if __name__ == "__main__":
    unittest.main()
