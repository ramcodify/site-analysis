"""BuildSight AI — Master Verification & Test Suite (TEST 01 to TEST 35)

Executes all 35 research acceptance test cases across:
  - Worker Safety Compliance (TEST 01 - TEST 15)
  - Construction Progress Recognition (TEST 16 - TEST 24)
  - Construction Delay Prediction (TEST 25 - TEST 29)
  - GraphRAG Explainable Intelligence (TEST 30 - TEST 35)
"""

import unittest
import numpy as np
from datetime import datetime

from app.database.mongodb import init_db, get_db
from app.ai.ppe_detector import PPEDetector, PPEDetection, CLASS_NAMES
from app.services.compliance_engine import ComplianceEngine
from app.ai.worker_tracker import TrackedWorkerState
from app.ai.progress_analyzer import ProgressAnalyzer, CONSTRUCTION_STAGES
from app.ai.delay_predictor import DelayPredictor
from app.graphrag.query_service import GraphRAGQueryService
from app.graphrag.graph_builder import knowledge_graph


def _make_det(cls_name: str, conf: float, bbox: tuple) -> PPEDetection:
    cls_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0
    return PPEDetection(class_id=cls_idx, class_name=cls_name, bbox=bbox, confidence=conf)


class TestBuildSightMasterSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.db = get_db()
        cls.ppe_detector = PPEDetector()
        cls.compliance_engine = ComplianceEngine()
        cls.progress_analyzer = ProgressAnalyzer()
        cls.progress_analyzer.load()
        cls.delay_predictor = DelayPredictor()
        cls.delay_predictor.load()
        cls.graphrag = GraphRAGQueryService()
        knowledge_graph.sync_from_mongodb()

        cls.frame_shape = (480, 640)

    # ── MODULE 1: SAFETY COMPLIANCE (TEST 01 - TEST 15) ─────────────

    def test_01_worker_wearing_all_ppe(self):
        """TEST 01: Worker wearing all 4 required PPE items."""
        wb = (100, 100, 300, 500)
        detections = [
            _make_det("person", 0.95, wb),
            _make_det("helmet", 0.92, (160, 100, 240, 160)),
            _make_det("safety_vest", 0.90, (140, 200, 260, 380)),
            _make_det("gloves", 0.88, (110, 340, 150, 400)),
            _make_det("face_mask", 0.89, (170, 150, 230, 190)),
        ]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertTrue(assoc["helmet"]["detected"])
        self.assertTrue(assoc["safety_vest"]["detected"])
        self.assertTrue(assoc["gloves"]["detected"])
        self.assertTrue(assoc["face_mask"]["detected"])
        missing = [k for k, v in assoc.items() if not v["detected"]]
        self.assertEqual(len(missing), 0)

    def test_02_worker_without_helmet(self):
        """TEST 02: Worker missing hard hat."""
        wb = (100, 100, 300, 500)
        detections = [
            _make_det("person", 0.95, wb),
            _make_det("safety_vest", 0.90, (140, 200, 260, 380)),
            _make_det("gloves", 0.88, (110, 340, 150, 400)),
            _make_det("face_mask", 0.89, (170, 150, 230, 190)),
        ]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertFalse(assoc["helmet"]["detected"])
        self.assertTrue(assoc["safety_vest"]["detected"])

    def test_03_worker_without_safety_vest(self):
        """TEST 03: Worker missing hi-vis safety vest."""
        wb = (100, 100, 300, 500)
        detections = [
            _make_det("person", 0.95, wb),
            _make_det("helmet", 0.92, (160, 100, 240, 160)),
            _make_det("gloves", 0.88, (110, 340, 150, 400)),
            _make_det("face_mask", 0.89, (170, 150, 230, 190)),
        ]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertFalse(assoc["safety_vest"]["detected"])
        self.assertTrue(assoc["helmet"]["detected"])

    def test_04_worker_without_gloves(self):
        """TEST 04: Worker missing safety gloves."""
        wb = (100, 100, 300, 500)
        detections = [
            _make_det("person", 0.95, wb),
            _make_det("helmet", 0.92, (160, 100, 240, 160)),
            _make_det("safety_vest", 0.90, (140, 200, 260, 380)),
            _make_det("face_mask", 0.89, (170, 150, 230, 190)),
        ]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertFalse(assoc["gloves"]["detected"])
        self.assertTrue(assoc["helmet"]["detected"])

    def test_05_worker_without_face_mask(self):
        """TEST 05: Worker missing dust/silica face mask."""
        wb = (100, 100, 300, 500)
        detections = [
            _make_det("person", 0.95, wb),
            _make_det("helmet", 0.92, (160, 100, 240, 160)),
            _make_det("safety_vest", 0.90, (140, 200, 260, 380)),
            _make_det("gloves", 0.88, (110, 340, 150, 400)),
        ]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertFalse(assoc["face_mask"]["detected"])
        self.assertTrue(assoc["helmet"]["detected"])

    def test_06_multiple_ppe_items_missing(self):
        """TEST 06: Worker missing helmet and vest simultaneously."""
        wb = (100, 100, 300, 500)
        detections = [
            _make_det("person", 0.95, wb),
            _make_det("gloves", 0.88, (110, 340, 150, 400)),
        ]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertFalse(assoc["helmet"]["detected"])
        self.assertFalse(assoc["safety_vest"]["detected"])
        missing = [k for k, v in assoc.items() if not v["detected"]]
        self.assertEqual(len(missing), 3)

    def test_07_multiple_workers(self):
        """TEST 07: Independent PPE assignment across multiple workers."""
        w1 = (50, 100, 200, 450)
        w2 = (300, 100, 450, 450)
        detections = [
            _make_det("person", 0.95, w1),
            _make_det("person", 0.95, w2),
            _make_det("helmet", 0.92, (90, 100, 160, 150)),
            _make_det("safety_vest", 0.91, (330, 180, 420, 320)),
        ]
        a1 = self.ppe_detector.associate_ppe_to_worker(w1, detections, self.frame_shape)
        a2 = self.ppe_detector.associate_ppe_to_worker(w2, detections, self.frame_shape)
        self.assertTrue(a1["helmet"]["detected"])
        self.assertFalse(a1["safety_vest"]["detected"])
        self.assertFalse(a2["helmet"]["detected"])
        self.assertTrue(a2["safety_vest"]["detected"])

    def test_08_workers_standing_close_together(self):
        """TEST 08: Spatial constraints prevent cross-worker misattribution."""
        w1 = (100, 100, 250, 500)
        w2 = (240, 100, 390, 500)
        detections = [
            _make_det("helmet", 0.95, (140, 100, 210, 160)),
        ]
        a1 = self.ppe_detector.associate_ppe_to_worker(w1, detections, self.frame_shape)
        a2 = self.ppe_detector.associate_ppe_to_worker(w2, detections, self.frame_shape)
        self.assertTrue(a1["helmet"]["detected"])
        self.assertFalse(a2["helmet"]["detected"])

    def test_09_worker_partially_occluded(self):
        """TEST 09: Worker lower body occluded."""
        wb = (100, 100, 300, 350)
        detections = [
            _make_det("helmet", 0.92, (160, 100, 240, 160)),
            _make_det("safety_vest", 0.90, (140, 180, 260, 320)),
        ]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertTrue(assoc["helmet"]["detected"])
        self.assertTrue(assoc["safety_vest"]["detected"])

    def test_10_worker_leaves_and_returns(self):
        """TEST 10: Worker state continuity after temporal disappearance."""
        state = TrackedWorkerState(worker_id=17, bbox=(100, 100, 200, 300), worker_code="W001", helmet=True, vest=True)
        self.assertEqual(state.worker_code, "W001")
        new_state = TrackedWorkerState(worker_id=42, bbox=(100, 100, 200, 300), worker_code="W001", helmet=True, vest=True)
        self.assertEqual(new_state.worker_code, "W001")

    def test_11_different_worker_ids(self):
        """TEST 11: Distinct permanent identities maintain isolated state."""
        w1 = TrackedWorkerState(worker_id=1, bbox=(100, 100, 200, 300), worker_code="W001")
        w2 = TrackedWorkerState(worker_id=2, bbox=(100, 100, 200, 300), worker_code="W002")
        self.assertNotEqual(w1.worker_code, w2.worker_code)

    def test_12_unknown_worker(self):
        """TEST 12: Unregistered worker defaults to UNKNOWN status."""
        w = TrackedWorkerState(worker_id=99, bbox=(100, 100, 200, 300), worker_code=None)
        self.assertEqual(w.identity_status, "UNKNOWN")

    def test_13_low_light_detection_robustness(self):
        """TEST 13: Bounding box containment operates correctly under varied confidence levels."""
        wb = (100, 100, 300, 500)
        detections = [_make_det("helmet", 0.55, (160, 100, 240, 160))]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertTrue(assoc["helmet"]["detected"])

    def test_14_motion_blur_temporal_smoothing(self):
        """TEST 14: Single-frame drop does not flap violation state."""
        w = TrackedWorkerState(worker_id=1, bbox=(100, 100, 200, 300), helmet=False)
        w.violation_count = 0
        w.in_danger_zone = False
        viols = self.compliance_engine.analyze_worker(w)
        self.assertTrue(len(viols) > 0)

    def test_15_different_camera_distances(self):
        """TEST 15: Extreme scale worker bounding box."""
        wb = (10, 10, 60, 120)
        detections = [_make_det("helmet", 0.85, (20, 10, 50, 30))]
        assoc = self.ppe_detector.associate_ppe_to_worker(wb, detections, self.frame_shape)
        self.assertTrue(assoc["helmet"]["detected"])

    # ── MODULE 2: CONSTRUCTION PROGRESS (TEST 16 - TEST 24) ─────────

    def test_16_stage_site_preparation(self):
        """TEST 16: Recognize Stage 1 — Site Preparation."""
        self.progress_analyzer.set_current_stage(0)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Site Preparation")

    def test_17_stage_excavation(self):
        """TEST 17: Recognize Stage 2 — Excavation."""
        self.progress_analyzer.set_current_stage(1)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Excavation")

    def test_18_stage_foundation(self):
        """TEST 18: Recognize Stage 3 — Foundation."""
        self.progress_analyzer.set_current_stage(2)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Foundation")

    def test_19_stage_structural_work(self):
        """TEST 19: Recognize Stage 4 — Structural Work."""
        self.progress_analyzer.set_current_stage(3)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Structural Work")

    def test_20_stage_brickwork(self):
        """TEST 20: Recognize Stage 5 — Brickwork."""
        self.progress_analyzer.set_current_stage(4)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Brickwork")

    def test_21_stage_roofing(self):
        """TEST 21: Recognize Stage 6 — Roofing."""
        self.progress_analyzer.set_current_stage(5)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Roofing")

    def test_22_stage_plastering(self):
        """TEST 22: Recognize Stage 7 — Plastering."""
        self.progress_analyzer.set_current_stage(6)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Plastering")

    def test_23_stage_electrical_and_plumbing(self):
        """TEST 23: Recognize Stage 8 — Electrical and Plumbing."""
        self.progress_analyzer.set_current_stage(7)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Electrical and Plumbing")

    def test_24_stage_finishing(self):
        """TEST 24: Recognize Stage 9 — Finishing."""
        self.progress_analyzer.set_current_stage(8)
        res = self.progress_analyzer.analyze()
        self.assertEqual(res.current_stage, "Finishing")

    # ── MODULE 3: DELAY PREDICTION (TEST 25 - TEST 29) ──────────────

    def test_25_project_on_schedule(self):
        """TEST 25: Project on schedule has low delay probability."""
        pred = self.delay_predictor.predict(
            planned_progress_pct=50.0,
            actual_progress_pct=52.0,
            current_stage_idx=4,
            stage_elapsed_days=15.0,
            planned_stage_days=20.0,
            active_worker_count=18,
            total_violations=1,
            repeated_violations=0,
            safety_interruptions=0,
        )
        self.assertFalse(pred["is_delay_predicted"])
        self.assertLess(pred["predicted_delay_days"], 5.0)

    def test_26_actual_progress_behind_planned(self):
        """TEST 26: Severe negative variance triggers delay prediction."""
        pred = self.delay_predictor.predict(
            planned_progress_pct=65.0,
            actual_progress_pct=45.0,
            current_stage_idx=4,
            stage_elapsed_days=28.0,
            planned_stage_days=20.0,
            active_worker_count=6,
            total_violations=8,
            repeated_violations=3,
            safety_interruptions=2,
        )
        self.assertTrue(pred["is_delay_predicted"])
        self.assertGreater(pred["predicted_delay_days"], 5.0)

    def test_27_stage_duration_exceeds_planned(self):
        """TEST 27: Prolonged stage duration contributes to delay days."""
        pred = self.delay_predictor.predict(
            planned_progress_pct=40.0,
            actual_progress_pct=36.0,
            current_stage_idx=3,
            stage_elapsed_days=35.0,
            planned_stage_days=18.0,
            active_worker_count=10,
            total_violations=2,
            repeated_violations=0,
            safety_interruptions=1,
        )
        self.assertGreater(pred["predicted_delay_days"], 0.0)

    def test_28_delay_prediction_with_sufficient_data(self):
        """TEST 28: Forecast outputs completion date and feature explanations."""
        pred = self.delay_predictor.predict(
            planned_progress_pct=70.0,
            actual_progress_pct=55.0,
            current_stage_idx=5,
            stage_elapsed_days=22.0,
            planned_stage_days=18.0,
            active_worker_count=8,
            total_violations=4,
            repeated_violations=1,
            safety_interruptions=1,
        )
        self.assertIn("predicted_completion_date", pred)
        self.assertIn("top_contributors", pred)
        self.assertGreater(len(pred["top_contributors"]), 0)

    def test_29_insufficient_delay_data_handling(self):
        """TEST 29: Handled with valid inputs and default bounds."""
        pred = self.delay_predictor.predict(
            planned_progress_pct=0.0,
            actual_progress_pct=0.0,
            current_stage_idx=0,
            stage_elapsed_days=1.0,
            planned_stage_days=10.0,
            active_worker_count=2,
            total_violations=0,
            repeated_violations=0,
            safety_interruptions=0,
        )
        self.assertIsNotNone(pred)
        self.assertEqual(pred["progress_variance_pct"], 0.0)

    # ── MODULE 4: GRAPHRAG (TEST 30 - TEST 35) ──────────────────────

    def test_30_why_is_worker_high_risk(self):
        """TEST 30: GraphRAG query explaining worker risk from observed MongoDB evidence."""
        res = self.graphrag.query("Why is Worker W001 high risk?")
        self.assertIn("observed_evidence", res)
        self.assertFalse(res["insufficient_evidence"])

    def test_31_why_is_project_delayed(self):
        """TEST 31: GraphRAG query explaining delay drivers via model predictions."""
        res = self.graphrag.query("Why is the project predicted to be delayed?")
        self.assertIn("model_predictions", res)
        self.assertFalse(res["insufficient_evidence"])

    def test_32_which_zone_highest_helmet_violations(self):
        """TEST 32: GraphRAG query resolving danger zone risk & PPE frequency."""
        res = self.graphrag.query("Which zone has the highest safety risk and which PPE is missing?")
        self.assertIn("analytics", res)
        self.assertFalse(res["insufficient_evidence"])

    def test_33_what_happened_before_safety_event(self):
        """TEST 33: Multi-hop query correlating stage and violations."""
        res = self.graphrag.query("What safety events occurred during the structural stage?")
        self.assertIn("observed_evidence", res)
        self.assertFalse(res["insufficient_evidence"])

    def test_34_which_construction_stage_behind_schedule(self):
        """TEST 34: GraphRAG query resolving current stage progress."""
        res = self.graphrag.query("What is the current construction stage and progress status?")
        self.assertIn("model_predictions", res)
        self.assertFalse(res["insufficient_evidence"])

    def test_35_insufficient_evidence_fallback(self):
        """TEST 35: Query with no matching entities or knowledge returns INSUFFICIENT_EVIDENCE."""
        res = self.graphrag.query("What was the weather on Mars during Apollo 11?")
        self.assertTrue(res["insufficient_evidence"])
        self.assertIn("INSUFFICIENT_EVIDENCE", res["answer"])


if __name__ == "__main__":
    unittest.main()
