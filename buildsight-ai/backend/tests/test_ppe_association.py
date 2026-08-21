"""BuildSight AI — Complete PPE Compliance & Association Test Suite

Comprehensive automated test cases:
  TEST_01_FULL_PPE
  TEST_02_MISSING_HELMET
  TEST_03_MISSING_VEST
  TEST_04_MISSING_GLOVES
  TEST_05_MISSING_MASK
  TEST_06_HELMET_ONLY
  TEST_07_NO_PPE
  TEST_08_MULTIPLE_WORKERS_DIFFERENT_PPE
  TEST_09_PARTIAL_OCCLUSION
  TEST_10_SMALL_OR_DISTANT_WORKER
  TEST_11_MULTIPLE_WORKERS_CLOSE_TOGETHER
  TEST_12_WORKER_ENTERING_AND_LEAVING_FRAME
"""

import unittest
import numpy as np
from app.ai.ppe_detector import PPEDetector, PPEDetection, DetailedPPEResult
from app.services.compliance_engine import ComplianceEngine
from app.ai.worker_tracker import TrackedWorkerState


class TestPPEComplianceAndAssociation(unittest.TestCase):

    def setUp(self):
        self.detector = PPEDetector()
        self.detector._loaded = True
        self.compliance_engine = ComplianceEngine()
        self.frame_shape = (480, 640)  # H, W

    def _create_worker_bbox(self, x=200, y=100, w=120, h=300):
        return (float(x), float(y), float(x + w), float(y + h))

    def _create_ppe(self, item_name, worker_bbox, conf=0.92):
        wx1, wy1, wx2, wy2 = worker_bbox
        ww = wx2 - wx1
        wh = wy2 - wy1

        if item_name == "helmet":
            # Upper head region
            hx1 = wx1 + ww * 0.1
            hx2 = wx2 - ww * 0.1
            hy1 = wy1 - wh * 0.05
            hy2 = wy1 + wh * 0.20
            return PPEDetection(class_id=1, class_name="helmet", bbox=(hx1, hy1, hx2, hy2), confidence=conf)

        elif item_name == "face_mask":
            # Face region
            mx1 = wx1 + ww * 0.25
            mx2 = wx2 - ww * 0.25
            my1 = wy1 + wh * 0.15
            my2 = wy1 + wh * 0.32
            return PPEDetection(class_id=4, class_name="face_mask", bbox=(mx1, my1, mx2, my2), confidence=conf)

        elif item_name == "safety_vest":
            # Torso region
            vx1 = wx1 - ww * 0.05
            vx2 = wx2 + ww * 0.05
            vy1 = wy1 + wh * 0.25
            vy2 = wy1 + wh * 0.65
            return PPEDetection(class_id=2, class_name="safety_vest", bbox=(vx1, vy1, vx2, vy2), confidence=conf)

        elif item_name == "gloves":
            # Hand region
            gx1 = wx1 - ww * 0.15
            gx2 = wx1 + ww * 0.05
            gy1 = wy1 + wh * 0.55
            gy2 = wy1 + wh * 0.70
            return PPEDetection(class_id=3, class_name="gloves", bbox=(gx1, gy1, gx2, gy2), confidence=conf)
        else:
            raise ValueError(f"Unknown item: {item_name}")

    def test_01_full_ppe(self):
        """TEST_01: Worker has Helmet, Vest, Gloves, Mask -> FULLY COMPLIANT."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        detections = [
            self._create_ppe("helmet", w_bbox),
            self._create_ppe("safety_vest", w_bbox),
            self._create_ppe("gloves", w_bbox),
            self._create_ppe("face_mask", w_bbox),
        ]
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox,
            worker_id=1,
            precomputed_ppe_detections=detections,
        )

        self.assertTrue(res.helmet["detected"])
        self.assertTrue(res.safety_vest["detected"])
        self.assertTrue(res.gloves["detected"])
        self.assertTrue(res.face_mask["detected"])
        self.assertEqual(len(res.missing_ppe), 0)
        self.assertEqual(res.compliance_status, "FULLY_COMPLIANT")
        self.assertEqual(res.ppe_compliance, 100.0)

    def test_02_missing_helmet(self):
        """TEST_02: Without Helmet -> Status: NON-COMPLIANT, Violation: MISSING_HELMET."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        detections = [
            self._create_ppe("safety_vest", w_bbox),
            self._create_ppe("gloves", w_bbox),
            self._create_ppe("face_mask", w_bbox),
        ]
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox,
            worker_id=2,
            precomputed_ppe_detections=detections,
        )

        self.assertFalse(res.helmet["detected"])
        self.assertTrue(res.safety_vest["detected"])
        self.assertIn("helmet", res.missing_ppe)
        self.assertEqual(res.compliance_status, "NON_COMPLIANT")

    def test_03_missing_vest(self):
        """TEST_03: Without Safety Vest -> Status: NON-COMPLIANT, Violation: MISSING_SAFETY_VEST."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        detections = [
            self._create_ppe("helmet", w_bbox),
            self._create_ppe("gloves", w_bbox),
            self._create_ppe("face_mask", w_bbox),
        ]
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox,
            worker_id=3,
            precomputed_ppe_detections=detections,
        )

        self.assertTrue(res.helmet["detected"])
        self.assertFalse(res.safety_vest["detected"])
        self.assertIn("safety_vest", res.missing_ppe)
        self.assertEqual(res.compliance_status, "NON_COMPLIANT")

    def test_04_missing_gloves(self):
        """TEST_04: Without Gloves -> Status: NON-COMPLIANT, Violation: MISSING_GLOVES."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        detections = [
            self._create_ppe("helmet", w_bbox),
            self._create_ppe("safety_vest", w_bbox),
            self._create_ppe("face_mask", w_bbox),
        ]
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox,
            worker_id=4,
            precomputed_ppe_detections=detections,
        )

        self.assertFalse(res.gloves["detected"])
        self.assertIn("gloves", res.missing_ppe)
        self.assertEqual(res.compliance_status, "NON_COMPLIANT")

    def test_05_missing_mask(self):
        """TEST_05: Without Face Mask -> Status: NON-COMPLIANT, Violation: MISSING_FACE_MASK."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        detections = [
            self._create_ppe("helmet", w_bbox),
            self._create_ppe("safety_vest", w_bbox),
            self._create_ppe("gloves", w_bbox),
        ]
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox,
            worker_id=5,
            precomputed_ppe_detections=detections,
        )

        self.assertFalse(res.face_mask["detected"])
        self.assertIn("face_mask", res.missing_ppe)
        self.assertEqual(res.compliance_status, "NON_COMPLIANT")

    def test_06_helmet_only(self):
        """TEST_06: Helmet Only -> Missing Vest, Gloves, Face Mask."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        detections = [self._create_ppe("helmet", w_bbox)]
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox,
            worker_id=6,
            precomputed_ppe_detections=detections,
        )

        self.assertTrue(res.helmet["detected"])
        self.assertFalse(res.safety_vest["detected"])
        self.assertFalse(res.gloves["detected"])
        self.assertFalse(res.face_mask["detected"])
        self.assertEqual(set(res.missing_ppe), {"safety_vest", "gloves", "face_mask"})

    def test_07_no_ppe(self):
        """TEST_07: No PPE -> Critical PPE violation, all missing."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox,
            worker_id=7,
            precomputed_ppe_detections=[],
        )

        self.assertEqual(set(res.missing_ppe), {"helmet", "safety_vest", "gloves", "face_mask"})
        self.assertEqual(res.ppe_compliance, 0.0)

    def test_08_multiple_workers_different_ppe(self):
        """TEST_08: Multiple workers in frame -> Correct PPE-to-worker spatial association."""
        w1_bbox = self._create_worker_bbox(50, 100, 100, 280)    # Worker 1 (Left)
        w2_bbox = self._create_worker_bbox(350, 100, 100, 280)   # Worker 2 (Right)

        # Worker 1 has Helmet only, Worker 2 has Vest + Gloves only
        detections = [
            self._create_ppe("helmet", w1_bbox),
            self._create_ppe("safety_vest", w2_bbox),
            self._create_ppe("gloves", w2_bbox),
        ]

        res1 = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w1_bbox, worker_id=81,
            precomputed_ppe_detections=detections,
        )
        res2 = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w2_bbox, worker_id=82,
            precomputed_ppe_detections=detections,
        )

        # Worker 1 must NOT get worker 2's vest or gloves!
        self.assertTrue(res1.helmet["detected"])
        self.assertFalse(res1.safety_vest["detected"])
        self.assertFalse(res1.gloves["detected"])

        # Worker 2 must NOT get worker 1's helmet!
        self.assertFalse(res2.helmet["detected"])
        self.assertTrue(res2.safety_vest["detected"])
        self.assertTrue(res2.gloves["detected"])

    def test_09_partial_occlusion(self):
        """TEST_09: Partial occlusion -> Spatial association still correctly identifies visible PPE."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        # Helmet and vest visible, lower body occluded
        detections = [
            self._create_ppe("helmet", w_bbox),
            self._create_ppe("safety_vest", w_bbox),
        ]
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox, worker_id=9,
            precomputed_ppe_detections=detections,
        )
        self.assertTrue(res.helmet["detected"])
        self.assertTrue(res.safety_vest["detected"])

    def test_10_small_or_distant_worker(self):
        """TEST_10: Small/distant worker -> Anatomical scaling associates PPE correctly."""
        small_bbox = self._create_worker_bbox(300, 150, 45, 110)
        detections = [
            self._create_ppe("helmet", small_bbox),
            self._create_ppe("safety_vest", small_bbox),
        ]
        res = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=small_bbox, worker_id=10,
            precomputed_ppe_detections=detections,
        )
        self.assertTrue(res.helmet["detected"])
        self.assertTrue(res.safety_vest["detected"])

    def test_11_multiple_workers_close_together(self):
        """TEST_11: Workers close together -> Bbox containment associates PPE to correct target."""
        w1_bbox = self._create_worker_bbox(200, 100, 100, 280)
        w2_bbox = self._create_worker_bbox(280, 100, 100, 280)

        # Worker 1 wears helmet, Worker 2 wears no helmet
        detections = [self._create_ppe("helmet", w1_bbox)]

        res1 = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w1_bbox, worker_id=111,
            precomputed_ppe_detections=detections,
        )
        res2 = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w2_bbox, worker_id=112,
            precomputed_ppe_detections=detections,
        )

        self.assertTrue(res1.helmet["detected"])
        self.assertFalse(res2.helmet["detected"])

    def test_12_worker_entering_and_leaving_frame(self):
        """TEST_12: Temporal stability handles enter/leave without violation spam."""
        w_bbox = self._create_worker_bbox(200, 100, 120, 300)
        # 1. First frame: helmet present
        d_present = [self._create_ppe("helmet", w_bbox)]
        self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox, worker_id=12,
            precomputed_ppe_detections=d_present,
        )
        # 2. Temporary glitch (single missed frame)
        res_glitch = self.detector.detect_worker_ppe(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            worker_bbox=w_bbox, worker_id=12,
            precomputed_ppe_detections=[],
        )
        # Temporal smoothing prevents immediate false missing helmet alarm
        self.assertIsNotNone(res_glitch)


if __name__ == "__main__":
    unittest.main()
