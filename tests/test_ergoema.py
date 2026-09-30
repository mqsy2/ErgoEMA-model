"""
Unit and Integration Tests for ErgoEMA Pipeline.
"""

import unittest
import numpy as np
import os
import tempfile
from src.pose_detector import PoseLandmarks, Point3D
from src.feature_extractor import FeatureExtractor, PostureFeatures
from src.ema_filter import EMAFilter
from src.calibrator import PostureCalibrator, BaselineProfile
from src.classifier import PostureClassifier, PostureState, PostureAssessment

class TestErgoEMA(unittest.TestCase):

    def setUp(self):
        # Mock upright posture landmarks
        self.mock_upright_pose = PoseLandmarks(
            raw_landmarks=None,
            nose=Point3D(x=0.5, y=0.3, z=-0.2, visibility=0.9),
            left_eye=Point3D(x=0.48, y=0.28, z=-0.2, visibility=0.9),
            right_eye=Point3D(x=0.52, y=0.28, z=-0.2, visibility=0.9),
            left_ear=Point3D(x=0.45, y=0.29, z=-0.15, visibility=0.9),
            right_ear=Point3D(x=0.55, y=0.29, z=-0.15, visibility=0.9),
            left_shoulder=Point3D(x=0.35, y=0.6, z=0.0, visibility=0.95),
            right_shoulder=Point3D(x=0.65, y=0.6, z=0.0, visibility=0.95),
            is_valid_upper_body=True
        )

        # Mock slouched posture (head drops vertically from y=0.3 to y=0.48, closer to shoulders at y=0.6)
        self.mock_slouch_pose = PoseLandmarks(
            raw_landmarks=None,
            nose=Point3D(x=0.5, y=0.48, z=-0.35, visibility=0.9), # lower y = dropped head, more negative z = forward head
            left_eye=Point3D(x=0.48, y=0.46, z=-0.35, visibility=0.9),
            right_eye=Point3D(x=0.52, y=0.46, z=-0.35, visibility=0.9),
            left_ear=Point3D(x=0.45, y=0.47, z=-0.3, visibility=0.9),
            right_ear=Point3D(x=0.55, y=0.47, z=-0.3, visibility=0.9),
            left_shoulder=Point3D(x=0.35, y=0.6, z=0.0, visibility=0.95),
            right_shoulder=Point3D(x=0.65, y=0.6, z=0.0, visibility=0.95),
            is_valid_upper_body=True
        )

    def test_feature_extractor_upright(self):
        extractor = FeatureExtractor()
        feat = extractor.extract(self.mock_upright_pose)
        self.assertIsNotNone(feat)
        
        # Shoulder width = 0.65 - 0.35 = 0.30
        # Delta Y = 0.6 - 0.3 = 0.30
        # Head-to-Shoulder Ratio = 0.30 / 0.30 = 1.0
        self.assertAlmostEqual(feat.shoulder_width_norm, 0.30, places=4)
        self.assertAlmostEqual(feat.head_to_shoulder_ratio, 1.0, places=4)
        self.assertAlmostEqual(feat.shoulder_tilt_deg, 0.0, places=1)
        self.assertAlmostEqual(feat.head_roll_deg, 0.0, places=1)

    def test_feature_extractor_slouch(self):
        extractor = FeatureExtractor()
        feat = extractor.extract(self.mock_slouch_pose)
        self.assertIsNotNone(feat)
        
        # In slouch, Delta Y = 0.6 - 0.48 = 0.12
        # Head-to-Shoulder Ratio = 0.12 / 0.30 = 0.40 (Significant compression!)
        self.assertAlmostEqual(feat.head_to_shoulder_ratio, 0.40, places=4)
        self.assertLess(feat.head_to_shoulder_ratio, 0.6)

    def test_ema_filter_smoothing(self):
        ema = EMAFilter(alpha=0.2)
        # Sequence of noisy values jumping around 1.0
        inputs = [1.0, 1.2, 0.8, 1.1, 0.9, 1.05]
        smoothed_vals = []
        for v in inputs:
            feat = PostureFeatures(
                head_to_shoulder_ratio=v,
                shoulder_tilt_deg=0.0,
                head_roll_deg=0.0,
                forward_head_z=0.0,
                shoulder_width_norm=0.3,
                neck_lateral_flexion_deg=0.0
            )
            s = ema.update(feat)
            smoothed_vals.append(s.head_to_shoulder_ratio)

        # First value should equal first input
        self.assertEqual(smoothed_vals[0], 1.0)
        # Smoothing should damp variance
        self.assertLess(np.std(smoothed_vals), np.std(inputs))

    def test_calibrator_and_profile_serialization(self):
        calibrator = PostureCalibrator(target_frames=10)
        calibrator.start("test_subject")
        extractor = FeatureExtractor()
        
        for _ in range(10):
            feat = extractor.extract(self.mock_upright_pose)
            completed, progress = calibrator.add_frame(feat)

        self.assertTrue(completed)
        self.assertEqual(progress, 1.0)
        profile = calibrator.profile
        self.assertIsNotNone(profile)
        self.assertAlmostEqual(profile.mean_h2s_ratio, 1.0, places=4)

        # Test JSON save/load
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            profile.save_json(tmp_path)
            loaded = BaselineProfile.load_json(tmp_path)
            self.assertEqual(loaded.user_id, "test_subject")
            self.assertAlmostEqual(loaded.mean_h2s_ratio, 1.0, places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_adaptive_classification_alert(self):
        classifier = PostureClassifier(slouch_ratio_drop_thresh=0.15, sustained_window_sec=1.0)
        baseline = BaselineProfile(
            mean_h2s_ratio=1.0,
            std_h2s_ratio=0.02,
            mean_shoulder_tilt_deg=0.0,
            std_shoulder_tilt_deg=0.5,
            mean_head_roll_deg=0.0,
            std_head_roll_deg=0.5,
            mean_forward_head_z=-0.2,
            std_forward_head_z=0.01,
            mean_shoulder_width_norm=0.3,
            sample_count=90
        )

        extractor = FeatureExtractor()
        slouch_feat = extractor.extract(self.mock_slouch_pose) # ratio = 0.40

        # At t=0s, bad posture detected, but sustained timer not met -> is_alert=False
        assessment1 = classifier.evaluate_adaptive(slouch_feat, baseline, current_time=0.0)
        self.assertEqual(assessment1.state, PostureState.SLOUCH)
        self.assertFalse(assessment1.is_alert)

        # At t=1.2s (> 1.0s window), bad posture sustained -> is_alert=True
        assessment2 = classifier.evaluate_adaptive(slouch_feat, baseline, current_time=1.2)
        self.assertEqual(assessment2.state, PostureState.SLOUCH)
        self.assertTrue(assessment2.is_alert)
        self.assertGreater(len(assessment2.reasons), 0)

    def test_lateral_spine_posterior_offsets(self):
        """Tests that lateral spine points are positioned posteriorly along the back in both directions."""
        from src.pose_detector import PoseDetector
        import cv2

        detector = PoseDetector()

        # Case 1: User facing RIGHT (+X). Back is to the LEFT (-X).
        # Ear at x=0.40, nose at x=0.60, shoulder at x=0.35
        class MockLandmark:
            def __init__(self, x, y, z=0.0, visibility=1.0):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility

        # 33 landmarks
        lms_right = [MockLandmark(0.5, 0.5)] * 33
        lms_right[0] = MockLandmark(0.60, 0.30)   # nose
        lms_right[7] = MockLandmark(0.55, 0.30, z=0.5)   # left ear (occluded)
        lms_right[8] = MockLandmark(0.40, 0.30, z=-0.5)  # right ear (visible)
        lms_right[11] = MockLandmark(0.45, 0.50, z=0.5)  # left shoulder
        lms_right[12] = MockLandmark(0.35, 0.50, z=-0.5) # right shoulder
        lms_right[23] = MockLandmark(0.45, 0.85, z=0.5)
        lms_right[24] = MockLandmark(0.35, 0.85, z=-0.5)

        # Case 2: User facing LEFT (-X). Back is to the RIGHT (+X).
        lms_left = [MockLandmark(0.5, 0.5)] * 33
        lms_left[0] = MockLandmark(0.40, 0.30)    # nose
        lms_left[7] = MockLandmark(0.60, 0.30, z=-0.5)   # left ear (visible)
        lms_left[8] = MockLandmark(0.45, 0.30, z=0.5)    # right ear (occluded)
        lms_left[11] = MockLandmark(0.65, 0.50, z=-0.5)  # left shoulder
        lms_left[12] = MockLandmark(0.55, 0.50, z=0.5)   # right shoulder
        lms_left[23] = MockLandmark(0.65, 0.85, z=-0.5)
        lms_left[24] = MockLandmark(0.55, 0.85, z=0.5)

        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Test drawing without errors
        res_r = detector.draw_skeleton(frame.copy(), lms_right, is_lateral=True)
        res_l = detector.draw_skeleton(frame.copy(), lms_left, is_lateral=True)
        self.assertIsNotNone(res_r)
        self.assertIsNotNone(res_l)
        detector.close()

    def test_lateral_feature_extraction_direction_invariance(self):
        """Verifies lateral FHP and lean metrics remain positive regardless of facing direction."""
        extractor = FeatureExtractor()

        # Facing Right: ear at 0.40, shoulder at 0.35 -> forward offset = +0.05
        pose_r = PoseLandmarks(
            raw_landmarks=None,
            nose=Point3D(x=0.60, y=0.30, z=-0.5, visibility=1.0),
            left_eye=Point3D(x=0.55, y=0.28, z=0.0, visibility=0.9),
            right_eye=Point3D(x=0.55, y=0.28, z=0.0, visibility=0.9),
            left_ear=Point3D(x=0.45, y=0.30, z=0.5, visibility=0.1),
            right_ear=Point3D(x=0.40, y=0.30, z=-0.5, visibility=1.0),
            left_shoulder=Point3D(x=0.38, y=0.50, z=0.5, visibility=0.1),
            right_shoulder=Point3D(x=0.35, y=0.50, z=-0.5, visibility=1.0),
            is_valid_upper_body=True
        )

        feat_r = extractor.extract(pose_r)
        self.assertIsNotNone(feat_r)
        self.assertEqual(feat_r.detected_view_angle, "lateral")
        self.assertGreater(feat_r.ear_shoulder_offset_x, 0.0)
        self.assertGreater(feat_r.nose_shoulder_angle_deg, 0.0)

        # Facing Left: ear at 0.60, shoulder at 0.65 -> forward offset = +0.05
        pose_l = PoseLandmarks(
            raw_landmarks=None,
            nose=Point3D(x=0.40, y=0.30, z=-0.5, visibility=1.0),
            left_eye=Point3D(x=0.45, y=0.28, z=0.0, visibility=0.9),
            right_eye=Point3D(x=0.45, y=0.28, z=0.0, visibility=0.9),
            left_ear=Point3D(x=0.60, y=0.30, z=-0.5, visibility=1.0),
            right_ear=Point3D(x=0.55, y=0.30, z=0.5, visibility=0.1),
            left_shoulder=Point3D(x=0.65, y=0.50, z=-0.5, visibility=1.0),
            right_shoulder=Point3D(x=0.62, y=0.50, z=0.5, visibility=0.1),
            is_valid_upper_body=True
        )

        feat_l = extractor.extract(pose_l)
        self.assertIsNotNone(feat_l)
        self.assertEqual(feat_l.detected_view_angle, "lateral")
        self.assertGreater(feat_l.ear_shoulder_offset_x, 0.0)
        self.assertGreater(feat_l.nose_shoulder_angle_deg, 0.0)

    def test_config_loads_tuned_hyperparameters(self):
        """Verifies the runtime config picks up train.py's tuned values and falls back to defaults otherwise."""
        import json
        from config import ErgoConfig

        defaults = ErgoConfig()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump({"optimal_hyperparameters": {"alpha": 0.08, "slouch_thresh": 0.1}}, tmp)
            tmp_path = tmp.name
        try:
            config = ErgoConfig()
            self.assertTrue(config.load_optimized(tmp_path))
            self.assertAlmostEqual(config.DEFAULT_EMA_ALPHA, 0.08)
            self.assertAlmostEqual(config.SLOUCH_RATIO_DROP_THRESH, 0.1)
            # The alert window is not tuned by train.py, so it keeps its default
            self.assertEqual(config.SUSTAINED_ALERT_WINDOW_SEC, defaults.SUSTAINED_ALERT_WINDOW_SEC)
        finally:
            os.remove(tmp_path)

        missing = ErgoConfig()
        self.assertFalse(missing.load_optimized(tmp_path))
        self.assertEqual(missing.DEFAULT_EMA_ALPHA, defaults.DEFAULT_EMA_ALPHA)
        self.assertEqual(missing.SLOUCH_RATIO_DROP_THRESH, defaults.SLOUCH_RATIO_DROP_THRESH)

    def test_loso_tuning_ignores_held_out_participant(self):
        """Verifies LOSO selects hyperparameters from the training participants only."""
        from train import metrics_from_counts, pool_metrics, select_best_params

        good = metrics_from_counts(tp=90, tn=90, fp=10, fn=10)
        bad = metrics_from_counts(tp=10, tn=90, fp=10, fn=90)
        # Setting A suits P1 and P2; setting B suits only P3
        grid = {
            (0.08, 0.08): {"P1": good, "P2": good, "P3": bad},
            (0.25, 0.18): {"P1": bad, "P2": bad, "P3": good},
        }

        key, train_metrics = select_best_params(grid, ["P1", "P2"])
        self.assertEqual(key, (0.08, 0.08))
        self.assertAlmostEqual(train_metrics["accuracy"], 0.90)
        # Held-out P3 is scored with the setting chosen without it, not its own best one
        self.assertAlmostEqual(grid[key]["P3"]["recall"], 0.10)

        pooled = pool_metrics(grid[(0.08, 0.08)], ["P1", "P2", "P3"])
        self.assertEqual((pooled["tp"], pooled["fn"]), (190, 110))

if __name__ == "__main__":
    unittest.main()
