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

if __name__ == "__main__":
    unittest.main()
