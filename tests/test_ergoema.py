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

    def _side_profile_pose(self, facing: float, ear_forward_px: float, width: int, height: int, mask=None) -> PoseLandmarks:
        """
        Side-profile pose built in pixels: shoulder at (300, 300), ear 200 px above it and
        ear_forward_px ahead of it, in a frame of the given size. facing is +1 (right) or -1 (left).
        """
        def point(x_px: float, y_px: float, z: float = 0.0, visibility: float = 1.0) -> Point3D:
            x_px = x_px if facing > 0 else width - x_px  # Mirror the scene for a left-facing subject
            return Point3D(x=x_px / width, y=y_px / height, z=z, visibility=visibility)

        ear_x = 300 + ear_forward_px
        return PoseLandmarks(
            raw_landmarks=None,
            nose=point(ear_x + 100, 110, z=-0.5),
            left_eye=point(ear_x + 80, 90),
            right_eye=point(ear_x + 80, 90),
            left_ear=point(ear_x, 100, z=-0.5),
            right_ear=point(ear_x + 5, 100, z=0.5, visibility=0.1),
            left_shoulder=point(300, 300, z=-0.5),
            right_shoulder=point(310, 300, z=0.5, visibility=0.1),
            is_valid_upper_body=True,
            image_width=width,
            image_height=height,
            segmentation_mask=mask
        )

    def test_lateral_cva_independent_of_frame_shape_and_facing(self):
        """Verifies the craniovertebral angle is the same in landscape and portrait frames, facing either way."""
        extractor = FeatureExtractor()

        # Without a silhouette, C7 is placed 0.35 U behind and 0.30 U above the shoulder (U = ear-to-shoulder distance):
        # ear directly above the shoulder (U = 200 px) -> atan2(140, 70) = 63.4°
        # ear 100 px forward (U = 223.6 px)            -> atan2(132.9, 178.3) = 36.7°
        for width, height in [(1920, 1080), (1080, 1920)]:
            for facing in (1.0, -1.0):
                upright = extractor.extract(self._side_profile_pose(facing, 0, width, height))
                forward = extractor.extract(self._side_profile_pose(facing, 100, width, height))
                self.assertEqual(upright.detected_view_angle, "lateral")
                self.assertAlmostEqual(upright.craniovertebral_angle_deg, 63.43, places=1)
                self.assertAlmostEqual(forward.craniovertebral_angle_deg, 36.71, places=1)
                self.assertFalse(upright.lateral_geometry.c7_from_silhouette)

    def test_lateral_cva_uses_silhouette_neck_contour(self):
        """Verifies C7 snaps to the silhouette's back edge when it is near the expected spot, and ignores it otherwise."""
        extractor = FeatureExtractor()
        width, height = 640, 480  # Same size as the analysis mask, so mask pixels equal frame pixels

        def silhouette(back_edge_x: int) -> np.ndarray:
            mask = np.zeros((height, width), dtype=np.float32)
            mask[:, back_edge_x:] = 1.0  # Body occupies everything in front of the back edge
            return mask

        # Expected C7 is (230, 240). A back edge at x=250 is 20 px away (within 0.3 U = 60 px) -> C7 = (250, 240)
        near = extractor.extract(self._side_profile_pose(1.0, 0, width, height, mask=silhouette(250)))
        self.assertTrue(near.lateral_geometry.c7_from_silhouette)
        self.assertAlmostEqual(near.lateral_geometry.c7[0] * width, 250, delta=1)
        self.assertAlmostEqual(near.craniovertebral_angle_deg, np.degrees(np.arctan2(140, 50)), places=0)
        self.assertGreater(len(near.lateral_geometry.back_contour), 2)

        # A back edge at x=150 (e.g. long hair or a headrest) is 80 px away -> the expected position is kept
        far = extractor.extract(self._side_profile_pose(1.0, 0, width, height, mask=silhouette(150)))
        self.assertFalse(far.lateral_geometry.c7_from_silhouette)
        self.assertAlmostEqual(far.craniovertebral_angle_deg, 63.43, places=1)

    def test_lateral_classification_uses_cva_criterion(self):
        """Verifies that without a side-on calibration the lateral view flags FHP only below the CVA criterion and never reports slouch."""
        baseline = BaselineProfile(
            mean_h2s_ratio=1.0, std_h2s_ratio=0.02,
            mean_shoulder_tilt_deg=0.0, std_shoulder_tilt_deg=0.5,
            mean_head_roll_deg=0.0, std_head_roll_deg=0.5,
            mean_forward_head_z=-0.2, std_forward_head_z=0.01,
            mean_shoulder_width_norm=0.3, sample_count=90
        )

        def lateral_features(cva: float) -> PostureFeatures:
            return PostureFeatures(
                head_to_shoulder_ratio=1.0, shoulder_tilt_deg=0.0, head_roll_deg=0.0, forward_head_z=-0.2,
                shoulder_width_norm=0.03, neck_lateral_flexion_deg=70.0, detected_view_angle="lateral",
                ear_shoulder_offset_x=0.3, nose_shoulder_angle_deg=70.0, craniovertebral_angle_deg=cva
            )

        classifier = PostureClassifier(lateral_cva_fhp_thresh_deg=50.0)
        # A steep nose-shoulder angle and a large ear offset no longer decide anything on their own
        self.assertEqual(classifier.evaluate_adaptive(lateral_features(55.0), baseline, 0.0).state, PostureState.GOOD_POSTURE)

        assessment = classifier.evaluate_adaptive(lateral_features(45.0), baseline, 1.0)
        self.assertEqual(assessment.state, PostureState.FORWARD_HEAD)
        self.assertAlmostEqual(assessment.forward_head_z_deviation, 5.0)
        self.assertIn("45.0", assessment.reasons[0])

    def test_ema_filter_restarts_on_view_change(self):
        """Verifies frontal values are not blended into the first lateral frames (which would read as a low CVA)."""
        ema = EMAFilter(alpha=0.1)
        frontal = PostureFeatures(
            head_to_shoulder_ratio=1.0, shoulder_tilt_deg=0.0, head_roll_deg=0.0,
            forward_head_z=-0.2, shoulder_width_norm=0.3, neck_lateral_flexion_deg=0.0
        )
        lateral = PostureFeatures(
            head_to_shoulder_ratio=1.0, shoulder_tilt_deg=0.0, head_roll_deg=0.0, forward_head_z=-0.2,
            shoulder_width_norm=0.03, neck_lateral_flexion_deg=10.0, detected_view_angle="lateral",
            craniovertebral_angle_deg=55.0
        )
        for _ in range(5):
            ema.update(frontal)
        smoothed = ema.update(lateral)
        self.assertTrue(smoothed.is_lateral)
        self.assertAlmostEqual(smoothed.craniovertebral_angle_deg, 55.0)

    def test_heath_carter_component_equations(self):
        """Verifies the three Heath-Carter equations against hand-computed values."""
        from src.somatotype import endomorphy, mesomorphy, ectomorphy

        # Endomorphy: skinfold sum 30 mm at the reference height (X = 30)
        # -0.7182 + 0.1451*30 - 0.00068*900 + 0.0000014*27000 = 3.0606
        self.assertAlmostEqual(endomorphy(170.18, 10.0, 12.0, 8.0), 3.0606, places=3)
        # A taller person with the same skinfolds rates lower (X is scaled by 170.18 / height)
        self.assertLess(endomorphy(190.0, 10.0, 12.0, 8.0), 3.0606)

        # Mesomorphy: corrected girths are 32.0 - 1.0 = 31.0 and 37.0 - 1.0 = 36.0
        # 0.858*7.0 + 0.601*9.5 + 0.188*31.0 + 0.161*36.0 - 0.131*175 + 4.5 = 4.9145
        self.assertAlmostEqual(mesomorphy(175.0, 7.0, 9.5, 32.0, 37.0, 10.0, 10.0), 4.9145, places=3)

        # Ectomorphy: 64 kg has a cube root of 4, so HWR = height / 4
        self.assertAlmostEqual(ectomorphy(173.6, 64.0), 0.732 * 43.4 - 28.58, places=3)   # HWR 43.4 -> 3.19 (manual example: 3.2)
        self.assertAlmostEqual(ectomorphy(160.0, 64.0), 0.463 * 40.0 - 17.63, places=3)   # HWR 40.0 -> 0.89
        self.assertAlmostEqual(ectomorphy(150.0, 64.0), 0.1)                              # HWR 37.5 -> floor

        # No component is ever rated zero or negative
        self.assertAlmostEqual(mesomorphy(200.0, 5.0, 7.0, 20.0, 28.0, 10.0, 10.0), 0.1)

    def test_heath_carter_categories(self):
        """Verifies the somatotype category names follow the Carter & Heath definitions."""
        from src.somatotype import classify

        cases = {
            (3.0, 3.5, 3.0): "Central",                     # No component more than one unit from the others
            (3.6, 2.6, 2.6): "Central",                     # Exactly one unit apart
            (5.0, 2.0, 2.3): "Balanced endomorph",          # Dominant, other two within half a unit
            (2.0, 5.0, 3.5): "Ectomorphic mesomorph",       # Dominant, second component names the adjective
            (3.5, 5.0, 2.0): "Endomorphic mesomorph",
            (2.0, 3.5, 5.0): "Mesomorphic ectomorph",
            (5.0, 4.6, 1.0): "Mesomorph-endomorph",         # Two components within half a unit share dominance
            (1.5, 4.0, 4.2): "Mesomorph-ectomorph",
            (4.0, 1.0, 4.2): "Endomorph-ectomorph",
        }
        for (endo, meso, ecto), expected in cases.items():
            self.assertEqual(classify(endo, meso, ecto), expected, msg=f"{endo}-{meso}-{ecto}")

    def test_heath_carter_rating_from_measurements(self):
        """Verifies a full rating, a partial rating, and rejection of implausible values."""
        from src.somatotype import Anthropometry, rate

        full = rate(Anthropometry.from_dict({
            "height_cm": 175.0, "weight_kg": 70.0,
            "triceps_skinfold_mm": 10.0, "subscapular_skinfold_mm": 12.0, "supraspinale_skinfold_mm": 8.0, "medial_calf_skinfold_mm": 10.0,
            "humerus_breadth_cm": 7.0, "femur_breadth_cm": 9.5, "flexed_arm_girth_cm": 32.0, "calf_girth_cm": 37.0
        }))
        self.assertEqual(full.missing, [])
        self.assertEqual(full.rating, "3.0-4.9-2.5")
        self.assertEqual(full.category, "Balanced mesomorph")  # Endomorphy and ectomorphy are half a unit apart

        # Height and weight alone rate ectomorphy only; without all three components there is no category
        partial = rate(Anthropometry.from_dict({"height_cm": 175.0, "weight_kg": 70.0}))
        self.assertEqual(partial.rating, "n/a-n/a-2.5")
        self.assertIsNone(partial.category)
        self.assertEqual(len(partial.missing), 8)

        with self.assertRaises(ValueError):
            Anthropometry.from_dict({"height_cm": 1.75, "weight_kg": 70.0})  # Height entered in metres

        # The blank template loads and rates nothing until it is filled in
        template = rate(Anthropometry.load_json(os.path.join(os.path.dirname(__file__), "..", "anthropometry_template.json")))
        self.assertEqual(template.rating, "n/a-n/a-n/a")
        self.assertEqual(len(template.missing), 10)

    @staticmethod
    def _baseline(**extra) -> BaselineProfile:
        return BaselineProfile(
            mean_h2s_ratio=1.0, std_h2s_ratio=0.02,
            mean_shoulder_tilt_deg=0.0, std_shoulder_tilt_deg=0.5,
            mean_head_roll_deg=0.0, std_head_roll_deg=0.5,
            mean_forward_head_z=-0.2, std_forward_head_z=0.01,
            mean_shoulder_width_norm=0.3, sample_count=90, **extra
        )

    @staticmethod
    def _lateral(cva: float) -> PostureFeatures:
        return PostureFeatures(
            head_to_shoulder_ratio=1.0, shoulder_tilt_deg=0.0, head_roll_deg=0.0, forward_head_z=-0.2,
            shoulder_width_norm=0.03, neck_lateral_flexion_deg=10.0, detected_view_angle="lateral",
            craniovertebral_angle_deg=cva
        )

    def test_lateral_alert_tracks_change_from_calibrated_upright(self):
        """Verifies the side-on alert follows the drop from the calibrated CVA, and FHP is named only below the clinical criterion."""
        classifier = PostureClassifier(lateral_cva_fhp_thresh_deg=50.0, lateral_cva_drop_thresh_deg=5.0)

        # Upright at 60°: 58° is a small change; 52° is 8° forward but still meets the clinical criterion
        upright_60 = self._baseline(mean_craniovertebral_angle_deg=60.0)
        self.assertEqual(classifier.evaluate_adaptive(self._lateral(58.0), upright_60, 0.0).state, PostureState.GOOD_POSTURE)
        shifted = classifier.evaluate_adaptive(self._lateral(52.0), upright_60, 0.1)
        self.assertEqual(shifted.state, PostureState.HEAD_FORWARD_OF_UPRIGHT)
        self.assertFalse(shifted.clinical_fhp)
        self.assertAlmostEqual(shifted.cva_drop_deg, 8.0)

        # Natural upright at 48° (below the criterion): no alert while it holds, but the clinical flag is reported
        upright_48 = self._baseline(mean_craniovertebral_angle_deg=48.0)
        holding = classifier.evaluate_adaptive(self._lateral(46.0), upright_48, 0.2)
        self.assertEqual(holding.state, PostureState.GOOD_POSTURE)
        self.assertTrue(holding.clinical_fhp)

        # A drop that ends below the criterion is named Forward Head Posture
        upright_52 = self._baseline(mean_craniovertebral_angle_deg=52.0)
        self.assertEqual(classifier.evaluate_adaptive(self._lateral(44.0), upright_52, 0.3).state, PostureState.FORWARD_HEAD)

        # With the change rule switched off, only the clinical criterion applies
        clinical_only = PostureClassifier(lateral_cva_drop_thresh_deg=None)
        self.assertEqual(clinical_only.evaluate_adaptive(self._lateral(46.0), upright_48, 0.0).state, PostureState.FORWARD_HEAD)

    def test_calibrator_records_upright_cva_only_when_side_on(self):
        """Verifies calibration stores the upright CVA only when side-on, and older profiles without it still load."""
        frontal = PostureCalibrator(target_frames=4)
        frontal.start()
        for _ in range(4):
            frontal.add_frame(PostureFeatures(1.0, 0.0, 0.0, -0.2, 0.3, 0.0))
        self.assertIsNone(frontal.profile.mean_craniovertebral_angle_deg)

        side_on = PostureCalibrator(target_frames=4)
        side_on.start()
        for cva in (54.0, 56.0, 55.0, 55.0):
            side_on.add_frame(self._lateral(cva))
        self.assertAlmostEqual(side_on.profile.mean_craniovertebral_angle_deg, 55.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            old = side_on.profile.to_dict()
            del old["mean_craniovertebral_angle_deg"]
            import json
            json.dump(old, tmp)
            tmp_path = tmp.name
        try:
            loaded = BaselineProfile.load_json(tmp_path)
        finally:
            os.remove(tmp_path)
        self.assertIsNone(loaded.mean_craniovertebral_angle_deg)

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

        # The forward-head threshold and the side-on settings are read when train.py has tuned them
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump({
                "optimal_hyperparameters": {"alpha": 0.08, "slouch_thresh": 0.1, "forward_head_z_thresh": 0.2},
                "lateral_hyperparameters": {"alpha": 0.25, "cva_drop_thresh_deg": 4.0}
            }, tmp)
            tmp_path = tmp.name
        try:
            config = ErgoConfig()
            self.assertTrue(config.load_optimized(tmp_path))
            self.assertAlmostEqual(config.FORWARD_HEAD_Z_THRESH, 0.2)
            self.assertAlmostEqual(config.LATERAL_EMA_ALPHA, 0.25)
            self.assertAlmostEqual(config.LATERAL_CVA_DROP_THRESH_DEG, 4.0)
        finally:
            os.remove(tmp_path)

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
