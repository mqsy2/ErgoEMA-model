"""
Pose Detection Module supporting both MediaPipe Solutions (classic)
and MediaPipe Tasks Vision PoseLandmarker (modern 0.10.20+).
Extracts 33 3D skeletal landmarks from RGB video frames.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any, List
import numpy as np
import cv2
import mediapipe as mp
from .sagittal_geometry import ANALYSIS_HEIGHT, LateralGeometry, analyze_lateral

@dataclass
class Point3D:
    x: float
    y: float
    z: float
    visibility: float

@dataclass
class PoseLandmarks:
    raw_landmarks: Any
    nose: Point3D
    left_eye: Point3D
    right_eye: Point3D
    left_ear: Point3D
    right_ear: Point3D
    left_shoulder: Point3D
    right_shoulder: Point3D
    left_elbow: Optional[Point3D] = None
    right_elbow: Optional[Point3D] = None
    left_wrist: Optional[Point3D] = None
    right_wrist: Optional[Point3D] = None
    left_hip: Optional[Point3D] = None
    right_hip: Optional[Point3D] = None
    is_valid_upper_body: bool = False
    image_width: int = 0                              # Source frame size in pixels (0 if unknown)
    image_height: int = 0
    segmentation_mask: Optional[np.ndarray] = None    # Person probability mask, resized to ANALYSIS_HEIGHT rows

# Standard upper-body connections for skeletal drawing (pairs of landmark indices)
UPPER_BODY_CONNECTIONS = [
    (11, 12), # Left Shoulder - Right Shoulder
    (11, 13), # Left Shoulder - Left Elbow
    (13, 15), # Left Elbow - Left Wrist
    (12, 14), # Right Shoulder - Right Elbow
    (14, 16), # Right Elbow - Right Wrist
    (11, 23), # Left Shoulder - Left Hip
    (12, 24), # Right Shoulder - Right Hip
    (23, 24), # Left Hip - Right Hip
    (0, 1),   # Nose - Left Eye Inner
    (1, 2),   # Left Eye Inner - Left Eye
    (2, 3),   # Left Eye - Left Eye Outer
    (3, 7),   # Left Eye Outer - Left Ear
    (0, 4),   # Nose - Right Eye Inner
    (4, 5),   # Right Eye Inner - Right Eye
    (5, 6),   # Right Eye - Right Eye Outer
    (6, 8),   # Right Eye Outer - Right Ear
    (9, 10),  # Mouth Left - Mouth Right
]

class PoseDetector:
    """Robust upper-body skeletal tracker with multi-version MediaPipe support."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
    ):
        self.use_tasks_api = not hasattr(mp, 'solutions')
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_complexity = model_complexity
        self._always_segment = enable_segmentation

        if self.use_tasks_api:
            # Modern MediaPipe Tasks API (MediaPipe 0.10.20+)
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(model_dir, exist_ok=True)
            self._model_path = os.path.join(model_dir, "pose_landmarker_full.task")

            if not os.path.exists(self._model_path):
                print("[INFO] Downloading MediaPipe PoseLandmarker model bundle...")
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
                urllib.request.urlretrieve(url, self._model_path)
                print(f"[INFO] Model saved to: {self._model_path}")

            self.landmarker = self._create_landmarker(output_masks=False)
            self.pose = None
        else:
            # Classic MediaPipe Solutions API
            self.mp_pose = mp.solutions.pose
            self.pose = self._create_classic_pose(enable_segmentation=False)
            self.landmarker = None

        # Segmentation costs extra inference time, so the mask-producing model is only loaded when first needed
        self._mask_landmarker = None
        self._mask_pose = None

        # Temporal smoothing state for lateral overlay rendering
        self._prev_lateral_pts: Dict[str, np.ndarray] = {}

    def _create_landmarker(self, output_masks: bool):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_pose_presence_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_segmentation_masks=output_masks
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _create_classic_pose(self, enable_segmentation: bool):
        return self.mp_pose.Pose(
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            model_complexity=self._model_complexity,
            enable_segmentation=enable_segmentation,
            static_image_mode=False
        )

    @staticmethod
    def _shrink_mask(mask: np.ndarray) -> np.ndarray:
        """Resizes a full-frame person mask to ANALYSIS_HEIGHT rows, keeping the frame's aspect ratio."""
        h, w = mask.shape[:2]
        return cv2.resize(mask, (max(1, int(round(ANALYSIS_HEIGHT * w / h))), ANALYSIS_HEIGHT), interpolation=cv2.INTER_AREA)

    def extract_landmarks(self, frame_bgr: np.ndarray, with_mask: bool = False) -> Tuple[Optional[PoseLandmarks], Any]:
        """
        Processes a BGR video frame and extracts 3D upper-body landmarks.
        with_mask also returns the person segmentation mask, which side-profile analysis uses to locate C7.
        """
        h, w = frame_bgr.shape[:2]
        with_mask = with_mask or self._always_segment
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = None

        if self.use_tasks_api:
            landmarker = self.landmarker
            x_scale = 1.0
            if with_mask:
                if self._mask_landmarker is None:
                    self._mask_landmarker = self._create_landmarker(output_masks=True)
                landmarker = self._mask_landmarker
                # MediaPipe can only expose the float mask as an array when its rows are 16-byte aligned
                pad = (-w) % 4
                if pad:
                    frame_rgb = cv2.copyMakeBorder(frame_rgb, 0, 0, 0, pad, cv2.BORDER_REPLICATE)
                    x_scale = (w + pad) / w

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = landmarker.detect(mp_image)

            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None, detection_result

            landmarks_list = detection_result.pose_landmarks[0]
            if x_scale != 1.0:
                for p in landmarks_list:
                    p.x *= x_scale
            if with_mask and detection_result.segmentation_masks:
                mask = self._shrink_mask(detection_result.segmentation_masks[0].numpy_view()[:, :w])

            def to_point(idx: int) -> Point3D:
                p = landmarks_list[idx]
                vis = getattr(p, 'visibility', 1.0)
                if vis is None:
                    vis = 1.0
                return Point3D(x=float(p.x), y=float(p.y), z=float(p.z), visibility=float(vis))

            raw_ref = landmarks_list
        else:
            pose_model = self.pose
            if with_mask:
                if self._mask_pose is None:
                    self._mask_pose = self._create_classic_pose(enable_segmentation=True)
                pose_model = self._mask_pose

            frame_rgb.flags.writeable = False
            results = pose_model.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if not results.pose_landmarks:
                return None, results

            if with_mask and getattr(results, "segmentation_mask", None) is not None:
                mask = self._shrink_mask(results.segmentation_mask)

            lm = results.pose_landmarks.landmark
            def to_point(idx: int) -> Point3D:
                p = lm[idx]
                return Point3D(x=float(p.x), y=float(p.y), z=float(p.z), visibility=float(p.visibility))

            raw_ref = results.pose_landmarks

        # Map upper-body keypoints
        nose = to_point(0)
        left_eye = to_point(2)
        right_eye = to_point(5)
        left_ear = to_point(7)
        right_ear = to_point(8)
        left_shoulder = to_point(11)
        right_shoulder = to_point(12)

        # Check essential visibility (lenient for lateral views where one shoulder is occluded)
        has_nose = nose.visibility > 0.35
        has_left_shoulder = left_shoulder.visibility > 0.35
        has_right_shoulder = right_shoulder.visibility > 0.35
        is_valid = has_nose and (has_left_shoulder or has_right_shoulder)

        pose_data = PoseLandmarks(
            raw_landmarks=raw_ref,
            nose=nose,
            left_eye=left_eye,
            right_eye=right_eye,
            left_ear=left_ear,
            right_ear=right_ear,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            left_elbow=to_point(13),
            right_elbow=to_point(14),
            left_wrist=to_point(15),
            right_wrist=to_point(16),
            left_hip=to_point(23),
            right_hip=to_point(24),
            is_valid_upper_body=is_valid,
            image_width=w,
            image_height=h,
            segmentation_mask=mask
        )

        return pose_data, raw_ref

    @staticmethod
    def _lateral_geometry_from_landmarks(landmarks: Any, w: int, h: int) -> Optional[LateralGeometry]:
        """Side-profile geometry from raw landmarks alone (no silhouette)."""
        def to_point(idx: int) -> Point3D:
            p = landmarks[idx]
            vis = getattr(p, 'visibility', 1.0)
            return Point3D(x=float(p.x), y=float(p.y), z=float(getattr(p, 'z', 0.0) or 0.0), visibility=float(1.0 if vis is None else vis))

        pose = PoseLandmarks(
            raw_landmarks=landmarks,
            nose=to_point(0),
            left_eye=to_point(2),
            right_eye=to_point(5),
            left_ear=to_point(7),
            right_ear=to_point(8),
            left_shoulder=to_point(11),
            right_shoulder=to_point(12),
            left_hip=to_point(23),
            right_hip=to_point(24),
            image_width=w,
            image_height=h
        )
        # In MediaPipe 3D coordinates, the camera-facing side has the smaller (more negative) Z
        return analyze_lateral(pose, near_left=pose.left_shoulder.z < pose.right_shoulder.z)

    def _draw_lateral_overlay(self, frame_bgr: np.ndarray, geom: LateralGeometry):
        """Draws the measured back contour, spine-level markers and the craniovertebral angle construction."""
        h, w = frame_bgr.shape[:2]
        scale = np.array([w, h], dtype=np.float32)

        # Temporal EMA smoothing of marker positions across video frames to prevent jitter
        markers = {"EAR": geom.ear, "C7": geom.c7, **geom.spine_levels}
        self._prev_lateral_pts = {name: pt for name, pt in self._prev_lateral_pts.items() if name in markers}
        pts = {}
        for name, point in markers.items():
            raw = np.array(point, dtype=np.float32) * scale
            prev = self._prev_lateral_pts.get(name)
            self._prev_lateral_pts[name] = raw if prev is None else 0.35 * raw + 0.65 * prev
            pts[name] = (int(self._prev_lateral_pts[name][0]), int(self._prev_lateral_pts[name][1]))

        def put_label(text: str, anchor: Tuple[int, int]):
            # Offset text toward body interior so it stays visible
            text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            org = (anchor[0] + (15 if geom.facing > 0 else -15 - text_w), anchor[1] + 5)
            cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Back contour as measured on the body silhouette (yellow)
        if len(geom.back_contour) >= 2:
            contour = (np.array(geom.back_contour, dtype=np.float32) * scale).astype(np.int32)
            cv2.polylines(frame_bgr, [contour], False, (0, 255, 255), 2, cv2.LINE_AA)

        # Craniovertebral angle: horizontal through C7 vs. the line from C7 to the ear
        c7, ear = pts["C7"], pts["EAR"]
        reach = int(np.hypot(ear[0] - c7[0], ear[1] - c7[1]))
        cv2.line(frame_bgr, c7, (c7[0] + int(geom.facing) * reach, c7[1]), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame_bgr, c7, ear, (255, 255, 255), 2, cv2.LINE_AA)
        cva = geom.craniovertebral_angle_deg
        arc = (-cva, 0.0) if geom.facing > 0 else (180.0, 180.0 + cva)
        cv2.ellipse(frame_bgr, c7, (reach // 3, reach // 3), 0, arc[0], arc[1], (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame_bgr, ear, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, ear, 6, (0, 0, 0), 1, cv2.LINE_AA)

        # Spine-level markers on the contour (only the levels that are in view)
        colors = {"C7": (0, 0, 255), "THORACIC": (0, 255, 0), "LUMBAR": (255, 0, 0), "SACRAL": (0, 255, 255)}  # Red, Green, Blue, Yellow
        for name, color in colors.items():
            if name in pts:
                cv2.circle(frame_bgr, pts[name], 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pts[name], 7, (0, 0, 0), 1, cv2.LINE_AA)
                put_label(name, pts[name])

    def draw_skeleton(self, frame_bgr: np.ndarray, results: Any, color: Tuple[int, int, int] = (0, 255, 0), is_lateral: bool = False, lateral_geometry: Optional[LateralGeometry] = None) -> np.ndarray:
        """
        Overlays the detected 3D skeleton onto the 2D video frame.
        If is_lateral is True, draws the side-profile measurement overlay instead: the back contour,
        the estimated C7 and the craniovertebral angle. Pass the frame's lateral_geometry so the overlay
        matches the measured values; without it, C7 is placed from landmark proportions alone.
        """
        h, w = frame_bgr.shape[:2]
        landmarks = None
        if self.use_tasks_api:
            if isinstance(results, list):
                landmarks = results
            elif hasattr(results, 'pose_landmarks') and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]
            elif hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
        else:
            if hasattr(results, 'landmark'):
                landmarks = results.landmark

        if landmarks is None:
            return frame_bgr

        if is_lateral:
            # === LATERAL MODE: Back Contour, C7 and Craniovertebral Angle ===
            if lateral_geometry is None:
                lateral_geometry = self._lateral_geometry_from_landmarks(landmarks, w, h)
            if lateral_geometry is not None:
                self._draw_lateral_overlay(frame_bgr, lateral_geometry)
            return frame_bgr

        # Reset overlay smoothing when returning to frontal view
        self._prev_lateral_pts = {}

        # === FRONTAL MODE: Standard Upper Body Skeleton ===
        # Draw bone connections
        for idx1, idx2 in UPPER_BODY_CONNECTIONS:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                p1 = landmarks[idx1]
                p2 = landmarks[idx2]
                v1 = getattr(p1, 'visibility', 1.0) or 1.0
                v2 = getattr(p2, 'visibility', 1.0) or 1.0
                if v1 > 0.3 and v2 > 0.3:
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(frame_bgr, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw landmark circles
        for i, p in enumerate(landmarks):
            if i > 24:
                continue # Only upper-body landmarks
            v = getattr(p, 'visibility', 1.0) or 1.0
            if v > 0.3:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)

        return frame_bgr

    def close(self):
        """Releases MediaPipe resources."""
        for model in (self.pose, self.landmarker, self._mask_pose, self._mask_landmarker):
            if model:
                model.close()
