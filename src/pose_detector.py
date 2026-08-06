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
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
    ):
        self.use_tasks_api = not hasattr(mp, 'solutions')
        
        if self.use_tasks_api:
            # Modern MediaPipe Tasks API (MediaPipe 0.10.20+)
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions

            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "pose_landmarker_full.task")

            if not os.path.exists(model_path):
                print("[INFO] Downloading MediaPipe PoseLandmarker model bundle...")
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
                urllib.request.urlretrieve(url, model_path)
                print(f"[INFO] Model saved to: {model_path}")

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=False
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.pose = None
        else:
            # Classic MediaPipe Solutions API
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity,
                enable_segmentation=enable_segmentation,
                static_image_mode=False
            )
            self.landmarker = None

    def extract_landmarks(self, frame_bgr: np.ndarray) -> Tuple[Optional[PoseLandmarks], Any]:
        """Processes a BGR video frame and extracts 3D upper-body landmarks."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.landmarker.detect(mp_image)

            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None, detection_result

            landmarks_list = detection_result.pose_landmarks[0]
            
            def to_point(idx: int) -> Point3D:
                p = landmarks_list[idx]
                vis = getattr(p, 'visibility', 1.0)
                if vis is None:
                    vis = 1.0
                return Point3D(x=float(p.x), y=float(p.y), z=float(p.z), visibility=float(vis))

            raw_ref = landmarks_list
        else:
            frame_rgb.flags.writeable = False
            results = self.pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if not results.pose_landmarks:
                return None, results

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

        # Check essential visibility
        essential_pts = [nose, left_shoulder, right_shoulder]
        is_valid = all(p.visibility > 0.35 for p in essential_pts)

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
            is_valid_upper_body=is_valid
        )

        return pose_data, raw_ref

    def draw_skeleton(self, frame_bgr: np.ndarray, results: Any, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Renders the skeletal keypoints and bone connections onto the frame."""
        if results is None:
            return frame_bgr

        h, w, _ = frame_bgr.shape
        landmarks = None

        if self.use_tasks_api:
            if isinstance(results, list):
                landmarks = results
            elif hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
        else:
            if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

        if landmarks is None:
            return frame_bgr

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
        if self.pose:
            self.pose.close()
        if self.landmarker:
            self.landmarker.close()
