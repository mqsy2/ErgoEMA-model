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
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
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

        # Temporal stability and smoothing state for lateral spine rendering
        self._prev_vis_side = 'LEFT'
        self._prev_back_sign = -1.0
        self._prev_spine_pts = None

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
            is_valid_upper_body=is_valid
        )

        return pose_data, raw_ref

    def draw_skeleton(self, frame_bgr: np.ndarray, results: Any, color: Tuple[int, int, int] = (0, 255, 0), is_lateral: bool = False) -> np.ndarray:
        """
        Overlays the detected 3D skeleton onto the 2D video frame.
        If is_lateral is True, draws the 4 synthesized Kaggle spine keypoints.
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
            # === LATERAL MODE: Anatomical Spine Keypoints (Cervical, Thoracic, Lumbar, Sacral) ===
            # Identify the camera-facing side (left or right) with hysteresis
            nose = landmarks[0]
            l_ear, r_ear = landmarks[7], landmarks[8]
            l_sh, r_sh = landmarks[11], landmarks[12]
            l_hip, r_hip = landmarks[23], landmarks[24]
            
            # In MediaPipe 3D coordinates, camera-facing side has smaller (more negative) Z
            if hasattr(l_sh, 'z') and hasattr(r_sh, 'z') and abs(l_sh.z - r_sh.z) > 0.12:
                vis_side = 'LEFT' if l_sh.z < r_sh.z else 'RIGHT'
                self._prev_vis_side = vis_side
            else:
                l_vis = (getattr(l_ear, 'visibility', 0.0) or 0.0) + (getattr(l_sh, 'visibility', 0.0) or 0.0)
                r_vis = (getattr(r_ear, 'visibility', 0.0) or 0.0) + (getattr(r_sh, 'visibility', 0.0) or 0.0)
                if abs(l_vis - r_vis) > 0.2:
                    vis_side = 'LEFT' if l_vis >= r_vis else 'RIGHT'
                    self._prev_vis_side = vis_side
                else:
                    vis_side = self._prev_vis_side
            
            vis_ear = l_ear if vis_side == 'LEFT' else r_ear
            vis_sh = l_sh if vis_side == 'LEFT' else r_sh
            vis_hip = l_hip if vis_side == 'LEFT' else r_hip

            # Determine facing direction stably:
            # In lateral profile, nose extends forward of ear and shoulder in facing direction.
            # Blend nose-to-ear and nose-to-shoulder with deadband hysteresis:
            ear_dx = getattr(nose, 'x', vis_ear.x) - vis_ear.x
            sh_dx = getattr(nose, 'x', vis_sh.x) - vis_sh.x
            facing_score = 0.6 * ear_dx + 0.4 * sh_dx

            if abs(facing_score) > 0.02:
                # facing_score > 0: nose is to the right (+X) -> user faces RIGHT. Back is to the LEFT (-X, back_sign = -1.0).
                # facing_score < 0: nose is to the left (-X)  -> user faces LEFT. Back is to the RIGHT (+X, back_sign = +1.0).
                self._prev_back_sign = -1.0 if facing_score > 0 else 1.0
            back_sign = self._prev_back_sign

            # Anatomical scale reference in profile view
            head_depth = max(abs(ear_dx), 0.08)

            # Torso depth offset from shoulder to dorsal back contour
            sh_offset = np.clip(0.65 * head_depth, 0.07, 0.11)

            # 1. THORACIC SPINE (T1-T12):
            # Anchor to shoulder X shifted back toward dorsal contour
            sh_y = min(vis_sh.y, vis_ear.y + 0.28)
            t_norm_x = vis_sh.x + back_sign * sh_offset
            t_norm_y = sh_y + 0.03

            # 2. CERVICAL SPINE (C7):
            # Base of the neck, directly above T1, smoothly meeting posterior nape
            nape_x = vis_ear.x + back_sign * (0.50 * head_depth)
            c_norm_x = t_norm_x + 0.35 * (nape_x - t_norm_x)
            c_norm_y = max(0.02, t_norm_y - 0.12)

            # 3. SACRAL SPINE (S1-S5):
            # Lower pelvic anchor resting against chair back
            hip_vis = getattr(vis_hip, 'visibility', 0.0) or 0.0
            if hip_vis > 0.4 and vis_hip.y < 0.95:
                s_norm_x = vis_hip.x + back_sign * 0.04
                s_norm_y = min(0.95, vis_hip.y)
            else:
                # Extrapolate downward along dorsal torso contour
                s_norm_x = t_norm_x + back_sign * 0.06
                s_norm_y = min(0.95, t_norm_y + 0.30)

            # 4. LUMBAR SPINE (L1-L5):
            # Mid-point between Thoracic and Sacral
            l_norm_x = (t_norm_x + s_norm_x) / 2.0
            l_norm_y = (t_norm_y + s_norm_y) / 2.0

            # Clamp coordinates to frame boundaries
            c_x = int(np.clip(c_norm_x, 0.02, 0.98) * w)
            c_y = int(np.clip(c_norm_y, 0.02, 0.98) * h)
            t_x = int(np.clip(t_norm_x, 0.02, 0.98) * w)
            t_y = int(np.clip(t_norm_y, 0.02, 0.98) * h)
            l_x = int(np.clip(l_norm_x, 0.02, 0.98) * w)
            l_y = int(np.clip(l_norm_y, 0.02, 0.98) * h)
            s_x = int(np.clip(s_norm_x, 0.02, 0.98) * w)
            s_y = int(np.clip(s_norm_y, 0.02, 0.98) * h)

            raw_pts = np.array([[c_x, c_y], [t_x, t_y], [l_x, l_y], [s_x, s_y]], dtype=np.float32)

            # Temporal EMA Smoothing across video frames to prevent jitter and spinning
            if self._prev_spine_pts is None:
                self._prev_spine_pts = raw_pts
            else:
                self._prev_spine_pts = 0.35 * raw_pts + 0.65 * self._prev_spine_pts

            spine_pts = [(int(p[0]), int(p[1])) for p in self._prev_spine_pts]
            spine_labels = ["CERVICAL SPINE", "THORACIC", "LUMBAR", "SACRAL"]

            # Draw connecting spine line (yellow)
            for i in range(3):
                cv2.line(frame_bgr, spine_pts[i], spine_pts[i+1], (0, 255, 255), 2, cv2.LINE_AA)

            # Draw the 4 keypoints with labels
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # Red, Green, Blue, Yellow
            for i in range(4):
                pt = spine_pts[i]
                cv2.circle(frame_bgr, pt, 6, colors[i], -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)
                # Offset text toward body interior so it stays visible
                text_offset_x = 15 if back_sign < 0 else -180
                cv2.putText(frame_bgr, spine_labels[i], (pt[0] + text_offset_x, pt[1] + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, spine_labels[i], (pt[0] + text_offset_x, pt[1] + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            return frame_bgr

        # Reset spine smoothing when returning to frontal view
        self._prev_spine_pts = None

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
        if self.pose:
            self.pose.close()
        if self.landmarker:
            self.landmarker.close()
