"""
Feature Extractor for Front-Facing Posture Biomechanics.
Calculates scale-invariant geometric ratios, angular alignments, and 3D depth indicators.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
from .pose_detector import PoseLandmarks, Point3D

@dataclass
class PostureFeatures:
    """Dataclass holding extracted frontal posture biomechanical features."""
    head_to_shoulder_ratio: float      # Scale-invariant vertical compression ratio (R_H2S)
    shoulder_tilt_deg: float           # Horizontal shoulder alignment angle (theta_shoulder)
    head_roll_deg: float               # Lateral head tilt / roll angle (theta_head)
    forward_head_z: float              # MediaPipe 3D sagittal head-to-shoulder displacement (Delta Z)
    shoulder_width_norm: float         # Biacromial pixel span in normalized coordinates
    neck_lateral_flexion_deg: float    # Angle between mid-shoulder-to-nose vector and true vertical
    timestamp: Optional[float] = None  # Video timestamp (if available)
    detected_view_angle: str = "frontal"  # Auto-detected: "frontal", "oblique", or "lateral"
    ear_shoulder_offset_x: float = 0.0    # Horizontal offset of ear relative to shoulder (lateral FHP metric)
    nose_shoulder_angle_deg: float = 0.0  # Angle of nose-to-shoulder vector vs vertical (lateral slouch metric)

    @property
    def is_lateral(self) -> bool:
        """Returns True if the detected view angle is lateral (90 degrees)."""
        return self.detected_view_angle == "lateral"

    def to_array(self) -> np.ndarray:
        """Returns the primary feature vector as a 1D NumPy array."""
        view_angle_code = 0.0
        if self.detected_view_angle == "oblique":
            view_angle_code = 1.0
        elif self.detected_view_angle == "lateral":
            view_angle_code = 2.0

        return np.array([
            self.head_to_shoulder_ratio,
            self.shoulder_tilt_deg,
            self.head_roll_deg,
            self.forward_head_z,
            self.neck_lateral_flexion_deg,
            self.ear_shoulder_offset_x,
            self.nose_shoulder_angle_deg,
            view_angle_code
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray, shoulder_width: float = 1.0) -> 'PostureFeatures':
        """Constructs PostureFeatures from a feature vector."""
        view_angle = "frontal"
        if len(arr) > 7:
            code = round(float(arr[7]))
            if code == 1:
                view_angle = "oblique"
            elif code >= 2:
                view_angle = "lateral"

        return cls(
            head_to_shoulder_ratio=float(arr[0]),
            shoulder_tilt_deg=float(arr[1]),
            head_roll_deg=float(arr[2]),
            forward_head_z=float(arr[3]),
            neck_lateral_flexion_deg=float(arr[4]) if len(arr) > 4 else 0.0,
            ear_shoulder_offset_x=float(arr[5]) if len(arr) > 5 else 0.0,
            nose_shoulder_angle_deg=float(arr[6]) if len(arr) > 6 else 0.0,
            detected_view_angle=view_angle,
            shoulder_width_norm=shoulder_width
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts to dictionary for logging and CSV export."""
        return {
            "head_to_shoulder_ratio": round(self.head_to_shoulder_ratio, 4),
            "shoulder_tilt_deg": round(self.shoulder_tilt_deg, 2),
            "head_roll_deg": round(self.head_roll_deg, 2),
            "forward_head_z": round(self.forward_head_z, 4),
            "neck_lateral_flexion_deg": round(self.neck_lateral_flexion_deg, 2),
            "shoulder_width_norm": round(self.shoulder_width_norm, 4)
        }

class FeatureExtractor:
    """Extracts scale-invariant geometric features from 3D skeletal landmarks.
    Supports both frontal and lateral (side-profile) views via automatic detection."""

    FEATURE_NAMES = [
        "Head-to-Shoulder Ratio",
        "Shoulder Tilt (deg)",
        "Head Roll (deg)",
        "Forward Head Z-depth",
        "Neck Lateral Flexion (deg)"
    ]

    # Shoulder width thresholds for automatic view angle detection (normalized coordinates)
    LATERAL_SHOULDER_WIDTH_THRESH: float = 0.06   # Below this = 90° lateral view
    OBLIQUE_SHOULDER_WIDTH_THRESH: float = 0.12   # Below this but above lateral = 45° oblique

    def extract(self, pose: PoseLandmarks, timestamp: Optional[float] = None) -> Optional[PostureFeatures]:
        """
        Computes front-facing posture metrics from landmarks.
        Returns PostureFeatures or None if critical points are invalid.
        """
        if not pose or not pose.is_valid_upper_body:
            return None

        ls = pose.left_shoulder
        rs = pose.right_shoulder
        nose = pose.nose
        le = pose.left_ear
        re = pose.right_ear
        leye = pose.left_eye
        reye = pose.right_eye

        # 1. Mid-Shoulder Centroid (M_S)
        mid_shoulder_x = (ls.x + rs.x) / 2.0
        mid_shoulder_y = (ls.y + rs.y) / 2.0
        mid_shoulder_z = (ls.z + rs.z) / 2.0

        # 2. Biacromial (Inter-Shoulder) Span in 2D
        shoulder_width = float(np.sqrt((rs.x - ls.x) ** 2 + (rs.y - ls.y) ** 2))
        
        # 3. Shoulder Horizontal Tilt Angle (theta_shoulder)
        # Using abs() on dx ensures the angle is bounded between -90 and +90 degrees,
        # making it robust against unmirrored camera inputs (where rs.x < ls.x).
        dx_sh = abs(rs.x - ls.x)
        dy_sh = rs.y - ls.y
        shoulder_tilt_rad = np.arctan2(dy_sh, dx_sh)
        shoulder_tilt_deg = float(np.degrees(shoulder_tilt_rad))

        if shoulder_width < 1e-4:
            # Shoulders completely overlapped — use the more visible one
            vis_shoulder = ls if ls.visibility > rs.visibility else rs
            return self._extract_lateral(pose, vis_shoulder.x, vis_shoulder.y, vis_shoulder.z, shoulder_width, timestamp)

        # Robust 3D Biomechanical View Angle Detection:
        # 1. In a 90° lateral profile view:
        #    - Cranium separates ears along camera optical axis (ear_z_diff > 0.08).
        #    - Torso width collapses (shoulder_width < 0.20) and shoulder_z_diff is large (> 0.20).
        # 2. In frontal or slight oblique view:
        #    - Both ears are roughly equidistant from camera (ear_z_diff < 0.06).
        ear_z_diff = abs(le.z - re.z)
        shoulder_z_diff = abs(ls.z - rs.z)
        
        is_lateral_view = False
        if ear_z_diff > 0.08:
            if shoulder_width < self.LATERAL_SHOULDER_WIDTH_THRESH:
                is_lateral_view = True
            elif shoulder_z_diff > 0.20 and (shoulder_z_diff / max(shoulder_width, 1e-4)) > 1.4:
                is_lateral_view = True
        elif shoulder_width < self.LATERAL_SHOULDER_WIDTH_THRESH and shoulder_z_diff > 0.25:
            is_lateral_view = True

        if is_lateral_view:
            # Use the camera-facing shoulder directly (via 3D Z-depth or visibility)
            # to avoid pulling the reference forward with the hallucinated occluded shoulder.
            if abs(ls.z - rs.z) > 0.08:
                vis_sh = ls if ls.z < rs.z else rs
            elif ls.visibility >= rs.visibility:
                vis_sh = ls
            else:
                vis_sh = rs
            return self._extract_lateral(pose, vis_sh.x, vis_sh.y, vis_sh.z, shoulder_width, timestamp)

        # 3. Normalized Head-to-Shoulder Vertical Compression Ratio (R_H2S)
        # In image coordinates, y=0 is top, y=1 is bottom.
        # Thus mid_shoulder_y > nose.y. Delta Y is positive in normal sitting.
        delta_y = mid_shoulder_y - nose.y
        h2s_ratio = float(delta_y / shoulder_width)

        # 4. Shoulder Horizontal Tilt Angle (theta_shoulder)
        # Already calculated above for lateral view detection: shoulder_tilt_deg

        # 5. Head Roll / Lateral Flexion Angle (theta_head)
        # Use ears if visible, else eyes. Use abs() on dx to bound angle between -90 and 90.
        if le.visibility > 0.4 and re.visibility > 0.4:
            dx_head = abs(re.x - le.x)
            dy_head = re.y - le.y
        else:
            dx_head = abs(reye.x - leye.x)
            dy_head = reye.y - leye.y
            
        head_roll_rad = np.arctan2(dy_head, dx_head)
        head_roll_deg = float(np.degrees(head_roll_rad))

        # 6. Sagittal Forward-Head 3D Depth Displacement (Delta Z)
        # Positive / negative shift relative to torso plane
        forward_head_z = float(nose.z - mid_shoulder_z)

        # 7. Neck Lateral Flexion (angle of mid-shoulder-to-nose vector vs vertical)
        # Vertical vector in image plane is [0, -1] (pointing up from mid-shoulder to head)
        vec_x = nose.x - mid_shoulder_x
        vec_y = nose.y - mid_shoulder_y # negative when head is above shoulder
        neck_angle_deg = float(np.degrees(np.arctan2(vec_x, -vec_y)))

        # Detect if this is an oblique view
        view_angle = "frontal"
        if shoulder_width < self.OBLIQUE_SHOULDER_WIDTH_THRESH:
            view_angle = "oblique"

        return PostureFeatures(
            head_to_shoulder_ratio=h2s_ratio,
            shoulder_tilt_deg=shoulder_tilt_deg,
            head_roll_deg=head_roll_deg,
            forward_head_z=forward_head_z,
            shoulder_width_norm=shoulder_width,
            neck_lateral_flexion_deg=neck_angle_deg,
            timestamp=timestamp,
            detected_view_angle=view_angle
        )

    def _extract_lateral(
        self,
        pose: PoseLandmarks,
        shoulder_ref_x: float,
        shoulder_ref_y: float,
        shoulder_ref_z: float,
        shoulder_width: float,
        timestamp: Optional[float] = None
    ) -> Optional[PostureFeatures]:
        """
        Computes lateral (90° side-profile) posture metrics.

        The shoulder_ref_x/y/z should be the VISIBLE (camera-facing) shoulder's
        coordinates, NOT the midpoint of both shoulders. In a 90° view, the
        occluded back shoulder is unreliably hallucinated by MediaPipe.

        Uses ear-to-shoulder horizontal displacement and nose-to-shoulder angle
        instead of the biacromial R_H2S ratio which breaks at 90°.
        """
        nose = pose.nose
        le = pose.left_ear
        re = pose.right_ear

        # Use whichever ear is more visible (in side view, only one ear faces the camera)
        visible_ear = None
        if le.visibility > re.visibility and le.visibility > 0.3:
            visible_ear = le
        elif re.visibility > 0.3:
            visible_ear = re

        # Determine facing direction: 1.0 if facing right (+X), -1.0 if facing left (-X)
        # Nose is horizontally forward of the shoulder in the direction the user faces
        facing_sign = 1.0 if nose.x >= shoulder_ref_x else -1.0

        # 1. Ear-to-Shoulder Horizontal Offset (primary FHP metric for lateral view)
        # Normalized by facing_sign so forward head displacement is ALWAYS positive
        ear_shoulder_offset_x = 0.0
        if visible_ear is not None:
            ear_shoulder_offset_x = float((visible_ear.x - shoulder_ref_x) * facing_sign)
        else:
            # Fallback: use nose horizontal offset
            ear_shoulder_offset_x = float((nose.x - shoulder_ref_x) * facing_sign)

        # 2. Nose-to-Shoulder Angle vs Vertical (primary slouch metric for lateral view)
        # A straight vertical line from shoulder to head = 0°. Forward lean increases this angle.
        # Normalized by facing_sign so forward lean angle is always positive
        vec_x = (nose.x - shoulder_ref_x) * facing_sign
        vec_y = nose.y - shoulder_ref_y  # negative when head is above shoulder
        nose_shoulder_angle = float(np.degrees(np.arctan2(vec_x, -vec_y)))

        # 3. Vertical distance normalized by a body-proportional reference
        # Use nose-to-shoulder vertical distance as a self-referencing scale
        delta_y = shoulder_ref_y - nose.y
        vertical_dist = max(abs(delta_y), 1e-4)
        # Create a pseudo H2S ratio using the vertical distance and horizontal offset
        lateral_h2s = float(delta_y / vertical_dist)  # Will be ~1.0 for upright, <1.0 for slouch

        # 4. Forward Head Z-depth (still available from MediaPipe 3D estimation)
        forward_head_z = float(nose.z - shoulder_ref_z)

        return PostureFeatures(
            head_to_shoulder_ratio=lateral_h2s,
            shoulder_tilt_deg=0.0,
            head_roll_deg=0.0,
            forward_head_z=forward_head_z,
            shoulder_width_norm=shoulder_width,
            neck_lateral_flexion_deg=nose_shoulder_angle,
            timestamp=timestamp,
            detected_view_angle="lateral",
            ear_shoulder_offset_x=ear_shoulder_offset_x,
            nose_shoulder_angle_deg=nose_shoulder_angle
        )
