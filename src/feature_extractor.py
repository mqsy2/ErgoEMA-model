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

    def to_array(self) -> np.ndarray:
        """Returns the primary feature vector as a 1D NumPy array."""
        return np.array([
            self.head_to_shoulder_ratio,
            self.shoulder_tilt_deg,
            self.head_roll_deg,
            self.forward_head_z,
            self.neck_lateral_flexion_deg
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray, shoulder_width: float = 1.0) -> 'PostureFeatures':
        """Constructs PostureFeatures from a feature vector."""
        return cls(
            head_to_shoulder_ratio=float(arr[0]),
            shoulder_tilt_deg=float(arr[1]),
            head_roll_deg=float(arr[2]),
            forward_head_z=float(arr[3]),
            neck_lateral_flexion_deg=float(arr[4]) if len(arr) > 4 else 0.0,
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
    """Extracts front-facing scale-invariant geometric features from 3D skeletal landmarks."""

    FEATURE_NAMES = [
        "Head-to-Shoulder Ratio",
        "Shoulder Tilt (deg)",
        "Head Roll (deg)",
        "Forward Head Z-depth",
        "Neck Lateral Flexion (deg)"
    ]

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
        if shoulder_width < 1e-4:
            return None

        # 3. Normalized Head-to-Shoulder Vertical Compression Ratio (R_H2S)
        # In image coordinates, y=0 is top, y=1 is bottom.
        # Thus mid_shoulder_y > nose.y. Delta Y is positive in normal sitting.
        delta_y = mid_shoulder_y - nose.y
        h2s_ratio = float(delta_y / shoulder_width)

        # 4. Shoulder Horizontal Tilt Angle (theta_shoulder)
        # Delta Y / Delta X between right and left shoulders
        # In upright posture, both shoulders are at the same horizontal height -> angle ~ 0 deg
        dx_sh = rs.x - ls.x
        dy_sh = rs.y - ls.y
        shoulder_tilt_rad = np.arctan2(dy_sh, dx_sh)
        shoulder_tilt_deg = float(np.degrees(shoulder_tilt_rad))

        # 5. Head Roll / Lateral Flexion Angle (theta_head)
        # Use ears if visible, else eyes
        if le.visibility > 0.4 and re.visibility > 0.4:
            dx_head = re.x - le.x
            dy_head = re.y - le.y
        else:
            dx_head = reye.x - leye.x
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

        return PostureFeatures(
            head_to_shoulder_ratio=h2s_ratio,
            shoulder_tilt_deg=shoulder_tilt_deg,
            head_roll_deg=head_roll_deg,
            forward_head_z=forward_head_z,
            shoulder_width_norm=shoulder_width,
            neck_lateral_flexion_deg=neck_angle_deg,
            timestamp=timestamp
        )
