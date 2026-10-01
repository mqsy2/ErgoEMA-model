"""
Feature Extractor for Front-Facing Posture Biomechanics.
Calculates scale-invariant geometric ratios, angular alignments, and 3D depth indicators.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
from .pose_detector import PoseLandmarks, Point3D
from .sagittal_geometry import LateralGeometry, analyze_lateral

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
    ear_shoulder_offset_x: float = 0.0    # Horizontal offset of ear relative to shoulder, in frame widths (lateral view)
    nose_shoulder_angle_deg: float = 0.0  # Angle of nose-to-shoulder vector vs vertical (lateral view)
    craniovertebral_angle_deg: float = 0.0  # Estimated CVA: horizontal through C7 vs. the C7-to-ear line (lateral FHP metric)
    lateral_geometry: Optional[LateralGeometry] = field(default=None, repr=False, compare=False)  # Points behind the CVA, for the overlay

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
            view_angle_code,
            self.craniovertebral_angle_deg
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
            craniovertebral_angle_deg=float(arr[8]) if len(arr) > 8 else 0.0,
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
            return self._extract_lateral(pose, ls.visibility > rs.visibility, shoulder_width, timestamp)

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
                near_left = ls.z < rs.z
            else:
                near_left = ls.visibility >= rs.visibility
            return self._extract_lateral(pose, near_left, shoulder_width, timestamp)

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
        near_left: bool,
        shoulder_width: float,
        timestamp: Optional[float] = None
    ) -> Optional[PostureFeatures]:
        """
        Computes lateral (90° side-profile) posture metrics.

        near_left selects the VISIBLE (camera-facing) side. In a 90° view, the
        occluded back shoulder is unreliably hallucinated by MediaPipe, so the
        midpoint of both shoulders is not used.

        The primary metric is the craniovertebral angle (CVA), the clinical measure of
        forward head posture. The biacromial R_H2S ratio breaks at 90° and is not used.
        """
        geometry = analyze_lateral(pose, near_left)
        if geometry is None:
            return None

        nose = pose.nose
        shoulder = pose.left_shoulder if near_left else pose.right_shoulder
        ear_x = geometry.ear[0]
        facing_sign = geometry.facing  # 1.0 if facing right (+X), -1.0 if facing left (-X)

        # Angles are measured in pixel space: normalized x and y are not on the same scale unless the frame is square
        aspect = pose.image_width / pose.image_height if pose.image_width and pose.image_height else 1.0

        # 1. Ear-to-Shoulder Horizontal Offset
        # Normalized by facing_sign so forward head displacement is ALWAYS positive
        ear_shoulder_offset_x = float((ear_x - shoulder.x) * facing_sign)

        # 2. Nose-to-Shoulder Angle vs Vertical
        # A straight vertical line from shoulder to head = 0°. Forward lean increases this angle.
        # Normalized by facing_sign so forward lean angle is always positive
        vec_x = (nose.x - shoulder.x) * facing_sign * aspect
        vec_y = nose.y - shoulder.y  # negative when head is above shoulder
        nose_shoulder_angle = float(np.degrees(np.arctan2(vec_x, -vec_y)))

        # 3. Vertical distance normalized by a body-proportional reference
        # Use nose-to-shoulder vertical distance as a self-referencing scale
        delta_y = shoulder.y - nose.y
        vertical_dist = max(abs(delta_y), 1e-4)
        # Create a pseudo H2S ratio using the vertical distance and horizontal offset
        lateral_h2s = float(delta_y / vertical_dist)  # Will be ~1.0 for upright, <1.0 for slouch

        # 4. Forward Head Z-depth (still available from MediaPipe 3D estimation)
        forward_head_z = float(nose.z - shoulder.z)

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
            nose_shoulder_angle_deg=nose_shoulder_angle,
            craniovertebral_angle_deg=geometry.craniovertebral_angle_deg,
            lateral_geometry=geometry
        )
