"""
Lateral Feature Extractor for 90° Side-Profile Posture Evaluation.
Computes spine curvature angles from 4 spinal keypoints (Cervical, Thoracic, Lumbar, Sacral).
Used exclusively for the Kaggle 'Posture Keypoints Detection' side-profile dataset.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np


@dataclass
class SpineKeypoints:
    """Raw 2D spine keypoints from YOLO-format annotations (normalized 0-1)."""
    cervical_x: float   # Beginning of cervical spine (top of neck)
    cervical_y: float
    cervical_vis: float
    thoracic_x: float   # Beginning of thoracic spine (upper back)
    thoracic_y: float
    thoracic_vis: float
    lumbar_x: float      # Beginning of lumbar spine (lower back)
    lumbar_y: float
    lumbar_vis: float
    sacral_x: float      # Beginning of sacral spine (tailbone)
    sacral_y: float
    sacral_vis: float


@dataclass
class LateralPostureFeatures:
    """Computed lateral posture features from spine keypoints."""
    cervical_thoracic_angle_deg: float   # Angle at thoracic joint (neck-to-upper-back curvature)
    thoracic_lumbar_angle_deg: float     # Angle at lumbar joint (upper-to-lower-back curvature)
    spine_vertical_deviation_deg: float  # Overall spine deviation from true vertical
    cervical_forward_offset: float       # Horizontal offset of cervical relative to thoracic (FHP indicator)
    spine_length_norm: float             # Total spine length (cervical to sacral), normalized
    label: Optional[str] = None          # Auto-assigned label: "upright", "slouch", or "forward_head"

    def to_dict(self) -> Dict[str, float]:
        """Converts to dictionary for CSV export."""
        return {
            "cervical_thoracic_angle_deg": round(self.cervical_thoracic_angle_deg, 2),
            "thoracic_lumbar_angle_deg": round(self.thoracic_lumbar_angle_deg, 2),
            "spine_vertical_deviation_deg": round(self.spine_vertical_deviation_deg, 2),
            "cervical_forward_offset": round(self.cervical_forward_offset, 4),
            "spine_length_norm": round(self.spine_length_norm, 4),
            "label": self.label or "unknown"
        }


class LateralFeatureExtractor:
    """
    Extracts biomechanical posture features from 4 lateral spine keypoints.
    
    Spine Geometry (side view, image coordinates where y=0 is top):
        Cervical (C) ---- top of neck
             |
        Thoracic (T) ---- upper back
             |
        Lumbar (L) ------- lower back
             |
        Sacral (S) ------- tailbone
    
    A perfectly upright spine forms a near-vertical line.
    Slouching causes the cervical-thoracic angle to deviate (head/neck moves forward).
    """

    # Thresholds for auto-labeling posture from spine angles
    SLOUCH_ANGLE_THRESHOLD_DEG: float = 155.0      # Below this = slouch (cervical-thoracic-lumbar angle)
    FHP_FORWARD_OFFSET_THRESHOLD: float = 0.03     # Cervical forward of thoracic by this normalized amount = FHP

    def _angle_between_three_points(
        self,
        ax: float, ay: float,
        bx: float, by: float,
        cx: float, cy: float
    ) -> float:
        """
        Computes the angle at point B formed by vectors BA and BC.
        Returns angle in degrees [0, 180].
        """
        vec_ba = np.array([ax - bx, ay - by])
        vec_bc = np.array([cx - bx, cy - by])

        norm_ba = np.linalg.norm(vec_ba)
        norm_bc = np.linalg.norm(vec_bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 180.0  # Degenerate case, treat as straight

        cos_angle = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def extract(self, keypoints: SpineKeypoints) -> Optional[LateralPostureFeatures]:
        """
        Computes lateral posture features from 4 spine keypoints.
        Returns LateralPostureFeatures or None if keypoints are invalid.
        """
        # Validate visibility - all 4 spine points should be visible
        if any(v < 0.5 for v in [
            keypoints.cervical_vis, keypoints.thoracic_vis,
            keypoints.lumbar_vis, keypoints.sacral_vis
        ]):
            return None

        # 1. Cervical-Thoracic Angle (angle at Thoracic, formed by Cervical-Thoracic-Lumbar)
        ct_angle = self._angle_between_three_points(
            keypoints.cervical_x, keypoints.cervical_y,
            keypoints.thoracic_x, keypoints.thoracic_y,
            keypoints.lumbar_x, keypoints.lumbar_y
        )

        # 2. Thoracic-Lumbar Angle (angle at Lumbar, formed by Thoracic-Lumbar-Sacral)
        tl_angle = self._angle_between_three_points(
            keypoints.thoracic_x, keypoints.thoracic_y,
            keypoints.lumbar_x, keypoints.lumbar_y,
            keypoints.sacral_x, keypoints.sacral_y
        )

        # 3. Overall Spine Vertical Deviation
        spine_dx = keypoints.cervical_x - keypoints.sacral_x
        spine_dy = keypoints.cervical_y - keypoints.sacral_y
        vertical_angle = float(np.degrees(np.arctan2(spine_dx, -spine_dy)))

        # 4. Cervical Forward Offset (FHP indicator)
        cervical_forward = keypoints.cervical_x - keypoints.thoracic_x

        # 5. Spine Length (Euclidean distance from cervical to sacral, normalized)
        spine_length = float(np.sqrt(
            (keypoints.cervical_x - keypoints.sacral_x) ** 2 +
            (keypoints.cervical_y - keypoints.sacral_y) ** 2
        ))

        # 6. Auto-label based on spine geometry
        label = self._classify_posture(ct_angle, cervical_forward)

        return LateralPostureFeatures(
            cervical_thoracic_angle_deg=ct_angle,
            thoracic_lumbar_angle_deg=tl_angle,
            spine_vertical_deviation_deg=vertical_angle,
            cervical_forward_offset=cervical_forward,
            spine_length_norm=spine_length,
            label=label
        )

    def _classify_posture(self, ct_angle: float, cervical_forward: float) -> str:
        """
        Auto-classifies posture based on spine curvature angles.
        
        Logic:
        - If cervical-thoracic-lumbar angle < threshold -> Slouch (Thoracic Kyphosis)
        - If cervical is significantly forward of thoracic -> Forward Head Posture (FHP)
        - Otherwise -> Upright
        """
        is_slouch = ct_angle < self.SLOUCH_ANGLE_THRESHOLD_DEG
        is_fhp = cervical_forward > self.FHP_FORWARD_OFFSET_THRESHOLD

        if is_slouch and is_fhp:
            return "slouch"
        elif is_fhp:
            return "forward_head"
        elif is_slouch:
            return "slouch"
        else:
            return "upright"
