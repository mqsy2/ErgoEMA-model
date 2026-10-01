"""
Posture Classifier Module.
Evaluates smoothed EMA features against personalized baseline tolerances,
focused strictly on Upright, Slouch (Thoracic Kyphosis), and Forward Head Posture (FHP).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from .feature_extractor import PostureFeatures
from .calibrator import BaselineProfile

class PostureState(Enum):
    GOOD_POSTURE = "Upright"
    SLOUCH = "Slouch"
    FORWARD_HEAD = "Forward Head Posture (FHP)"
    HEAD_FORWARD_OF_UPRIGHT = "Head Forward of Your Upright"  # Lateral change alert while the CVA still meets the clinical criterion
    UNKNOWN = "Uncalibrated / Unknown"

@dataclass
class PostureAssessment:
    """Detailed diagnostic output for a single frame evaluation."""
    state: PostureState
    is_alert: bool
    reasons: List[str] = field(default_factory=list)
    h2s_ratio_deviation_pct: float = 0.0
    forward_head_z_deviation: float = 0.0
    sustained_duration_sec: float = 0.0
    clinical_fhp: Optional[bool] = None           # Lateral view: CVA below the clinical FHP criterion
    cva_drop_deg: Optional[float] = None          # Lateral view: degrees below the calibrated upright CVA

class PostureClassifier:
    """Adaptive and Static threshold posture classifier for Upright, Slouch, and FHP."""

    def __init__(
        self,
        slouch_ratio_drop_thresh: float = 0.12,      # 12% compression drop below baseline ratio
        forward_head_z_thresh: float = 0.06,         # Sagittal anterior depth shift deviation
        sustained_window_sec: float = 1.0,           # Time buffer needed to confirm persistent bad posture
        lateral_cva_fhp_thresh_deg: float = 50.0,    # Lateral view: craniovertebral angle below this = FHP
        lateral_cva_drop_thresh_deg: Optional[float] = 5.0  # Lateral view: CVA this far below the calibrated upright = alert (None: clinical criterion only)
    ):
        self.slouch_ratio_drop_thresh = slouch_ratio_drop_thresh
        self.forward_head_z_thresh = forward_head_z_thresh
        self.sustained_window_sec = sustained_window_sec
        self.lateral_cva_fhp_thresh_deg = lateral_cva_fhp_thresh_deg
        self.lateral_cva_drop_thresh_deg = lateral_cva_drop_thresh_deg

        self._bad_posture_start_time: Optional[float] = None
        self._last_state: PostureState = PostureState.GOOD_POSTURE

    def evaluate_adaptive(
        self,
        smoothed_feat: PostureFeatures,
        baseline: BaselineProfile,
        current_time: float
    ) -> PostureAssessment:
        """
        Evaluates smoothed features against personalized baseline profile.
        Automatically adapts evaluation strategy based on detected view angle:
        - Frontal/Oblique: Uses H2S compression ratio and Z-depth
        - Lateral (90°): Uses the craniovertebral angle (CVA), relative to the calibrated upright CVA when available
        """
        reasons: List[str] = []
        state = PostureState.GOOD_POSTURE
        active_violations = {}
        clinical_fhp = None
        cva_drop = None

        if smoothed_feat.is_lateral:
            # === LATERAL MODE (90° Side Profile) ===
            # Clinical criteria (expert review):
            #   - Forward Head Posture (FHP): craniovertebral angle (CVA) below ~50°.
            #   - Slouching (hyperkyphosis): thoracic kyphosis above 40° (normal: 20°-40°).
            # The CVA is estimated per frame (see sagittal_geometry). Thoracic kyphosis cannot be
            # measured from pose landmarks or the body silhouette, so slouch is not assessed in this view.
            # When calibration was done side-on, the alert tracks the drop from the user's own upright CVA,
            # because some people's natural upright posture already measures near the clinical criterion.
            # The state is named FHP only when the CVA is also below that criterion.
            cva = smoothed_feat.craniovertebral_angle_deg
            clinical_fhp = cva < self.lateral_cva_fhp_thresh_deg
            base_cva = baseline.mean_craniovertebral_angle_deg

            if base_cva is not None and self.lateral_cva_drop_thresh_deg is not None:
                cva_drop = base_cva - cva
                if cva_drop > self.lateral_cva_drop_thresh_deg:
                    lateral_state = PostureState.FORWARD_HEAD if clinical_fhp else PostureState.HEAD_FORWARD_OF_UPRIGHT
                    active_violations[lateral_state] = cva_drop / max(1e-4, self.lateral_cva_drop_thresh_deg)
                    reasons.append(
                        f"[LATERAL] Head {cva_drop:.1f}° further forward than your upright (CVA {cva:.1f}° vs {base_cva:.1f}°)"
                    )
                    if clinical_fhp:
                        reasons.append(f"CVA below the {self.lateral_cva_fhp_thresh_deg:.0f}° FHP criterion")
            elif clinical_fhp:
                active_violations[PostureState.FORWARD_HEAD] = self.lateral_cva_fhp_thresh_deg / max(1.0, cva)
                reasons.append(
                    f"[LATERAL] Craniovertebral angle {cva:.1f}° below the {self.lateral_cva_fhp_thresh_deg:.0f}° FHP criterion"
                )

            ratio_drop_pct = 0.0
            z_dev = self.lateral_cva_fhp_thresh_deg - cva  # Degrees below the FHP criterion (negative = within normal)

        else:
            # === FRONTAL / OBLIQUE MODE ===
            # 1. Head-to-Shoulder Vertical Compression Ratio Drop (Slouch / Hunching)
            ratio_drop_pct = (baseline.mean_h2s_ratio - smoothed_feat.head_to_shoulder_ratio) / max(1e-4, baseline.mean_h2s_ratio)

            # 2. Sagittal Forward-Head Z-depth Deviation (Anterior Translation)
            z_dev = baseline.mean_forward_head_z - smoothed_feat.forward_head_z

            if ratio_drop_pct > self.slouch_ratio_drop_thresh:
                active_violations[PostureState.SLOUCH] = ratio_drop_pct / max(1e-4, self.slouch_ratio_drop_thresh)
                reasons.append(
                    f"Head compressed {ratio_drop_pct * 100:.1f}% below baseline (Current: {smoothed_feat.head_to_shoulder_ratio:.2f}, Base: {baseline.mean_h2s_ratio:.2f})"
                )

            if z_dev > self.forward_head_z_thresh:
                active_violations[PostureState.FORWARD_HEAD] = z_dev / max(1e-4, self.forward_head_z_thresh)
                reasons.append(
                    f"Head jutted forward by {z_dev:+.3f} units (FHP detected)"
                )

        # Primary state is the one with highest relative severity exceeding threshold
        if active_violations:
            state = max(active_violations, key=active_violations.get)

        # Sustained posture tracking (filtering transient fidgets)
        sustained_duration = 0.0
        if state != PostureState.GOOD_POSTURE:
            if self._bad_posture_start_time is None:
                self._bad_posture_start_time = current_time
            sustained_duration = current_time - self._bad_posture_start_time
        else:
            self._bad_posture_start_time = None

        is_alert = (state != PostureState.GOOD_POSTURE) and (sustained_duration >= self.sustained_window_sec)

        return PostureAssessment(
            state=state,
            is_alert=is_alert,
            reasons=reasons,
            h2s_ratio_deviation_pct=ratio_drop_pct * 100.0,
            forward_head_z_deviation=z_dev,
            sustained_duration_sec=sustained_duration,
            clinical_fhp=clinical_fhp,
            cva_drop_deg=cva_drop
        )

    def evaluate_static_baseline(
        self,
        raw_feat: PostureFeatures,
        fixed_ratio_thresh: float = 0.55,
        fixed_fhp_z_thresh: float = -0.15
    ) -> PostureAssessment:
        """
        Evaluates raw features against rigid, uncalibrated static thresholds.
        Used as the control / baseline model in thesis experiments.
        """
        reasons: List[str] = []
        state = PostureState.GOOD_POSTURE
        active_violations = {}

        # Fixed rigid rules without calibration or EMA smoothing
        if raw_feat.head_to_shoulder_ratio < fixed_ratio_thresh:
            active_violations[PostureState.SLOUCH] = (fixed_ratio_thresh - raw_feat.head_to_shoulder_ratio) / max(1e-4, fixed_ratio_thresh)
            reasons.append(f"Static rule: Ratio {raw_feat.head_to_shoulder_ratio:.2f} < {fixed_ratio_thresh}")

        if raw_feat.forward_head_z < fixed_fhp_z_thresh:
            active_violations[PostureState.FORWARD_HEAD] = abs(raw_feat.forward_head_z - fixed_fhp_z_thresh)
            reasons.append(f"Static rule: FHP Z-depth {raw_feat.forward_head_z:.3f} < {fixed_fhp_z_thresh}")

        if active_violations:
            state = max(active_violations, key=active_violations.get)

        # Static model immediately alerts on any violation without temporal tolerance
        return PostureAssessment(
            state=state,
            is_alert=(state != PostureState.GOOD_POSTURE),
            reasons=reasons,
            h2s_ratio_deviation_pct=0.0,
            forward_head_z_deviation=0.0,
            sustained_duration_sec=0.0
        )
