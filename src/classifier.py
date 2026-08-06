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

class PostureClassifier:
    """Adaptive and Static threshold posture classifier for Upright, Slouch, and FHP."""

    def __init__(
        self,
        slouch_ratio_drop_thresh: float = 0.12,      # 12% compression drop below baseline ratio
        forward_head_z_thresh: float = 0.06,         # Sagittal anterior depth shift deviation
        sustained_window_sec: float = 1.0            # Time buffer needed to confirm persistent bad posture
    ):
        self.slouch_ratio_drop_thresh = slouch_ratio_drop_thresh
        self.forward_head_z_thresh = forward_head_z_thresh
        self.sustained_window_sec = sustained_window_sec

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
        Focused strictly on Slouch (H2S compression) and Forward Head Posture (FHP Z-depth).
        """
        reasons: List[str] = []
        state = PostureState.GOOD_POSTURE

        # 1. Head-to-Shoulder Vertical Compression Ratio Drop (Slouch / Hunching)
        # Drop percentage: (baseline - current) / baseline
        ratio_drop_pct = (baseline.mean_h2s_ratio - smoothed_feat.head_to_shoulder_ratio) / max(1e-4, baseline.mean_h2s_ratio)
        
        # 2. Sagittal Forward-Head Z-depth Deviation (Anterior Translation)
        # When head moves closer to camera than shoulders compared to baseline
        z_dev = baseline.mean_forward_head_z - smoothed_feat.forward_head_z

        # Track active violations and relative severity
        active_violations = {}

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
            sustained_duration_sec=sustained_duration
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
