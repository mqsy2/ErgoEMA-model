"""
Calibration Module for Online Live Fitting.
Captures user-specific upright posture baselines to eliminate morphological bias.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import json
import numpy as np
from .feature_extractor import PostureFeatures

@dataclass
class BaselineProfile:
    """User-specific personalized baseline profile."""
    mean_h2s_ratio: float
    std_h2s_ratio: float
    mean_shoulder_tilt_deg: float
    std_shoulder_tilt_deg: float
    mean_head_roll_deg: float
    std_head_roll_deg: float
    mean_forward_head_z: float
    std_forward_head_z: float
    mean_shoulder_width_norm: float
    sample_count: int
    user_id: str = "default_user"
    somatotype: Optional[str] = None  # Ectomorph, Mesomorph, Endomorph
    bmi_category: Optional[str] = None # Underweight, Normal, Overweight

    def to_dict(self) -> Dict[str, Any]:
        """Serializes baseline profile to dictionary."""
        return {
            "user_id": self.user_id,
            "somatotype": self.somatotype,
            "bmi_category": self.bmi_category,
            "sample_count": self.sample_count,
            "mean_h2s_ratio": round(self.mean_h2s_ratio, 4),
            "std_h2s_ratio": round(self.std_h2s_ratio, 4),
            "mean_shoulder_tilt_deg": round(self.mean_shoulder_tilt_deg, 2),
            "std_shoulder_tilt_deg": round(self.std_shoulder_tilt_deg, 2),
            "mean_head_roll_deg": round(self.mean_head_roll_deg, 2),
            "std_head_roll_deg": round(self.std_head_roll_deg, 2),
            "mean_forward_head_z": round(self.mean_forward_head_z, 4),
            "std_forward_head_z": round(self.std_forward_head_z, 4),
            "mean_shoulder_width_norm": round(self.mean_shoulder_width_norm, 4)
        }

    def save_json(self, filepath: str):
        """Saves profile to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, filepath: str) -> 'BaselineProfile':
        """Loads profile from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            user_id=data.get("user_id", "default_user"),
            somatotype=data.get("somatotype"),
            bmi_category=data.get("bmi_category"),
            sample_count=data.get("sample_count", 90),
            mean_h2s_ratio=data["mean_h2s_ratio"],
            std_h2s_ratio=data["std_h2s_ratio"],
            mean_shoulder_tilt_deg=data["mean_shoulder_tilt_deg"],
            std_shoulder_tilt_deg=data["std_shoulder_tilt_deg"],
            mean_head_roll_deg=data["mean_head_roll_deg"],
            std_head_roll_deg=data["std_head_roll_deg"],
            mean_forward_head_z=data["mean_forward_head_z"],
            std_forward_head_z=data["std_forward_head_z"],
            mean_shoulder_width_norm=data["mean_shoulder_width_norm"]
        )

class PostureCalibrator:
    """
    Collects a 3-second live window of upright sitting frames,
    computing mean and standard deviation for personalized thresholding.
    """

    def __init__(self, target_frames: int = 90, user_id: str = "default_user"):
        self.target_frames = target_frames
        self.user_id = user_id
        self._samples: List[PostureFeatures] = []
        self._is_calibrating: bool = False
        self._is_completed: bool = False
        self._profile: Optional[BaselineProfile] = None

    def start(self, user_id: Optional[str] = None):
        """Initiates a new calibration session."""
        if user_id:
            self.user_id = user_id
        self._samples.clear()
        self._is_calibrating = True
        self._is_completed = False
        self._profile = None

    def add_frame(self, features: Optional[PostureFeatures]) -> Tuple[bool, float]:
        """
        Appends a valid frame's features during calibration.
        Returns (is_completed, progress_percentage_0_to_1).
        """
        if not self._is_calibrating or features is None:
            return self._is_completed, self.progress

        self._samples.append(features)

        if len(self._samples) >= self.target_frames:
            self._finalize()
            return True, 1.0

        return False, self.progress

    def _finalize(self):
        """Computes statistical baseline parameters across captured frames."""
        if not self._samples:
            return

        ratios = [s.head_to_shoulder_ratio for s in self._samples]
        shoulder_tilts = [s.shoulder_tilt_deg for s in self._samples]
        head_rolls = [s.head_roll_deg for s in self._samples]
        z_depths = [s.forward_head_z for s in self._samples]
        widths = [s.shoulder_width_norm for s in self._samples]

        self._profile = BaselineProfile(
            user_id=self.user_id,
            sample_count=len(self._samples),
            mean_h2s_ratio=float(np.mean(ratios)),
            std_h2s_ratio=float(np.std(ratios)),
            mean_shoulder_tilt_deg=float(np.mean(shoulder_tilts)),
            std_shoulder_tilt_deg=float(np.std(shoulder_tilts)),
            mean_head_roll_deg=float(np.mean(head_rolls)),
            std_head_roll_deg=float(np.std(head_rolls)),
            mean_forward_head_z=float(np.mean(z_depths)),
            std_forward_head_z=float(np.std(z_depths)),
            mean_shoulder_width_norm=float(np.mean(widths))
        )
        self._is_calibrating = False
        self._is_completed = True

    @property
    def progress(self) -> float:
        """Returns progress ratio between 0.0 and 1.0."""
        if self._is_completed:
            return 1.0
        return min(1.0, len(self._samples) / float(self.target_frames))

    @property
    def is_calibrating(self) -> bool:
        return self._is_calibrating

    @property
    def is_completed(self) -> bool:
        return self._is_completed

    @property
    def profile(self) -> Optional[BaselineProfile]:
        return self._profile
