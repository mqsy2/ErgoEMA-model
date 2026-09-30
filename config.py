"""
Configuration parameters for ErgoEMA Posture Monitoring Model.
Refined scope: Upright, Slouch (Thoracic Kyphosis), and Forward Head Posture (FHP).
"""

import os
import json
from dataclasses import dataclass

# Hyperparameters tuned by train.py
OPTIMIZED_PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "optimized_parameters.json")

@dataclass
class ErgoConfig:
    # Camera settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1920
    FRAME_HEIGHT: int = 1080
    TARGET_FPS: int = 30
    
    # Calibration settings
    CALIBRATION_DURATION_SECONDS: float = 3.0   # 3 seconds calibration buffer
    CALIBRATION_MIN_FRAMES: int = 60            # Minimum valid frames required
    
    # EMA Smoothing Factor (alpha in [0.0, 1.0])
    DEFAULT_EMA_ALPHA: float = 0.25
    
    # Adaptive Classification Thresholds (Relative deviations from baseline)
    # Head-to-Shoulder Vertical Compression ratio drop threshold (e.g. 12% drop)
    SLOUCH_RATIO_DROP_THRESH: float = 0.12
    
    # Forward Head Jutting / Sagittal Z-displacement threshold (relative units)
    FORWARD_HEAD_Z_THRESH: float = 0.06
    
    # MediaPipe Pose Settings (0.35 accommodates single-side profile occlusion in lateral view)
    POSE_MIN_DETECTION_CONFIDENCE: float = 0.35
    POSE_MIN_TRACKING_CONFIDENCE: float = 0.35
    POSE_MODEL_COMPLEXITY: int = 1  # 0: Lite, 1: Full, 2: Heavy
    
    # Static Threshold Baseline Parameters (For benchmark comparison in paper)
    STATIC_SLOUCH_RATIO_FIXED: float = 0.55
    STATIC_FORWARD_HEAD_Z_FIXED: float = -0.15
    
    # Time window for alert confirmation (seconds of sustained bad posture before alert)
    SUSTAINED_ALERT_WINDOW_SEC: float = 1.0

    def load_optimized(self, path: str = OPTIMIZED_PARAMS_PATH) -> bool:
        """
        Overrides the defaults above with the hyperparameters tuned by train.py.
        Returns False (leaving the defaults untouched) if the file is missing or unreadable.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)["optimal_hyperparameters"]
            alpha = float(params["alpha"])
            slouch_thresh = float(params["slouch_thresh"])
        except (OSError, ValueError, KeyError, TypeError):
            return False

        self.DEFAULT_EMA_ALPHA = alpha
        self.SLOUCH_RATIO_DROP_THRESH = slouch_thresh
        return True
