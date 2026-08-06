"""
Configuration parameters for ErgoEMA Posture Monitoring Model.
Refined scope: Upright, Slouch (Thoracic Kyphosis), and Forward Head Posture (FHP).
"""

from dataclasses import dataclass

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
    
    # MediaPipe Pose Settings
    POSE_MIN_DETECTION_CONFIDENCE: float = 0.5
    POSE_MIN_TRACKING_CONFIDENCE: float = 0.5
    POSE_MODEL_COMPLEXITY: int = 1  # 0: Lite, 1: Full, 2: Heavy
    
    # Static Threshold Baseline Parameters (For benchmark comparison in paper)
    STATIC_SLOUCH_RATIO_FIXED: float = 0.55
    STATIC_FORWARD_HEAD_Z_FIXED: float = -0.15
    
    # Time window for alert confirmation (seconds of sustained bad posture before alert)
    SUSTAINED_ALERT_WINDOW_SEC: float = 1.0
