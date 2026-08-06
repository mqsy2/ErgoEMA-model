"""
ErgoEMA Core Package
Adaptive Posture Monitoring using Time-Series Skeletal Data
"""

from .pose_detector import PoseDetector, PoseLandmarks
from .feature_extractor import FeatureExtractor, PostureFeatures
from .ema_filter import EMAFilter
from .calibrator import PostureCalibrator, BaselineProfile
from .classifier import PostureClassifier, PostureAssessment, PostureState
from .xai_explainer import XAIExplainer

__all__ = [
    "PoseDetector",
    "PoseLandmarks",
    "FeatureExtractor",
    "PostureFeatures",
    "EMAFilter",
    "PostureCalibrator",
    "BaselineProfile",
    "PostureClassifier",
    "PostureAssessment",
    "PostureState",
    "XAIExplainer",
]
