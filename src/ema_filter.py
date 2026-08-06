"""
Exponential Moving Average (EMA) Temporal Smoothing Filter.
Operates in constant O(1) time complexity with minimal memory footprint.
"""

from typing import Optional, Union
import numpy as np
from .feature_extractor import PostureFeatures

class EMAFilter:
    """
    Multi-channel recursive Exponential Moving Average filter.
    Mathematical Formula:
        EMA_t = alpha * X_t + (1 - alpha) * EMA_{t-1}
    """

    def __init__(self, alpha: float = 0.15):
        """
        Args:
            alpha: Smoothing factor in range (0.0, 1.0].
                   Higher values give more weight to recent measurements.
                   Lower values produce smoother, noise-dampened trends.
        """
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._smoothed_vector: Optional[np.ndarray] = None
        self._is_initialized: bool = False
        self._last_features: Optional[PostureFeatures] = None

    def update(self, features: Union[PostureFeatures, np.ndarray]) -> PostureFeatures:
        """
        Updates the EMA state with a new frame's feature vector in O(1) time.
        Returns the smoothed PostureFeatures instance.
        """
        if isinstance(features, PostureFeatures):
            x_t = features.to_array()
            shoulder_w = features.shoulder_width_norm
            ts = features.timestamp
        else:
            x_t = np.asarray(features, dtype=np.float64)
            shoulder_w = 1.0
            ts = None

        if not self._is_initialized or self._smoothed_vector is None:
            self._smoothed_vector = x_t.copy()
            self._is_initialized = True
        else:
            # Recursive EMA calculation
            self._smoothed_vector = (self.alpha * x_t) + ((1.0 - self.alpha) * self._smoothed_vector)

        smoothed_feat = PostureFeatures.from_array(self._smoothed_vector, shoulder_width=shoulder_w)
        smoothed_feat.timestamp = ts
        self._last_features = smoothed_feat
        return smoothed_feat

    def get_current(self) -> Optional[PostureFeatures]:
        """Returns the most recent smoothed features without updating."""
        return self._last_features

    def set_alpha(self, new_alpha: float):
        """Dynamically tunes the smoothing factor."""
        self.alpha = float(np.clip(new_alpha, 0.01, 1.0))

    def reset(self):
        """Resets filter memory."""
        self._smoothed_vector = None
        self._is_initialized = False
        self._last_features = None

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
