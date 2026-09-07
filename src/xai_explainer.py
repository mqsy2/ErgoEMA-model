"""
Explainable AI (XAI) Output Engine and Visual HUD Renderer.
Provides transparent, interpretable diagnostic feedback for Upright, Slouch, and FHP posture states.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from .classifier import PostureAssessment, PostureState
from .feature_extractor import PostureFeatures
from .calibrator import BaselineProfile

class XAIExplainer:
    """Generates human-readable ergonomic explanations and renders graphical HUD overlays."""

    # Colors (BGR)
    COLOR_GOOD = (70, 210, 80)        # Vibrant Green
    COLOR_WARNING = (0, 190, 255)     # Amber / Orange
    COLOR_ALERT = (50, 50, 230)       # Crimson Red
    COLOR_CALIB = (255, 180, 50)      # Cyan / Light Blue
    COLOR_BG = (20, 20, 24)           # Dark Slate Background
    COLOR_TEXT = (245, 245, 245)      # White
    COLOR_TEXT_DIM = (180, 180, 180)  # Light Gray
    COLOR_ACCENT = (147, 112, 219)    # Purple Accent

    def generate_explanation(self, assessment: PostureAssessment) -> str:
        """Returns a concise, plain-English ergonomic explanation of the current posture state."""
        if assessment.state == PostureState.GOOD_POSTURE:
            return "Good Posture: Neutral spine and head alignment within baseline limits."
        
        if assessment.reasons:
            return " | ".join(assessment.reasons)
        
        return f"Postural Deviation: {assessment.state.value}"

    def render_hud(
        self,
        frame: np.ndarray,
        assessment: Optional[PostureAssessment],
        features: Optional[PostureFeatures],
        baseline: Optional[BaselineProfile],
        fps: float,
        latency_ms: float,
        is_calibrating: bool = False,
        calib_progress: float = 0.0
    ) -> np.ndarray:
        """
        Renders a modern, semi-transparent ergonomic telemetry HUD directly onto the video frame.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Top Diagnostic Card
        card_w, card_h = min(480, w - 40), 160
        card_x, card_y = 20, 20
        
        # Draw translucent background card
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Header with Logo / System Name
        cv2.putText(frame, "ErgoEMA • Adaptive Posture Monitor", (card_x + 15, card_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_ACCENT, 2, cv2.LINE_AA)

        # System Metrics (FPS & Latency)
        perf_text = f"{fps:.1f} FPS | {latency_ms:.1f} ms"
        cv2.putText(frame, perf_text, (card_x + card_w - 140, card_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # State 1: Calibration Mode
        if is_calibrating:
            status_text = f"CALIBRATING BASELINE... {int(calib_progress * 100)}%"
            cv2.putText(frame, status_text, (card_x + 15, card_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_CALIB, 2, cv2.LINE_AA)
            
            # Calibration progress bar
            bar_w = card_w - 30
            bar_fill = int(bar_w * calib_progress)
            cv2.rectangle(frame, (card_x + 15, card_y + 80), (card_x + 15 + bar_w, card_y + 95), (60, 60, 60), -1)
            cv2.rectangle(frame, (card_x + 15, card_y + 80), (card_x + 15 + bar_fill, card_y + 95), self.COLOR_CALIB, -1)
            
            cv2.putText(frame, "Sit comfortably upright at eye level (~60 cm).", (card_x + 15, card_y + 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            return frame

        # State 2: Uncalibrated
        if baseline is None or assessment is None or features is None:
            cv2.putText(frame, "PRESS 'C' TO START CALIBRATION", (card_x + 15, card_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_WARNING, 2, cv2.LINE_AA)
            cv2.putText(frame, "Sit upright facing camera, then press 'C'.", (card_x + 15, card_y + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            return frame

        # State 3: Active Monitoring
        if assessment.state == PostureState.GOOD_POSTURE:
            badge_color = self.COLOR_GOOD
            status_title = "GOOD POSTURE (UPRIGHT)"
        elif not assessment.is_alert:
            badge_color = self.COLOR_WARNING
            status_title = f"DETECTING ({assessment.sustained_duration_sec:.1f}s)"
        else:
            badge_color = self.COLOR_ALERT
            status_title = f"ALERT: {assessment.state.value.upper()}"

        # Status Badge
        cv2.rectangle(frame, (card_x + 15, card_y + 40), (card_x + 15 + 10, card_y + 65), badge_color, -1)
        cv2.putText(frame, status_title, (card_x + 35, card_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, badge_color, 2, cv2.LINE_AA)

        if features.is_lateral:
            # Lateral Mode Telemetry Row 1: Nose-Shoulder Angle (Slouch Metric)
            nose_angle = features.nose_shoulder_angle_deg
            slouch_str = f"Nose-Shoulder Angle: {nose_angle:.1f}° (Slouch threshold: 15.0°)"
            cv2.putText(frame, slouch_str, (card_x + 15, card_y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_TEXT, 1, cv2.LINE_AA)

            # Lateral Mode Telemetry Row 2: Ear Offset (FHP Metric)
            ear_offset = features.ear_shoulder_offset_x
            fhp_str = f"Ear-Shoulder Offset: {ear_offset:.3f} (FHP threshold: 0.08)"
            cv2.putText(frame, fhp_str, (card_x + 15, card_y + 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_TEXT, 1, cv2.LINE_AA)
        else:
            # Frontal Mode Telemetry Row 1: Head-to-Shoulder Ratio (Slouch Metric)
            ratio_cur = features.head_to_shoulder_ratio
            ratio_base = baseline.mean_h2s_ratio
            ratio_drop_pct = ((ratio_base - ratio_cur) / max(1e-4, ratio_base)) * 100.0
            ratio_str = f"Head Ratio: {ratio_cur:.2f} (Base: {ratio_base:.2f}) [Drop: {max(0.0, ratio_drop_pct):.1f}%]"
            cv2.putText(frame, ratio_str, (card_x + 15, card_y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_TEXT, 1, cv2.LINE_AA)

            # Frontal Mode Telemetry Row 2: Forward Head Posture Z-Depth Metric (FHP)
            z_cur = features.forward_head_z
            z_base = baseline.mean_forward_head_z
            z_shift = z_base - z_cur
            fhp_str = f"Forward Head (FHP): Shift {z_shift:+.3f} (Z: {z_cur:.3f}, Base: {z_base:.3f})"
            cv2.putText(frame, fhp_str, (card_x + 15, card_y + 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_TEXT, 1, cv2.LINE_AA)

        # XAI Explanation Footer on Card
        explanation = self.generate_explanation(assessment)
        if len(explanation) > 55:
            explanation = explanation[:52] + "..."
        cv2.putText(frame, f"XAI: {explanation}", (card_x + 15, card_y + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Draw Persistent Bottom Banner during Active Sustained Alert
        if assessment.is_alert:
            banner_h = 50
            banner_overlay = frame.copy()
            cv2.rectangle(banner_overlay, (0, h - banner_h), (w, h), self.COLOR_ALERT, -1)
            cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
            
            alert_msg = f"⚠ ERGONOMIC ALERT: {assessment.state.value} detected. Please adjust posture!"
            cv2.putText(frame, alert_msg, (w // 2 - 280, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
