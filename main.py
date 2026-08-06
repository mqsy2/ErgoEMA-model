"""
Main Real-Time Application for ErgoEMA.
Front-Facing Adaptive Posture Monitoring System.
"""

import time
import argparse
import cv2
import numpy as np

from config import ErgoConfig
from src.pose_detector import PoseDetector
from src.feature_extractor import FeatureExtractor
from src.ema_filter import EMAFilter
from src.calibrator import PostureCalibrator, BaselineProfile
from src.classifier import PostureClassifier, PostureState, PostureAssessment
from src.xai_explainer import XAIExplainer

def main():
    parser = argparse.ArgumentParser(description="ErgoEMA Live Real-Time Posture Monitor")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--alpha", type=float, default=0.15, help="EMA smoothing factor (default: 0.15)")
    parser.add_argument("--baseline", type=str, default=None, help="Path to pre-saved baseline JSON")
    args = parser.parse_args()

    config = ErgoConfig()
    config.CAMERA_INDEX = args.camera
    config.DEFAULT_EMA_ALPHA = args.alpha

    # Initialize Modules
    detector = PoseDetector(
        min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        model_complexity=config.POSE_MODEL_COMPLEXITY
    )
    extractor = FeatureExtractor()
    ema_filter = EMAFilter(alpha=config.DEFAULT_EMA_ALPHA)
    calibrator = PostureCalibrator(target_frames=int(config.CALIBRATION_DURATION_SECONDS * config.TARGET_FPS))
    classifier = PostureClassifier(
        slouch_ratio_drop_thresh=config.SLOUCH_RATIO_DROP_THRESH,
        forward_head_z_thresh=config.FORWARD_HEAD_Z_THRESH,
        sustained_window_sec=config.SUSTAINED_ALERT_WINDOW_SEC
    )
    explainer = XAIExplainer()

    # Load baseline if specified
    baseline: BaselineProfile = None
    if args.baseline:
        try:
            baseline = BaselineProfile.load_json(args.baseline)
            print(f"[INFO] Successfully loaded baseline profile from: {args.baseline}")
        except Exception as e:
            print(f"[WARN] Failed to load baseline: {e}")

    # Video Capture
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {config.CAMERA_INDEX}.")
        print("Please check your camera connection or specify another index using --camera <index>.")
        return

    print("\n" + "="*65)
    print("      ErgoEMA: Front-Facing Adaptive Posture Monitoring")
    print("="*65)
    print("Controls:")
    print("  [C] - Start 3-Second Upright Posture Calibration")
    print("  [S] - Save Calibrated Baseline to 'user_baseline.json'")
    print("  [L] - Load Baseline from 'user_baseline.json'")
    print("  [A] - Toggle Mode: Adaptive ErgoEMA vs. Static Baseline")
    print("  [Q] - Quit Application")
    print("="*65 + "\n")

    # FPS & Latency tracking
    prev_time = time.time()
    fps_smooth = 30.0
    use_adaptive_mode = True

    try:
        while cap.isOpened():
            t_frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to capture video frame.")
                break

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

            # 1. Pose Detection
            pose_data, results = detector.extract_landmarks(frame)

            raw_feat = None
            smoothed_feat = None
            assessment = None

            if pose_data and pose_data.is_valid_upper_body:
                # 2. Feature Extraction
                raw_feat = extractor.extract(pose_data, timestamp=now)

                if raw_feat is not None:
                    # 3. Calibration Handling
                    if calibrator.is_calibrating:
                        completed, progress = calibrator.add_frame(raw_feat)
                        if completed:
                            baseline = calibrator.profile
                            ema_filter.reset()
                            print("\n[SUCCESS] Calibration Complete! Baseline profile locked in:")
                            print(f"  - Baseline Head-to-Shoulder Ratio: {baseline.mean_h2s_ratio:.3f}")
                            print(f"  - Baseline Shoulder Tilt: {baseline.mean_shoulder_tilt_deg:.2f}°")
                            print(f"  - Baseline Head Roll: {baseline.mean_head_roll_deg:.2f}°")
                            print(f"  - Baseline Forward Head Z: {baseline.mean_forward_head_z:.3f}\n")
                    
                    # 4. Active Classification
                    elif baseline is not None:
                        if use_adaptive_mode:
                            # EMA Temporal Smoothing
                            smoothed_feat = ema_filter.update(raw_feat)
                            assessment = classifier.evaluate_adaptive(smoothed_feat, baseline, current_time=now)
                        else:
                            # Static Baseline Mode
                            assessment = classifier.evaluate_static_baseline(
                                raw_feat,
                                fixed_ratio_thresh=config.STATIC_SLOUCH_RATIO_FIXED,
                                fixed_fhp_z_thresh=config.STATIC_FORWARD_HEAD_Z_FIXED
                            )
                            smoothed_feat = raw_feat

            # Determine Skeleton Color based on state
            if calibrator.is_calibrating:
                skel_color = explainer.COLOR_CALIB
            elif assessment is None or assessment.state == PostureState.GOOD_POSTURE:
                skel_color = explainer.COLOR_GOOD
            elif not assessment.is_alert:
                skel_color = explainer.COLOR_WARNING
            else:
                skel_color = explainer.COLOR_ALERT

            # Draw Skeletal Lines
            if results is not None:
                detector.draw_skeleton(frame, results, color=skel_color)

            # Compute Execution Latency (ms)
            latency_ms = (time.perf_counter() - t_frame_start) * 1000.0

            # Render Explainable AI HUD
            frame = explainer.render_hud(
                frame=frame,
                assessment=assessment,
                features=smoothed_feat if smoothed_feat else raw_feat,
                baseline=baseline,
                fps=fps_smooth,
                latency_ms=latency_ms,
                is_calibrating=calibrator.is_calibrating,
                calib_progress=calibrator.progress
            )

            # Mode Indicator Banner (Top Right)
            mode_str = "MODE: ADAPTIVE (ErgoEMA)" if use_adaptive_mode else "MODE: STATIC BASELINE"
            mode_col = (200, 200, 200) if use_adaptive_mode else (0, 165, 255)
            cv2.putText(frame, mode_str, (frame.shape[1] - 320, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_col, 2, cv2.LINE_AA)

            # Show Window
            cv2.imshow("ErgoEMA - Front-Facing Posture Monitor", frame)

            # Handle Keypress Events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # Q or ESC
                break
            elif key == ord('c') or key == ord('C'):
                print("[INFO] Starting 3-second upright posture calibration. Sit upright...")
                calibrator.start()
            elif key == ord('s') or key == ord('S'):
                if baseline is not None:
                    baseline.save_json("user_baseline.json")
                    print("[INFO] Saved baseline profile to 'user_baseline.json'")
                else:
                    print("[WARN] No calibrated baseline to save. Press 'C' first.")
            elif key == ord('l') or key == ord('L'):
                try:
                    baseline = BaselineProfile.load_json("user_baseline.json")
                    ema_filter.reset()
                    print("[INFO] Loaded baseline profile from 'user_baseline.json'")
                except Exception as e:
                    print(f"[WARN] Failed to load 'user_baseline.json': {e}")
            elif key == ord('a') or key == ord('A'):
                use_adaptive_mode = not use_adaptive_mode
                print(f"[MODE SWITCH] Active Mode: {'ADAPTIVE (ErgoEMA)' if use_adaptive_mode else 'STATIC BASELINE'}")

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] ErgoEMA system shut down cleanly.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Session interrupted by user. Exited cleanly.")
