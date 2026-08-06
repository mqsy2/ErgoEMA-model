"""
Dataset Collection Tool for Front-Facing Posture Benchmark.
Records time-series skeletal keypoints, calculated features, and ground-truth posture labels.
"""

import os
import time
import argparse
import csv
import cv2
import numpy as np
from config import ErgoConfig
from src.pose_detector import PoseDetector
from src.feature_extractor import FeatureExtractor

def run_collector(
    subject_id: str = "subject_01",
    somatotype: str = "Mesomorph",
    bmi_category: str = "Normal",
    output_dir: str = "data/raw"
):
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{subject_id}_posture_dataset.csv")

    config = ErgoConfig()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    detector = PoseDetector(
        min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        model_complexity=config.POSE_MODEL_COMPLEXITY
    )
    extractor = FeatureExtractor()

    # Posture Label Mapping
    labels = {
        ord('1'): "upright",
        ord('2'): "slouch",
        ord('3'): "left_lean",
        ord('4'): "right_lean",
        ord('5'): "forward_head",
        ord('0'): "micro_movement"
    }

    current_label = "upright"
    is_recording = False
    frame_count = 0

    csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # Header
    csv_writer.writerow([
        "timestamp", "subject_id", "somatotype", "bmi_category",
        "label", "h2s_ratio", "shoulder_tilt_deg", "head_roll_deg",
        "forward_head_z", "neck_flexion_deg", "shoulder_width_norm",
        "nose_x", "nose_y", "nose_z",
        "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
        "right_shoulder_x", "right_shoulder_y", "right_shoulder_z"
    ])

    print("\n" + "="*60)
    print("      ErgoEMA Front-Facing Dataset Collection Tool")
    print("="*60)
    print(f"Subject ID: {subject_id} | Somatotype: {somatotype} | BMI: {bmi_category}")
    print("Key Bindings to Set Active Posture Label:")
    print("  [1] - Upright (Neutral baseline)")
    print("  [2] - Slouch / Kyphosis")
    print("  [3] - Left Shoulder Drop / Lean")
    print("  [4] - Right Shoulder Drop / Lean")
    print("  [5] - Forward Head Jutting")
    print("  [0] - Natural Micro-movement / Fidget")
    print("Controls:")
    print("  [SPACE] - Toggle Recording ON/OFF")
    print("  [Q]     - Save & Quit")
    print("="*60 + "\n")

    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time() - start_time
            pose_data, results = detector.extract_landmarks(frame)

            feat = None
            if pose_data and pose_data.is_valid_upper_body:
                feat = extractor.extract(pose_data, timestamp=now)
                detector.draw_skeleton(frame, results, color=(0, 255, 0) if current_label == "upright" else (0, 165, 255))

                if is_recording and feat is not None:
                    frame_count += 1
                    csv_writer.writerow([
                        round(now, 4), subject_id, somatotype, bmi_category,
                        current_label,
                        feat.head_to_shoulder_ratio,
                        feat.shoulder_tilt_deg,
                        feat.head_roll_deg,
                        feat.forward_head_z,
                        feat.neck_lateral_flexion_deg,
                        feat.shoulder_width_norm,
                        pose_data.nose.x, pose_data.nose.y, pose_data.nose.z,
                        pose_data.left_shoulder.x, pose_data.left_shoulder.y, pose_data.left_shoulder.z,
                        pose_data.right_shoulder.x, pose_data.right_shoulder.y, pose_data.right_shoulder.z
                    ])

            # UI HUD Overlay
            rec_status = f"REC: [{frame_count} frames]" if is_recording else "PAUSED [Press SPACE]"
            rec_color = (0, 0, 255) if is_recording else (150, 150, 150)
            cv2.putText(frame, rec_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)
            cv2.putText(frame, f"Active Label: {current_label.upper()} (Keys 1-5, 0)", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Subject: {subject_id} ({somatotype}, {bmi_category})", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("ErgoEMA Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                is_recording = not is_recording
                print(f"[STATUS] Recording {'STARTED' if is_recording else 'PAUSED'}")
            elif key in labels:
                current_label = labels[key]
                print(f"[LABEL SWITCH] -> {current_label}")

    finally:
        csv_file.close()
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print(f"\n[SAVED] Dataset recording saved to: {csv_filename} ({frame_count} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record front-facing posture dataset")
    parser.add_argument("--subject", type=str, default="subject_01", help="Subject identifier")
    parser.add_argument("--somatotype", type=str, default="Mesomorph", choices=["Ectomorph", "Mesomorph", "Endomorph"])
    parser.add_argument("--bmi", type=str, default="Normal", choices=["Underweight", "Normal", "Overweight"])
    args = parser.parse_args()

    run_collector(subject_id=args.subject, somatotype=args.somatotype, bmi_category=args.bmi)
