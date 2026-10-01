"""
Kaggle Posture Keypoints Dataset Processor for ErgoEMA.

Processes the 'Posture Keypoints Detection - Photos & Labels' Kaggle dataset
(YOLO Pose format) into ErgoEMA-compatible CSV datasets for the 90° ablation study.

Pipeline:
1. Parses YOLO-format spine keypoint labels (cervical, thoracic, lumbar, sacral)
2. Runs each image through MediaPipe to extract standard ErgoEMA features (R_H2S, Delta_Z)
3. Computes lateral spine curvature angles using LateralFeatureExtractor
4. Auto-labels each image as 'upright', 'slouch', or 'forward_head' based on spine geometry
5. Exports a merged CSV for the ablation study
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Optional, Tuple

from config import ErgoConfig
from src.pose_detector import PoseDetector
from src.feature_extractor import FeatureExtractor
from src.lateral_feature_extractor import (
    LateralFeatureExtractor,
    SpineKeypoints,
    LateralPostureFeatures
)


def parse_yolo_label(label_path: str) -> Optional[SpineKeypoints]:
    """
    Parses a YOLO Pose format label file.
    
    Format: class_id cx cy w h kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis kp3_x kp3_y kp3_vis kp4_x kp4_y kp4_vis
    
    Keypoints:
        1: Cervical spine (top of neck)
        2: Thoracic spine (upper back)
        3: Lumbar spine (lower back)
        4: Sacral spine (tailbone)
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Use the first detection (first line) if multiple people annotated
        parts = lines[0].strip().split()
        
        # Expected: class_id(1) + bbox(4) + 4 keypoints * 3 values = 17 values
        if len(parts) < 17:
            return None
        
        # Skip class_id (0) and bbox (cx, cy, w, h) -> start at index 5
        kp_start = 5
        
        return SpineKeypoints(
            cervical_x=float(parts[kp_start]),
            cervical_y=float(parts[kp_start + 1]),
            cervical_vis=float(parts[kp_start + 2]),
            thoracic_x=float(parts[kp_start + 3]),
            thoracic_y=float(parts[kp_start + 4]),
            thoracic_vis=float(parts[kp_start + 5]),
            lumbar_x=float(parts[kp_start + 6]),
            lumbar_y=float(parts[kp_start + 7]),
            lumbar_vis=float(parts[kp_start + 8]),
            sacral_x=float(parts[kp_start + 9]),
            sacral_y=float(parts[kp_start + 10]),
            sacral_vis=float(parts[kp_start + 11]),
        )
    except Exception as e:
        print(f"  [WARN] Failed to parse label: {label_path} — {e}")
        return None


def process_kaggle_dataset(
    dataset_dir: str,
    output_dir: str = "data/processed",
    run_mediapipe: bool = True
):
    """
    Processes the entire Kaggle Posture Keypoints dataset.
    
    Args:
        dataset_dir: Path to the Kaggle dataset root (contains images/ and labels/)
        output_dir: Output directory for processed CSV
        run_mediapipe: If True, also runs MediaPipe on each image for front-facing features
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("   ErgoEMA Kaggle Posture Dataset Processor (90° Lateral Profile)")
    print("=" * 72)

    # Discover all images and labels
    image_dirs = []
    label_dirs = []
    for split in ["train", "val"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            image_dirs.append((split, img_dir, lbl_dir))

    if not image_dirs:
        print(f"[ERROR] No images/labels found in: {dataset_dir}")
        return

    # Initialize extractors
    lateral_extractor = LateralFeatureExtractor()
    
    mediapipe_detector = None
    mediapipe_extractor = None
    if run_mediapipe:
        config = ErgoConfig()
        mediapipe_detector = PoseDetector(
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            model_complexity=config.POSE_MODEL_COMPLEXITY
        )
        mediapipe_extractor = FeatureExtractor()

    all_records = []
    total_images = 0
    valid_lateral = 0
    valid_mediapipe = 0

    for split, img_dir, lbl_dir in image_dirs:
        image_files = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.jpeg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )
        
        print(f"\n[PROCESSING] {split.upper()} split: {len(image_files)} images")

        for img_path in image_files:
            total_images += 1
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(lbl_dir, f"{basename}.txt")

            record = {
                "image_name": basename,
                "split": split,
                "view_angle": "90deg",
                "subject_id": f"kaggle_{split}_{basename[:20]}",
                "somatotype": "Unknown",
                "bmi_category": "Unknown",
            }

            # 1. Parse YOLO spine keypoints and extract lateral features
            spine_kps = None
            lateral_feats = None
            if os.path.exists(label_path):
                spine_kps = parse_yolo_label(label_path)
                if spine_kps is not None:
                    lateral_feats = lateral_extractor.extract(spine_kps)

            if lateral_feats is not None:
                valid_lateral += 1
                record.update(lateral_feats.to_dict())
            else:
                record.update({
                    "cervical_thoracic_angle_deg": np.nan,
                    "thoracic_lumbar_angle_deg": np.nan,
                    "spine_vertical_deviation_deg": np.nan,
                    "cervical_forward_offset": np.nan,
                    "spine_length_norm": np.nan,
                    "label": "unknown",
                })

            # 2. Run MediaPipe on the image for standard ErgoEMA features
            # This tests if front-facing formulas work on side-view images (for ablation)
            if run_mediapipe and mediapipe_detector is not None:
                frame = cv2.imread(img_path)
                if frame is not None:
                    pose_data, _ = mediapipe_detector.extract_landmarks(frame, with_mask=True)
                    if pose_data and pose_data.is_valid_upper_body:
                        mp_feat = mediapipe_extractor.extract(pose_data)
                        if mp_feat is not None:
                            valid_mediapipe += 1
                            record["h2s_ratio"] = mp_feat.head_to_shoulder_ratio
                            record["shoulder_tilt_deg"] = mp_feat.shoulder_tilt_deg
                            record["head_roll_deg"] = mp_feat.head_roll_deg
                            record["forward_head_z"] = mp_feat.forward_head_z
                            record["neck_flexion_deg"] = mp_feat.neck_lateral_flexion_deg
                            record["shoulder_width_norm"] = mp_feat.shoulder_width_norm
                            record["ear_shoulder_offset_x"] = mp_feat.ear_shoulder_offset_x
                            record["nose_shoulder_angle_deg"] = mp_feat.nose_shoulder_angle_deg
                            record["craniovertebral_angle_deg"] = mp_feat.craniovertebral_angle_deg
                            record["detected_view_angle"] = mp_feat.detected_view_angle
                        else:
                            record.update(_empty_mediapipe_record())
                    else:
                        record.update(_empty_mediapipe_record())
                else:
                    record.update(_empty_mediapipe_record())
            else:
                record.update(_empty_mediapipe_record())

            all_records.append(record)

            if total_images % 50 == 0:
                print(f"    Processed {total_images} images...")

    # Close MediaPipe detector
    if mediapipe_detector:
        mediapipe_detector.close()

    # Create DataFrame
    df = pd.DataFrame(all_records)

    print(f"\n[SUMMARY]")
    print(f"  Total images processed: {total_images}")
    print(f"  Valid lateral spine extractions: {valid_lateral}")
    print(f"  Valid MediaPipe extractions: {valid_mediapipe}")
    
    # Show label distribution
    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        print(f"\n  Auto-Label Distribution:")
        for label, count in label_counts.items():
            print(f"    * {label}: {count} images")

    # Save the 90-degree dataset CSV
    kaggle_csv_path = os.path.join(output_dir, "kaggle_90deg_dataset.csv")
    df.to_csv(kaggle_csv_path, index=False)
    print(f"\n[SAVED] Kaggle 90° dataset: {kaggle_csv_path} ({len(df)} rows)")

    # Also create a MediaPipe-only version for the ablation study
    # (only rows where MediaPipe successfully extracted features)
    mp_cols = ["h2s_ratio", "shoulder_tilt_deg", "head_roll_deg", "forward_head_z", 
               "neck_flexion_deg", "shoulder_width_norm"]
    df_mp_valid = df.dropna(subset=mp_cols).copy()
    
    if len(df_mp_valid) > 0:
        # Update the all_angles_dataset.csv for the ablation study
        all_angles_path = os.path.join(output_dir, "all_angles_dataset.csv")
        
        if os.path.exists(all_angles_path):
            existing_df = pd.read_csv(all_angles_path)
            # Remove any old 90deg entries
            existing_df = existing_df[existing_df.get("view_angle", "Front") != "90deg"]
            combined_df = pd.concat([existing_df, df_mp_valid], ignore_index=True)
        else:
            combined_df = df_mp_valid
        
        combined_df.to_csv(all_angles_path, index=False)
        print(f"[SAVED] Updated all_angles_dataset.csv: {all_angles_path} ({len(combined_df)} total rows)")
        
        # Also save a standalone 90deg side view dataset for direct use
        side_90_path = os.path.join(output_dir, "side_90deg_dataset.csv")
        df_mp_valid.to_csv(side_90_path, index=False)
        print(f"[SAVED] 90° side-view dataset (MediaPipe): {side_90_path} ({len(df_mp_valid)} rows)")

    print(f"\n[DONE] Kaggle dataset processing complete.")
    return df


def _empty_mediapipe_record() -> Dict:
    """Returns NaN-filled MediaPipe feature columns."""
    return {
        "h2s_ratio": np.nan,
        "shoulder_tilt_deg": np.nan,
        "head_roll_deg": np.nan,
        "forward_head_z": np.nan,
        "neck_flexion_deg": np.nan,
        "shoulder_width_norm": np.nan,
        "ear_shoulder_offset_x": np.nan,
        "nose_shoulder_angle_deg": np.nan,
        "craniovertebral_angle_deg": np.nan,
        "detected_view_angle": np.nan,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Kaggle Posture Keypoints dataset for ErgoEMA 90° ablation study"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=r"C:\Users\Moises James Q. Sy\Documents\Datasets\Posture",
        help="Path to the Kaggle Posture Keypoints Detection dataset root"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed CSVs"
    )
    parser.add_argument(
        "--skip-mediapipe",
        action="store_true",
        help="Skip running MediaPipe on images (only extract lateral spine features)"
    )
    args = parser.parse_args()

    process_kaggle_dataset(
        dataset_dir=args.dataset,
        output_dir=args.output,
        run_mediapipe=not args.skip_mediapipe
    )
