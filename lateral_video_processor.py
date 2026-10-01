"""
Process 90° lateral pose videos into the ErgoEMA side_90deg_dataset format.

Reads MP4 videos from a directory, extracts MediaPipe pose features per frame,
parses posture labels from filenames (Upright/Slouch/FHP), and appends to the
existing lateral_video_90deg.csv, side_90deg_dataset.csv, and all_angles_dataset.csv.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ErgoConfig
from src.pose_detector import PoseDetector
from src.feature_extractor import FeatureExtractor

# Standard participant demographics mapping across the study
PARTICIPANT_PROFILES = {
    "Participant1": {"somatotype": "Mesomorph", "bmi": "Normal"},
    "Participant2": {"somatotype": "Mesomorph", "bmi": "Normal"},
    "Participant3": {"somatotype": "Endomorph", "bmi": "Normal"},
    "Participant4": {"somatotype": "Endomorph", "bmi": "Normal"},
    "Participant5": {"somatotype": "Endomorph", "bmi": "Normal"},
    "Participant6": {"somatotype": "Mesomorph", "bmi": "Normal"},
    "Participant7": {"somatotype": "Ectomorph", "bmi": "Normal"},
    "Participant8": {"somatotype": "Mesomorph", "bmi": "Normal"},
    "Participant9": {"somatotype": "Endomorph", "bmi": "Overweight"},
}


def parse_label_from_filename(filename: str) -> str:
    """Extracts posture label from video filename convention.
    
    Expected format: 'Participant N-Label.mp4' or 'Participant N-Label N.mp4'
    Examples: 'Participant 5-Slouch.mp4', 'Participant 5-Upright 2.mp4', 'Participant 1-FHP.mp4'
    """
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    
    if "slouch" in name:
        return "slouch"
    elif "fhp" in name or "forward" in name:
        return "forward_head"
    elif "upright" in name:
        return "upright"
    else:
        return "upright"


def parse_participant_from_filename(filename: str) -> str:
    """Extracts standardized participant ID (e.g. 'Participant5') from video filename."""
    name = os.path.splitext(os.path.basename(filename))[0]
    # Extract 'Participant N' or 'Pariticipant N'
    parts = name.split("-")
    p = parts[0].strip() if parts else name
    p = p.replace("Pariticipant", "Participant").replace(" ", "")
    return p


def process_lateral_videos(
    video_dir: str,
    output_dir: str = "data/processed",
    sample_every_n_frames: int = 1,
    z_score_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Processes all MP4 videos in video_dir and extracts lateral posture features.
    
    Args:
        video_dir: Directory containing 90-degree lateral pose MP4 videos
        output_dir: Output directory for CSVs
        sample_every_n_frames: Step size for sampling frames (default: 1, full frame rate)
        z_score_threshold: Z-score threshold for outlier removal
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    if not video_files:
        print(f"[ERROR] No video files found in: {video_dir}")
        return pd.DataFrame()
    
    video_files.sort()
    print("=" * 72)
    print("   ErgoEMA 90 Degree Lateral Video Processor")
    print("=" * 72)
    print(f"   Source Directory : {video_dir}")
    print(f"   Videos Found     : {len(video_files)}")
    print(f"   Frame Sampling   : every {sample_every_n_frames} frame(s)")
    print()
    
    # Initialize MediaPipe
    config = ErgoConfig()
    detector = PoseDetector(
        min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        model_complexity=config.POSE_MODEL_COMPLEXITY
    )
    extractor = FeatureExtractor()
    
    all_records = []
    
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        label = parse_label_from_filename(video_name)
        participant_id = parse_participant_from_filename(video_name)
        subject_id = f"lateral_{participant_id}"
        
        profile = PARTICIPANT_PROFILES.get(participant_id, {"somatotype": "Mesomorph", "bmi": "Normal"})
        somatotype = profile["somatotype"]
        bmi_category = profile["bmi"]
        
        print(f"[{i:2d}/{len(video_files)}] Processing: {video_name}")
        print(f"       Participant: {participant_id} ({somatotype}, {bmi_category}) | Label: {label}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"       [ERROR] Could not open video: {video_path}")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_idx = 0
        valid_frames = 0
        skipped_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if sample_every_n_frames > 1 and (frame_idx % sample_every_n_frames != 0):
                frame_idx += 1
                continue
            
            timestamp = frame_idx / float(fps)
            pose_data, results = detector.extract_landmarks(frame, with_mask=True)

            if pose_data and pose_data.is_valid_upper_body:
                feat = extractor.extract(pose_data, timestamp=timestamp)
                if feat is not None:
                    valid_frames += 1
                    
                    record = {
                        "frame_idx": frame_idx,
                        "timestamp": round(timestamp, 4),
                        "image_name": f"{os.path.splitext(video_name)[0]}_frame{frame_idx:05d}",
                        "split": "train",
                        "view_angle": "90deg",
                        "subject_id": subject_id,
                        "participant_id": participant_id,
                        "somatotype": somatotype,
                        "bmi_category": bmi_category,
                        "cervical_thoracic_angle_deg": np.nan,
                        "thoracic_lumbar_angle_deg": np.nan,
                        "spine_vertical_deviation_deg": np.nan,
                        "cervical_forward_offset": np.nan,
                        "spine_length_norm": np.nan,
                        "label": label,
                        "h2s_ratio": feat.head_to_shoulder_ratio,
                        "shoulder_tilt_deg": feat.shoulder_tilt_deg,
                        "head_roll_deg": feat.head_roll_deg,
                        "forward_head_z": feat.forward_head_z,
                        "neck_flexion_deg": feat.neck_lateral_flexion_deg,
                        "shoulder_width_norm": feat.shoulder_width_norm,
                        "ear_shoulder_offset_x": feat.ear_shoulder_offset_x,
                        "nose_shoulder_angle_deg": feat.nose_shoulder_angle_deg,
                        "craniovertebral_angle_deg": feat.craniovertebral_angle_deg,
                        "detected_view_angle": feat.detected_view_angle,
                        "is_valid": 1
                    }
                    all_records.append(record)
                else:
                    skipped_frames += 1
            else:
                skipped_frames += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"       -> Valid frames: {valid_frames} (skipped: {skipped_frames})")
    
    detector.close()
    
    if not all_records:
        print("\n[ERROR] No valid frames extracted from any video.")
        return pd.DataFrame()
    
    df_new = pd.DataFrame(all_records)
    
    # Outlier removal (Z-score > threshold) per feature
    feature_cols = ["h2s_ratio", "shoulder_tilt_deg", "head_roll_deg", 
                    "forward_head_z", "neck_flexion_deg", "shoulder_width_norm",
                    "ear_shoulder_offset_x", "nose_shoulder_angle_deg", "craniovertebral_angle_deg"]
    
    outliers_removed = 0
    for col in feature_cols:
        valid_vals = df_new[col].dropna()
        if len(valid_vals) > 10:
            mean_val = np.mean(valid_vals)
            std_val = np.std(valid_vals)
            if std_val > 1e-6:
                z_scores = np.abs((df_new[col] - mean_val) / std_val)
                outlier_mask = z_scores > z_score_threshold
                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    df_new = df_new[~outlier_mask]
                    outliers_removed += n_outliers
    
    if outliers_removed > 0:
        print(f"\n[CLEANING] Removed {outliers_removed} outlier frames (Z > {z_score_threshold})")
    
    df_new = df_new.reset_index(drop=True)
    
    # Summary
    print(f"\n{'=' * 72}")
    print(f"[SUMMARY] Extracted {len(df_new):,} clean frames from {len(video_files)} videos")
    print("\n  Label distribution:")
    for label_name, count in df_new["label"].value_counts().items():
        print(f"    {label_name:<16}: {count:>5} frames")
    print("\n  View angle detection:")
    for va, count in df_new["detected_view_angle"].value_counts().items():
        print(f"    {va:<16}: {count:>5} frames")
    print("\n  Per-participant breakdown:")
    for pid, count in df_new["participant_id"].value_counts().items():
        sub = df_new[df_new["participant_id"] == pid]
        n_up = (sub["label"] == "upright").sum()
        n_sl = (sub["label"] == "slouch").sum()
        print(f"    {pid:<16}: {count:>5} frames ({n_up} upright, {n_sl} slouch)")
    
    # 1. Update lateral_video_90deg.csv
    lateral_csv = os.path.join(output_dir, "lateral_video_90deg.csv")
    if os.path.exists(lateral_csv):
        df_prev = pd.read_csv(lateral_csv, low_memory=False)
        # Update any legacy somatotype mappings in existing data
        for pid, prof in PARTICIPANT_PROFILES.items():
            mask = df_prev["subject_id"].astype(str).str.contains(pid)
            if mask.any():
                df_prev.loc[mask, "participant_id"] = pid
                df_prev.loc[mask, "somatotype"] = prof["somatotype"]
                df_prev.loc[mask, "bmi_category"] = prof["bmi"]
        df_lat_combined = pd.concat([df_prev, df_new], ignore_index=True)
        # Re-processing a video refreshes its rows
        df_lat_combined = df_lat_combined.drop_duplicates(subset=["image_name"], keep="last").reset_index(drop=True)
    else:
        df_lat_combined = df_new
    df_lat_combined.to_csv(lateral_csv, index=False)
    print(f"\n[SAVED] lateral_video_90deg.csv: {len(df_lat_combined):,} total frames")
    
    # 2. Merge into side_90deg_dataset.csv
    side_path = os.path.join(output_dir, "side_90deg_dataset.csv")
    if os.path.exists(side_path):
        df_existing = pd.read_csv(side_path, low_memory=False)
        # Update participant_id and demographics for any existing lateral entries
        for pid, prof in PARTICIPANT_PROFILES.items():
            mask = df_existing["subject_id"].astype(str).str.contains(pid)
            if mask.any():
                df_existing.loc[mask, "participant_id"] = pid
                df_existing.loc[mask, "somatotype"] = prof["somatotype"]
                df_existing.loc[mask, "bmi_category"] = prof["bmi"]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["image_name"], keep="last").reset_index(drop=True)
    else:
        df_combined = df_new
    df_combined.to_csv(side_path, index=False)
    print(f"[SAVED] side_90deg_dataset.csv: {len(df_combined):,} total frames")
    
    # 3. Merge into all_angles_dataset.csv (Union of Front 0°, Side 45°, and Lateral 90°)
    all_path = os.path.join(output_dir, "all_angles_dataset.csv")
    front_path = os.path.join(output_dir, "front_view_dataset.csv")
    side_45_path = os.path.join(output_dir, "side_view_dataset.csv")
    
    dfs_to_unite = []
    if os.path.exists(front_path):
        dfs_to_unite.append(pd.read_csv(front_path, low_memory=False))
    if os.path.exists(side_45_path):
        dfs_to_unite.append(pd.read_csv(side_45_path, low_memory=False))
    if os.path.exists(side_path):
        dfs_to_unite.append(pd.read_csv(side_path, low_memory=False))
        
    if dfs_to_unite:
        df_all_combined = pd.concat(dfs_to_unite, ignore_index=True)
    else:
        df_all_combined = df_new
    df_all_combined.to_csv(all_path, index=False)
    print(f"[SAVED] all_angles_dataset.csv: {len(df_all_combined):,} total frames")
    
    print(f"\n{'=' * 72}")
    print("[DONE] 90 degree lateral video processing and dataset merge complete.")
    print("=" * 72)
    
    return df_new


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process 90 degree lateral pose videos into ErgoEMA dataset"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=r"C:\Users\Moises James Q. Sy\Pictures\90 degree poses",
        help="Directory containing 90 degree lateral pose MP4 videos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed CSVs"
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Sample every Nth frame (default: 1, full frame rate)"
    )
    args = parser.parse_args()
    
    process_lateral_videos(
        video_dir=args.dir,
        output_dir=args.output,
        sample_every_n_frames=args.sample_every
    )
