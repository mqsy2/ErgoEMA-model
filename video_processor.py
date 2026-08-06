"""
Raw Video Processing and Data Cleaning Pipeline for ErgoEMA.
Converts raw video files into frames, extracts skeletal keypoints,
cleans noise/outliers (Z-score > 3), imputes missing frames,
and outputs clean, structured time-series datasets for training/testing.
"""

import os
import argparse
import glob
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import pandas as pd

from config import ErgoConfig
from src.pose_detector import PoseDetector
from src.feature_extractor import FeatureExtractor, PostureFeatures

class VideoDataPipeline:
    """
    Processes raw front-facing workstation video recordings into cleaned,
    standardized time-series ergonomic datasets.
    """

    def __init__(
        self,
        min_visibility: float = 0.5,
        z_score_threshold: float = 3.0,
        max_missing_consecutive_frames: int = 15
    ):
        self.config = ErgoConfig()
        self.min_visibility = min_visibility
        self.z_score_threshold = z_score_threshold
        self.max_missing_consecutive_frames = max_missing_consecutive_frames
        
        self.detector = PoseDetector(
            min_detection_confidence=self.config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.POSE_MIN_TRACKING_CONFIDENCE,
            model_complexity=self.config.POSE_MODEL_COMPLEXITY
        )
        self.extractor = FeatureExtractor()

    # Known study participant demographic mapping
    PARTICIPANT_PROFILES = {
        "bigid": {"somatotype": "Endomorph", "bmi": "Overweight"},
        "jasper": {"somatotype": "Mesomorph", "bmi": "Normal"},
        "gab": {"somatotype": "Ectomorph", "bmi": "Normal"},
        "margot": {"somatotype": "Mesomorph", "bmi": "Normal"}
    }

    def _resolve_demographics(self, subject_id: str, default_somatotype: str, default_bmi: str) -> Tuple[str, str]:
        """Auto-resolves somatotype and BMI from participant name if matched."""
        sub_lower = subject_id.lower()
        for key, profile in self.PARTICIPANT_PROFILES.items():
            if key in sub_lower:
                return profile["somatotype"], profile["bmi"]
        return default_somatotype, default_bmi

    def process_video(
        self,
        video_path: str,
        subject_id: str,
        somatotype: str = "Mesomorph",
        bmi_category: str = "Normal",
        label: str = "auto",
        save_annotated_video: bool = False,
        save_frames: bool = False,
        output_dir: str = "data/processed"
    ) -> pd.DataFrame:
        """
        Extracts frames from video, computes pose features, cleans data, and exports CSV.
        """
        # Automatically lookup participant somatotype and BMI
        somatotype, bmi_category = self._resolve_demographics(subject_id, somatotype, bmi_category)

        os.makedirs(output_dir, exist_ok=True)
        frames_out_dir = None
        if save_frames:
            frames_out_dir = os.path.join("data", "frames", subject_id)
            os.makedirs(frames_out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n[VIDEO PIPELINE] Processing: {os.path.basename(video_path)}")
        print(f"  - Total Frames: {total_frames} | FPS: {fps:.2f} | Resolution: {width}x{height}")

        video_writer = None
        if save_annotated_video:
            annotated_path = os.path.join(output_dir, f"{subject_id}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        raw_records = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / float(fps)
            pose_data, results = self.detector.extract_landmarks(frame)

            if pose_data and pose_data.is_valid_upper_body:
                feat = self.extractor.extract(pose_data, timestamp=timestamp)
                if feat is not None:
                    raw_records.append({
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "subject_id": subject_id,
                        "somatotype": somatotype,
                        "bmi_category": bmi_category,
                        "label": label,
                        "h2s_ratio": feat.head_to_shoulder_ratio,
                        "shoulder_tilt_deg": feat.shoulder_tilt_deg,
                        "head_roll_deg": feat.head_roll_deg,
                        "forward_head_z": feat.forward_head_z,
                        "neck_flexion_deg": feat.neck_lateral_flexion_deg,
                        "shoulder_width_norm": feat.shoulder_width_norm,
                        "is_valid": 1
                    })
                    if video_writer or save_frames:
                        self.detector.draw_skeleton(frame, results, color=(0, 255, 0))
                else:
                    raw_records.append(self._make_empty_record(frame_idx, timestamp, subject_id, somatotype, bmi_category, label))
            else:
                raw_records.append(self._make_empty_record(frame_idx, timestamp, subject_id, somatotype, bmi_category, label))

            if save_frames and frames_out_dir:
                frame_filename = os.path.join(frames_out_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(frame_filename, frame)

            if video_writer:
                video_writer.write(frame)

            frame_idx += 1

        cap.release()
        if video_writer:
            video_writer.release()

        df_raw = pd.DataFrame(raw_records)
        print(f"  - Extracted {len(df_raw)} raw frame records (Valid pose frames: {df_raw['is_valid'].sum()})")
        if save_frames and frames_out_dir:
            print(f"  - [SAVED] Exported {frame_idx} frame images to: {frames_out_dir}/")

        # Clean and Impute
        df_cleaned = self.clean_and_impute(df_raw)
        
        # Save processed CSV
        out_csv = os.path.join(output_dir, f"{subject_id}_cleaned.csv")
        df_cleaned.to_csv(out_csv, index=False)
        print(f"  - [SUCCESS] Cleaned dataset saved to: {out_csv}")

        return df_cleaned

    def _make_empty_record(self, frame_idx: int, timestamp: float, subject_id: str, somatotype: str, bmi_category: str, label: str) -> Dict:
        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "subject_id": subject_id,
            "somatotype": somatotype,
            "bmi_category": bmi_category,
            "label": label,
            "h2s_ratio": np.nan,
            "shoulder_tilt_deg": np.nan,
            "head_roll_deg": np.nan,
            "forward_head_z": np.nan,
            "neck_flexion_deg": np.nan,
            "shoulder_width_norm": np.nan,
            "is_valid": 0
        }

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Systematic data cleaning:
        1. Outlier detection using Z-score > 3 on feature variations.
        2. Replaces outliers with NaN.
        3. Linear interpolation for missing/corrupted values up to max_missing_consecutive_frames.
        4. Forward/backward fill boundary edges.
        """
        df_clean = df.copy()
        feature_cols = ["h2s_ratio", "shoulder_tilt_deg", "head_roll_deg", "forward_head_z", "neck_flexion_deg", "shoulder_width_norm"]

        # 1. Z-Score Outlier Filtering (Pure NumPy calculation: Z = |x - mean| / std)
        for col in feature_cols:
            valid_vals = df_clean[col].dropna()
            if len(valid_vals) > 10:
                mean_val = np.mean(valid_vals)
                std_val = np.std(valid_vals)
                if std_val > 1e-6:
                    z_scores = np.abs((valid_vals - mean_val) / std_val)
                    outlier_indices = valid_vals.index[z_scores > self.z_score_threshold]
                    if len(outlier_indices) > 0:
                        print(f"    * Detected & removed {len(outlier_indices)} Z-score outliers in '{col}' (Z > {self.z_score_threshold})")
                        df_clean.loc[outlier_indices, col] = np.nan

        # 2. Linear Interpolation for brief landmark dropouts
        for col in feature_cols:
            df_clean[col] = df_clean[col].interpolate(method='linear', limit=self.max_missing_consecutive_frames)
            # Edge fill
            df_clean[col] = df_clean[col].bfill().ffill()

        # Drop any remaining unfillable rows (e.g. completely empty video sections)
        df_clean = df_clean.dropna(subset=feature_cols).reset_index(drop=True)
        return df_clean

    def process_directory(
        self,
        input_dir: str,
        output_dir: str = "data/processed",
        somatotype: str = "Mesomorph",
        bmi_category: str = "Normal",
        label: str = "auto",
        save_frames: bool = False
    ):
        """Processes all raw video files found in a folder."""
        valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".flv"}
        video_files = []
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    video_files.append(os.path.join(root, file))

        if not video_files:
            print(f"[WARN] No video files ({', '.join(valid_extensions)}) found in: {input_dir}")
            return

        print(f"\n[INFO] Found {len(video_files)} video(s) to process in '{input_dir}'.")
        all_dfs = []
        for vpath in video_files:
            sub_id = os.path.splitext(os.path.basename(vpath))[0]
            df_sub = self.process_video(
                video_path=vpath,
                subject_id=sub_id,
                somatotype=somatotype,
                bmi_category=bmi_category,
                label=label,
                save_frames=save_frames,
                output_dir=output_dir
            )
            all_dfs.append(df_sub)

        if all_dfs:
            new_combined = pd.concat(all_dfs, ignore_index=True)
            combined_path = os.path.join(output_dir, "combined_dataset.csv")
            
            if os.path.exists(combined_path):
                existing_df = pd.read_csv(combined_path)
                combined_df = pd.concat([existing_df, new_combined], ignore_index=True)
                # Deduplicate if re-run on same subjects
                combined_df = combined_df.drop_duplicates(subset=["timestamp", "subject_id"]).reset_index(drop=True)
            else:
                combined_df = new_combined

            combined_df.to_csv(combined_path, index=False)
            print(f"\n[ALL DONE] Total cumulative clean dataset: {len(combined_df)} frames saved in: {combined_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw video files into cleaned posture datasets")
    parser.add_argument("--video", nargs="+", default=None, help="Path to single raw video file")
    parser.add_argument("--dir", nargs="+", default=None, help="Path to directory containing raw video files")
    parser.add_argument("--subject", type=str, default="subject_01", help="Subject identifier")
    parser.add_argument("--somatotype", type=str, default="Mesomorph", choices=["Ectomorph", "Mesomorph", "Endomorph"])
    parser.add_argument("--bmi", type=str, default="Normal", choices=["Underweight", "Normal", "Overweight"])
    parser.add_argument("--label", type=str, default="auto", choices=["upright", "slouch", "fhp", "auto"], help="Posture label (upright, slouch, fhp, or auto)")
    parser.add_argument("--view_angle", type=str, default="Front", choices=["Front", "Side"], help="Camera view angle (Front or Side)")
    parser.add_argument("--save_frames", action="store_true", help="Export individual extracted image frames to data/frames/")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory for processed CSVs")
    args = parser.parse_args()

    pipeline = VideoDataPipeline()

    if args.video:
        video_path = " ".join(args.video) if isinstance(args.video, list) else args.video
        sub_id = args.subject
        if sub_id == "subject_01":
            sub_id = os.path.splitext(os.path.basename(video_path))[0]

        df_sub = pipeline.process_video(
            video_path=video_path,
            subject_id=sub_id,
            somatotype=args.somatotype,
            bmi_category=args.bmi,
            label=args.label,
            save_frames=args.save_frames,
            output_dir=args.output
        )

        df_sub["view_angle"] = args.view_angle

        # Automatically update combined dataset files
        combined_path = os.path.join(args.output, "combined_dataset.csv")
        if os.path.exists(combined_path):
            existing_df = pd.read_csv(combined_path)
            combined_df = pd.concat([existing_df, df_sub], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["timestamp", "subject_id"]).reset_index(drop=True)
        else:
            combined_df = df_sub

        combined_df.to_csv(combined_path, index=False)
        print(f"\n[DATASET UPDATED] Added {len(df_sub)} frames. Total dataset: {len(combined_df):,} frames in {combined_path}")

        # Also update front_view_dataset.csv if view is Front
        if args.view_angle == "Front":
            front_path = os.path.join(args.output, "front_view_dataset.csv")
            if os.path.exists(front_path):
                f_df = pd.read_csv(front_path)
                f_df = pd.concat([f_df, df_sub], ignore_index=True).drop_duplicates(subset=["timestamp", "subject_id"]).reset_index(drop=True)
            else:
                f_df = df_sub
            f_df.to_csv(front_path, index=False)

    elif args.dir:
        input_dir = " ".join(args.dir) if isinstance(args.dir, list) else args.dir
        pipeline.process_directory(
            input_dir=input_dir,
            output_dir=args.output,
            somatotype=args.somatotype,
            bmi_category=args.bmi,
            label=args.label,
            save_frames=args.save_frames
        )
    else:
        print("[USAGE] Please specify either --video <path_to_video> or --dir <path_to_video_folder>.")
