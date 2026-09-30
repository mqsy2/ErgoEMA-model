"""
Batch Video Frame Extraction for ErgoEMA Dataset Expansion.

Processes all Participant videos across all 12 posture/view subfolders in the
renamed Google Forms response directory. Extracts skeletal keypoints and
biomechanical features via the existing VideoDataPipeline, then appends
cleaned frame data to the combined and view-specific dataset CSVs.

Usage:
    python batch_extract.py
    python batch_extract.py --dry-run          # Preview without processing
    python batch_extract.py --front-only       # Process only Front View folders
    python batch_extract.py --side-only        # Process only Side View folders
"""

import os
import re
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from video_processor import VideoDataPipeline

# ============================================================================
# Constants
# ============================================================================

VIDEO_BASE_DIR = Path(
    r"C:\Users\Moises James Q. Sy\Pictures\frontfacing"
    r"\ErgoEMA_ Adaptive Time-Series Skeletal Data Posture Detection Model (File responses)"
)

OUTPUT_DIR = Path(r"c:\Users\Moises James Q. Sy\Documents\GitHub\ErgoEMA-model\data\processed")

# Participant demographic profiles (de-identified Participant# -> attributes)
PARTICIPANT_PROFILES: Dict[str, Dict[str, str]] = {
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


def infer_label(folder_name: str) -> str:
    """Infers the posture label from the subfolder name."""
    name = folder_name.lower()
    if "full upright" in name or "slightly upright" in name:
        return "upright"
    elif "full slouch" in name or "slightly slouch" in name:
        return "slouch"
    else:
        raise ValueError(f"Cannot infer label from folder: {folder_name}")


def infer_view_angle(folder_name: str) -> str:
    """Infers the camera view angle from the subfolder name."""
    name = folder_name.lower()
    if "front" in name:
        return "Front"
    elif "left" in name or "right" in name:
        return "Side"
    else:
        raise ValueError(f"Cannot infer view angle from folder: {folder_name}")


def get_subject_id(participant_key: str, folder_name: str) -> str:
    """
    Generates a unique subject_id for each video, encoding the participant,
    view angle, and posture intensity to avoid collisions across folders.

    Format: {ParticipantKey}_{ViewAbbrev}_{PostureAbbrev}
    Example: Participant1_FV_SlUp -> Participant 1, Front View, Slightly Upright
    """
    name = folder_name.lower()

    # View abbreviation
    if "front" in name:
        view_abbr = "FV"
    elif "left" in name:
        view_abbr = "LS"
    elif "right" in name:
        view_abbr = "RS"
    else:
        view_abbr = "XX"

    # Posture abbreviation
    if "full upright" in name:
        posture_abbr = "FuUp"
    elif "slightly upright" in name:
        posture_abbr = "SlUp"
    elif "slightly slouch" in name:
        posture_abbr = "SlSl"
    elif "full slouch" in name:
        posture_abbr = "FuSl"
    else:
        posture_abbr = "XX"

    return f"{participant_key}_{view_abbr}_{posture_abbr}"


def discover_jobs(base_dir: Path, front_only: bool = False, side_only: bool = False) -> List[Dict]:
    """
    Discovers all video files to process and returns a list of job descriptors.
    Each job contains: video_path, subject_id, label, view_angle, somatotype, bmi, folder_name.
    """
    jobs = []

    for subfolder in sorted(base_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        try:
            label = infer_label(subfolder.name)
            view_angle = infer_view_angle(subfolder.name)
        except ValueError as e:
            print(f"[WARN] Skipping folder: {e}")
            continue

        # Apply view filter
        if front_only and view_angle != "Front":
            continue
        if side_only and view_angle != "Side":
            continue

        for video_file in sorted(subfolder.iterdir()):
            if not video_file.is_file():
                continue
            if video_file.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}:
                continue

            # Extract participant key (e.g., "Participant1" from "Participant1.mov")
            participant_key = video_file.stem  # e.g., "Participant1"

            if participant_key not in PARTICIPANT_PROFILES:
                print(f"[WARN] Unknown participant: {participant_key} in {subfolder.name}")
                continue

            profile = PARTICIPANT_PROFILES[participant_key]
            subject_id = get_subject_id(participant_key, subfolder.name)

            jobs.append({
                "video_path": str(video_file),
                "subject_id": subject_id,
                "participant_key": participant_key,
                "participant_name": profile["name"],
                "label": label,
                "view_angle": view_angle,
                "somatotype": profile["somatotype"],
                "bmi": profile["bmi"],
                "folder_name": subfolder.name,
            })

    return jobs


def run_batch(
    jobs: List[Dict],
    output_dir: Path,
    dry_run: bool = False,
    resume_from: int = 0
) -> pd.DataFrame:
    """
    Processes all video jobs through the VideoDataPipeline.
    Returns the combined DataFrame of all newly extracted frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = VideoDataPipeline()

    all_dfs = []
    total = len(jobs)
    failed_jobs = []
    progress_file = output_dir / "batch_progress.json"

    print(f"\n{'='*70}")
    print(f"  ErgoEMA Batch Frame Extraction")
    print(f"  Total videos to process: {total}")
    print(f"  Output directory: {output_dir}")
    print(f"  Resume from: job #{resume_from + 1}" if resume_from > 0 else "")
    print(f"{'='*70}\n")

    if dry_run:
        print("[DRY RUN] Previewing jobs without processing:\n")
        for i, job in enumerate(jobs):
            print(f"  [{i+1:3d}/{total}] {job['subject_id']:30s}  "
                  f"label={job['label']:8s}  view={job['view_angle']:5s}  "
                  f"participant={job['participant_name']}")
        print(f"\n[DRY RUN] {total} videos would be processed.")
        return pd.DataFrame()

    start_time = time.time()

    for i, job in enumerate(jobs):
        if i < resume_from:
            continue

        elapsed = time.time() - start_time
        rate = (i - resume_from + 1) / max(elapsed, 1)
        remaining = (total - i - 1) / max(rate, 0.001)

        print(f"\n[{i+1:3d}/{total}] Processing: {job['subject_id']}")
        print(f"  Video: {Path(job['video_path']).name}")
        print(f"  Label: {job['label']} | View: {job['view_angle']} | "
              f"Participant: {job['participant_name']}")
        print(f"  ETA: {remaining/60:.1f} min remaining")

        try:
            df = pipeline.process_video(
                video_path=job["video_path"],
                subject_id=job["subject_id"],
                somatotype=job["somatotype"],
                bmi_category=job["bmi"],
                label=job["label"],
                save_annotated_video=False,
                save_frames=False,
                output_dir=str(output_dir)
            )

            # Add view_angle and participant_id columns
            df["view_angle"] = job["view_angle"]
            df["participant_id"] = job["participant_key"]

            all_dfs.append(df)
            print(f"  -> Extracted {len(df)} clean frames")

            # Save progress checkpoint
            progress = {
                "last_completed_index": i,
                "last_completed_subject": job["subject_id"],
                "total_frames_extracted": sum(len(d) for d in all_dfs),
                "elapsed_seconds": time.time() - start_time
            }
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            traceback.print_exc()
            failed_jobs.append({"index": i, "subject_id": job["subject_id"], "error": str(e)})

    # Combine all new data
    if all_dfs:
        new_data = pd.concat(all_dfs, ignore_index=True)
    else:
        new_data = pd.DataFrame()

    elapsed_total = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"  Batch Extraction Complete!")
    print(f"  Videos processed: {total - len(failed_jobs)}/{total}")
    print(f"  Total new frames: {len(new_data):,}")
    print(f"  Elapsed time: {elapsed_total/60:.1f} minutes")
    if failed_jobs:
        print(f"  Failed jobs: {len(failed_jobs)}")
        for fj in failed_jobs:
            print(f"    - [{fj['index']}] {fj['subject_id']}: {fj['error']}")
    print(f"{'='*70}\n")

    return new_data


def merge_into_datasets(new_data: pd.DataFrame, output_dir: Path):
    """
    Merges new frame data into the existing dataset CSVs:
    1. combined_dataset.csv  (front view data)
    2. front_view_dataset.csv
    3. side_view_dataset.csv
    4. all_angles_dataset.csv
    """
    if new_data.empty:
        print("[WARN] No new data to merge.")
        return

    # Split by view angle
    front_data = new_data[new_data["view_angle"] == "Front"]
    side_data = new_data[new_data["view_angle"] == "Side"]

    # 1. Update combined_dataset.csv (historically Front-only)
    combined_path = output_dir / "combined_dataset.csv"
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        merged = pd.concat([existing, front_data], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp", "subject_id"]).reset_index(drop=True)
    else:
        merged = front_data
    merged.to_csv(combined_path, index=False)
    print(f"[SAVED] combined_dataset.csv: {len(merged):,} total frames")

    # 2. Update front_view_dataset.csv
    front_path = output_dir / "front_view_dataset.csv"
    if front_path.exists():
        existing = pd.read_csv(front_path)
        merged_front = pd.concat([existing, front_data], ignore_index=True)
        merged_front = merged_front.drop_duplicates(subset=["timestamp", "subject_id"]).reset_index(drop=True)
    else:
        merged_front = front_data
    merged_front.to_csv(front_path, index=False)
    print(f"[SAVED] front_view_dataset.csv: {len(merged_front):,} total frames")

    # 3. Update side_view_dataset.csv
    side_path = output_dir / "side_view_dataset.csv"
    if side_path.exists():
        existing = pd.read_csv(side_path)
        merged_side = pd.concat([existing, side_data], ignore_index=True)
        merged_side = merged_side.drop_duplicates(subset=["timestamp", "subject_id"]).reset_index(drop=True)
    else:
        merged_side = side_data
    merged_side.to_csv(side_path, index=False)
    print(f"[SAVED] side_view_dataset.csv: {len(merged_side):,} total frames")

    # 4. Update all_angles_dataset.csv (union of all views)
    all_path = output_dir / "all_angles_dataset.csv"
    if all_path.exists():
        existing_all = pd.read_csv(all_path)
        merged_all = pd.concat([existing_all, new_data], ignore_index=True)
        merged_all = merged_all.drop_duplicates(subset=["timestamp", "subject_id"]).reset_index(drop=True)
    else:
        merged_all = new_data
    merged_all.to_csv(all_path, index=False)
    print(f"[SAVED] all_angles_dataset.csv: {len(merged_all):,} total frames")


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract frame data from all participant videos into ErgoEMA dataset"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview all jobs without processing any video")
    parser.add_argument("--front-only", action="store_true",
                        help="Only process Front View folders")
    parser.add_argument("--side-only", action="store_true",
                        help="Only process Side View (Left/Right) folders")
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from job index N (0-based)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for processed CSVs")
    args = parser.parse_args()

    # Discover all video jobs
    jobs = discover_jobs(
        VIDEO_BASE_DIR,
        front_only=args.front_only,
        side_only=args.side_only
    )

    if not jobs:
        print("[ERROR] No video files found to process.")
        sys.exit(1)

    print(f"\n[INFO] Discovered {len(jobs)} video files across "
          f"{len(set(j['folder_name'] for j in jobs))} folders.")

    # Run extraction
    new_data = run_batch(
        jobs,
        output_dir=Path(args.output),
        dry_run=args.dry_run,
        resume_from=args.resume
    )

    # Merge into datasets
    if not args.dry_run and not new_data.empty:
        merge_into_datasets(new_data, Path(args.output))


if __name__ == "__main__":
    main()
