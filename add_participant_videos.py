"""
Adds a recording session's videos (front, oblique and lateral) to the ErgoEMA datasets.

Expected layout of --dir:
    Front Facing/Participant N-Upright.mp4, Participant N-Upright 2.mp4, Participant N-Slouch.mp4, ...
    Oblique/Participant N-Upright Left.mp4, Participant N-Slouch 2 Right.mp4, ...
    Lateral/Participant N-Upright.mp4, Participant N-Slouch 2.mp4, ...

Front and oblique clips go through the same per-clip pipeline as batch_extract.py, lateral clips through
lateral_video_processor.py. The lateral recordings number their participants separately, so their IDs come from
the lateral file names and their demographics from LATERAL_PARTICIPANT_PROFILES. The new rows are appended to the
dataset files as text, so every existing line stays byte-for-byte unchanged; recordings already in the datasets
are refused.

Usage:
    python add_participant_videos.py --dir "C:\\...\\Ectomorph Posture"
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd

from video_processor import VideoDataPipeline
from lateral_video_processor import PARTICIPANT_PROFILES, LATERAL_PARTICIPANT_PROFILES, extract_lateral_videos
from clean_datasets import EXCLUDED_RECORDINGS

PROCESSED_DIR = "data/processed"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CLIP_NAME = re.compile(r"^Par\w*icipant\s*(\d+)\s*-\s*(Upright|Slouch|FHP)\s*(\d*)\s*(Left|Right)?$", re.IGNORECASE)
LABELS = {"upright": ("upright", "Up"), "slouch": ("slouch", "Sl"), "fhp": ("forward_head", "Fh")}
POSTURE_ORDER = {"upright": 0, "forward_head": 1, "slouch": 2}

def parse_clip(path: str) -> Tuple[str, str, str, int, Optional[str]]:
    """'Participant 11-Slouch 2 Left.mp4' -> ('Participant11', 'slouch', 'Sl', 2, 'Left')."""
    match = CLIP_NAME.match(os.path.splitext(os.path.basename(path))[0].strip())
    if match is None:
        raise ValueError(f"unexpected video name: {os.path.basename(path)}")
    number, posture, take, side = match.groups()
    label, code = LABELS[posture.lower()]
    return f"Participant{number}", label, code, int(take or 1), side.capitalize() if side else None

def videos_in(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]

def profile_of(participant: str, lateral: bool = False) -> Dict[str, str]:
    table, table_name = (LATERAL_PARTICIPANT_PROFILES, "LATERAL_PARTICIPANT_PROFILES") if lateral else (PARTICIPANT_PROFILES, "PARTICIPANT_PROFILES")
    if participant not in table:
        raise ValueError(f"{participant} has no entry in {table_name} (lateral_video_processor.py); add its somatotype and BMI first")
    return table[participant]

def front_oblique_jobs(folder: str, view_angle: str) -> List[Dict]:
    """One job per clip, ordered per participant: upright takes before slouch takes, left camera before right."""
    jobs = []
    for path in videos_in(folder):
        participant, label, code, take, side = parse_clip(path)
        view_code = "FV" if view_angle == "Front" else {"Left": "LS", "Right": "RS"}[side]
        jobs.append({
            "path": path, "participant": participant, "label": label, "take": take, "side": side or "",
            "subject_id": f"{participant}_{view_code}_{code}{take}", "view_angle": view_angle
        })
    jobs.sort(key=lambda j: (int(j["participant"][len("Participant"):]), POSTURE_ORDER[j["label"]], j["take"], j["side"]))
    return jobs

def extract_front_oblique(jobs: List[Dict], pipeline: VideoDataPipeline) -> pd.DataFrame:
    frames = []
    for job in jobs:
        prof = profile_of(job["participant"])
        df = pipeline.process_video(
            video_path=job["path"], subject_id=job["subject_id"], somatotype=prof["somatotype"], bmi_category=prof["bmi"],
            label=job["label"], save_annotated_video=False, save_frames=False, output_dir=PROCESSED_DIR
        )
        df["view_angle"] = job["view_angle"]
        df["participant_id"] = job["participant"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def append_rows(path: str, rows: pd.DataFrame, after_view: Optional[str] = None, as_float: Tuple[str, ...] = ()) -> int:
    """
    Appends rows as CSV text in the file's column order and line ending, leaving existing lines untouched.
    With after_view, the rows go right after the last existing row of that view_angle (all_angles keeps views grouped).
    """
    if rows.empty:
        return 0
    with open(path, "rb") as f:
        raw = f.read()
    newline = b"\r\n" if b"\r\n" in raw[:4096] else b"\n"
    lines = raw.split(newline)
    trailing = lines[-1] == b""
    if trailing:
        lines = lines[:-1]
    header = lines[0].decode("utf-8").split(",")

    block = rows.reindex(columns=header)
    for col in as_float:
        block[col] = block[col].astype(float)
    text = block.to_csv(header=False, index=False, lineterminator="\n").encode("utf-8")
    new_lines = text.split(b"\n")[:-1]

    position = len(lines)
    if after_view is not None:
        col = header.index("view_angle")
        matches = [i for i, line in enumerate(lines[1:], 1) if line.decode("utf-8").split(",")[col] == after_view]
        if matches:
            position = matches[-1] + 1
    lines = lines[:position] + new_lines + lines[position:]
    with open(path, "wb") as f:
        f.write(newline.join(lines) + (newline if trailing else b""))
    return len(new_lines)

def already_present(path: str, column: str, values: List[str]) -> List[str]:
    existing = set(pd.read_csv(path, usecols=[column], dtype=str)[column])
    return sorted(set(values) & existing)

def main():
    parser = argparse.ArgumentParser(description="Add a recording session's front, oblique and lateral videos to the ErgoEMA datasets")
    parser.add_argument("--dir", required=True, help="Folder with 'Front Facing', 'Oblique' and 'Lateral' subfolders")
    parser.add_argument("--lateral-map", nargs="*", default=[], metavar="FROM=TO",
                        help="Renames participant IDs parsed from the lateral file names (they already use the lateral numbering), e.g. Participant9=Participant12")
    parser.add_argument("--dry-run", action="store_true", help="List the clips and their IDs without extracting anything")
    args = parser.parse_args()

    lateral_map = dict(item.split("=", 1) for item in args.lateral_map)
    front_jobs = front_oblique_jobs(os.path.join(args.dir, "Front Facing"), "Front")
    oblique_jobs = front_oblique_jobs(os.path.join(args.dir, "Oblique"), "Side")
    lateral_files = sorted(videos_in(os.path.join(args.dir, "Lateral")))

    print("=" * 72)
    print(f"   Adding recordings from: {args.dir}")
    print("=" * 72)
    for job in front_jobs + oblique_jobs:
        prof = profile_of(job["participant"])
        print(f"  {job['view_angle']:<6} {os.path.basename(job['path']):<36} -> {job['subject_id']:<24} {job['label']:<8} ({prof['somatotype']}, BMI {prof['bmi']})")
    for path in lateral_files:
        participant = lateral_map.get(parse_clip(path)[0], parse_clip(path)[0])
        prof = profile_of(participant, lateral=True)
        print(f"  90deg  {os.path.basename(path):<36} -> lateral_{participant:<16} {parse_clip(path)[1]:<8} ({prof['somatotype']}, BMI {prof['bmi']})")

    subject_ids = [j["subject_id"] for j in front_jobs + oblique_jobs]
    excluded = [s for s in subject_ids if s in EXCLUDED_RECORDINGS]
    if excluded:
        raise SystemExit(f"[ERROR] These recordings are listed in EXCLUDED_RECORDINGS: {excluded}")
    clashes = already_present(os.path.join(PROCESSED_DIR, "all_angles_dataset.csv"), "subject_id",
                              subject_ids + [f"lateral_{lateral_map.get(parse_clip(p)[0], parse_clip(p)[0])}" for p in lateral_files])
    if clashes:
        raise SystemExit(f"[ERROR] Already in all_angles_dataset.csv, refusing to add twice: {clashes}")
    if args.dry_run:
        return

    pipeline = VideoDataPipeline()
    front = extract_front_oblique(front_jobs, pipeline)
    oblique = extract_front_oblique(oblique_jobs, pipeline)
    lateral = extract_lateral_videos(lateral_files, participant_map=lateral_map) if lateral_files else pd.DataFrame()
    if not lateral.empty:
        lateral["view_angle"] = "90deg"

    P = lambda name: os.path.join(PROCESSED_DIR, name)
    added = {
        "combined_dataset.csv": append_rows(P("combined_dataset.csv"), front),
        "front_view_dataset.csv": append_rows(P("front_view_dataset.csv"), front),
        "side_view_dataset.csv": append_rows(P("side_view_dataset.csv"), oblique),
        "lateral_video_90deg.csv": append_rows(P("lateral_video_90deg.csv"), lateral),
        "side_90deg_dataset.csv": append_rows(P("side_90deg_dataset.csv"), lateral),
    }
    # all_angles_dataset.csv keeps the views grouped (front, oblique, lateral); its integer columns are written as floats
    added["all_angles_dataset.csv"] = (
        append_rows(P("all_angles_dataset.csv"), front, after_view="Front", as_float=("frame_idx", "is_valid"))
        + append_rows(P("all_angles_dataset.csv"), oblique, after_view="Side", as_float=("frame_idx", "is_valid"))
        + append_rows(P("all_angles_dataset.csv"), lateral, as_float=("frame_idx", "is_valid"))
    )

    print("\n" + "=" * 72)
    for name, n in added.items():
        print(f"[SAVED] {name:<26} +{n:,} rows")
    for title, df in [("Front", front), ("Oblique", oblique), ("Lateral", lateral)]:
        if not df.empty:
            counts = df.groupby(["participant_id", "label"]).size().unstack(fill_value=0)
            print(f"\n  {title} frames added:\n" + counts.to_string())

if __name__ == "__main__":
    main()
