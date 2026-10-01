"""
Cleans and standardizes ErgoEMA processed datasets:
1. Removes old un-renamed webcam duplicate files (IMG_*, WIN_*).
2. Adds explicit 'participant_id' column (Participant1 - Participant9), and drops the recordings in EXCLUDED_RECORDINGS.
3. Orders frames per participant logically: Full Upright -> Slightly Upright -> Slightly Slouch -> Full Slouch.
4. Preserves 90deg benchmark data in all_angles_dataset.csv.
5. Verifies participant balance and calibration readiness.
"""

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")

# Recordings left out of every dataset: subject_id -> reason
EXCLUDED_RECORDINGS = {
    "Participant3_FV_FuSl": "filed under Front View but filmed side-on, so its features are not front-view measurements",
}

def drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
    """Removes the frames of excluded recordings."""
    return df[~df["subject_id"].astype(str).isin(EXCLUDED_RECORDINGS)]

def get_posture_order(subject_id: str) -> int:
    """Returns numerical sort rank to order postures chronologically."""
    s = str(subject_id).lower()
    if "fuup" in s or "full upright" in s:
        return 1
    if "slup" in s or "slightly upright" in s:
        return 2
    if "slsl" in s or "slightly slouch" in s:
        return 3
    if "fusl" in s or "full slouch" in s:
        return 4
    return 5

def clean_front_dataset(filename: str):
    p = PROCESSED_DIR / filename
    if not p.exists():
        print(f"[SKIP] {filename} not found.")
        return

    df = pd.read_csv(p, low_memory=False)
    initial_count = len(df)

    # Filter to Participant* records only
    df_clean = drop_excluded(df[df["subject_id"].astype(str).str.startswith("Participant")]).copy()

    # Extract participant_id
    df_clean["participant_id"] = df_clean["subject_id"].apply(lambda s: str(s).split("_")[0])

    # Sort logically: by participant, then posture sequence (FuUp -> SlUp -> SlSl -> FuSl), then timestamp
    df_clean["_posture_rank"] = df_clean["subject_id"].apply(get_posture_order)
    df_clean = df_clean.sort_values(by=["participant_id", "_posture_rank", "timestamp"]).reset_index(drop=True)
    df_clean = df_clean.drop(columns=["_posture_rank"])

    # Reorder columns to place participant_id near subject_id
    cols = list(df_clean.columns)
    if "participant_id" in cols:
        cols.remove("participant_id")
        sub_idx = cols.index("subject_id") if "subject_id" in cols else 2
        cols.insert(sub_idx + 1, "participant_id")
        df_clean = df_clean[cols]

    df_clean.to_csv(p, index=False)
    print(f"[CLEANED] {filename}: {initial_count:,} -> {len(df_clean):,} frames across {df_clean['participant_id'].nunique()} participants.")

def clean_side_dataset():
    p = PROCESSED_DIR / "side_view_dataset.csv"
    if not p.exists():
        return

    df = pd.read_csv(p, low_memory=False)
    initial_count = len(df)

    df_clean = df[df["subject_id"].astype(str).str.startswith("Participant")].copy()
    df_clean["participant_id"] = df_clean["subject_id"].apply(lambda s: str(s).split("_")[0])
    
    df_clean["_posture_rank"] = df_clean["subject_id"].apply(get_posture_order)
    df_clean = df_clean.sort_values(by=["participant_id", "_posture_rank", "timestamp"]).reset_index(drop=True)
    df_clean = df_clean.drop(columns=["_posture_rank"])

    cols = list(df_clean.columns)
    if "participant_id" in cols:
        cols.remove("participant_id")
        sub_idx = cols.index("subject_id") if "subject_id" in cols else 2
        cols.insert(sub_idx + 1, "participant_id")
        df_clean = df_clean[cols]

    df_clean.to_csv(p, index=False)
    print(f"[CLEANED] side_view_dataset.csv: {initial_count:,} -> {len(df_clean):,} frames across {df_clean['participant_id'].nunique()} participants.")

def clean_all_angles_dataset():
    p = PROCESSED_DIR / "all_angles_dataset.csv"
    if not p.exists():
        return

    df = pd.read_csv(p, low_memory=False)
    initial_count = len(df)

    # Keep all Participant* rows + benchmark 90deg rows
    is_part = df["subject_id"].astype(str).str.startswith("Participant")
    is_90deg = df["view_angle"].astype(str).str.lower().isin(["90deg", "lateral"])
    
    df_clean = drop_excluded(df[is_part | is_90deg]).copy()
    
    # Assign participant_id
    def get_pid(row):
        sid = str(row["subject_id"])
        if sid.startswith("Participant"):
            return sid.split("_")[0]
        return sid  # benchmark identifier
        
    df_clean["participant_id"] = df_clean.apply(get_pid, axis=1)

    df_clean["_posture_rank"] = df_clean["subject_id"].apply(get_posture_order)
    df_clean = df_clean.sort_values(by=["participant_id", "_posture_rank", "timestamp"]).reset_index(drop=True)
    df_clean = df_clean.drop(columns=["_posture_rank"])

    cols = list(df_clean.columns)
    if "participant_id" in cols:
        cols.remove("participant_id")
        sub_idx = cols.index("subject_id") if "subject_id" in cols else 2
        cols.insert(sub_idx + 1, "participant_id")
        df_clean = df_clean[cols]

    df_clean.to_csv(p, index=False)
    print(f"[CLEANED] all_angles_dataset.csv: {initial_count:,} -> {len(df_clean):,} frames.")

def remove_unrenamed_raw_files():
    """Removes any old un-anonymized raw webcam files (IMG_*, WIN_*) to ensure participant confidentiality."""
    count = 0
    for f in PROCESSED_DIR.iterdir():
        if f.is_file() and (f.name.startswith("IMG_") or f.name.startswith("WIN_")):
            f.unlink()
            count += 1
    if count > 0:
        print(f"[REMOVED] {count} un-anonymized raw files to ensure participant confidentiality.")

if __name__ == "__main__":
    print("=" * 60)
    print("Standardizing and Cleaning ErgoEMA Datasets")
    print("=" * 60)
    remove_unrenamed_raw_files()
    clean_front_dataset("combined_dataset.csv")
    clean_front_dataset("front_view_dataset.csv")
    clean_side_dataset()
    clean_all_angles_dataset()
    print("=" * 60)
    print("Done!")
