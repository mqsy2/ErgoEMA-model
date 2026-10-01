"""
ErgoEMA Model Training, Hyperparameter Optimization, and Benchmark Suite.

Performs:
1. Personalized Calibration & Adaptive EMA Simulation grouped by participant.
2. Grid Search Optimization of ErgoEMA hyperparameters (alpha_EMA, delta_slouch, forward-head depth threshold).
3. Leave-One-Subject-Out (LOSO) Cross-Validation: hyperparameters are tuned on N-1 participants and scored on the held-out one.
4. Benchmarking under the same LOSO protocol (ErgoEMA, Random Forest, SVM, Logistic Regression, Static Baseline),
   with measured per-frame latency.
5. The same benchmark repeated on the oblique (45°) and lateral (90°) recordings, and a camera viewpoint ablation of the
   configuration the live application runs.
6. Generates thesis publication tables, JSON parameters, and visual figures for Chapter 4.
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Sequence, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from config import ErgoConfig
from src.ema_filter import EMAFilter
from src.feature_extractor import PostureFeatures
from src.calibrator import BaselineProfile
from src.classifier import PostureClassifier, PostureState

# Number of frames timed per model when measuring per-frame latency
LATENCY_SAMPLE_FRAMES = 1000

# Recorded frames are replayed at ~30 fps
FRAME_INTERVAL_SEC = 0.033

# ErgoEMA hyperparameter grids
ALPHAS = [0.08, 0.12, 0.15, 0.20, 0.25]
SLOUCH_THRESHOLDS = [0.08, 0.10, 0.12, 0.15, 0.18]
FHP_Z_THRESHOLDS = [0.04, 0.06, 0.10, 0.15, 0.20, 0.30]
CVA_DROP_THRESHOLDS_DEG = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]

FRONTAL_GRID = [
    {"alpha": a, "slouch_ratio_drop_thresh": s, "forward_head_z_thresh": f}
    for a in ALPHAS for s in SLOUCH_THRESHOLDS for f in FHP_Z_THRESHOLDS
]

def lateral_grid(front_params: Dict[str, float], drops: Sequence[Optional[float]]) -> List[Dict[str, Any]]:
    """Side-on settings to search. The frontal settings stay at the deployed values; they apply only to frames detected as frontal."""
    return [dict(front_params, lateral_alpha=a, lateral_cva_drop_thresh_deg=d) for a in ALPHAS for d in drops]

# How each tuned setting is shown in the LOSO tables
TUNED_COLUMNS = {
    "alpha": ("Tuned alpha", lambda v: v),
    "slouch_ratio_drop_thresh": ("Tuned Slouch Drop (%)", lambda v: round(v * 100, 1)),
    "forward_head_z_thresh": ("Tuned FHP Depth", lambda v: v),
    "lateral_alpha": ("Tuned alpha", lambda v: v),
    "lateral_cva_drop_thresh_deg": ("Tuned CVA Drop (°)", lambda v: v),
}

ERGOEMA_NAME = "ErgoEMA (Adaptive Time-Series)"
CLINICAL_VARIANT_NAME = "ErgoEMA, Clinical CVA Criterion Only"

# Features the ML baselines are trained on. The lateral classifier reads only the craniovertebral angle,
# so in that view the baselines get the side-view measurements instead of the frontal ratios.
FRONTAL_FEATURES = ["h2s_ratio", "shoulder_tilt_deg", "head_roll_deg", "forward_head_z", "neck_flexion_deg", "shoulder_width_norm"]
LATERAL_FEATURES = ["craniovertebral_angle_deg", "ear_shoulder_offset_x", "nose_shoulder_angle_deg"]

@dataclass
class StaticRule:
    """Non-adaptive baseline: flags a frame when `column` is below a threshold fitted on the training participants."""
    name: str
    column: str
    fit_threshold: Callable[[np.ndarray], float]

FRONTAL_STATIC_RULE = StaticRule("Static Rigid Threshold", "h2s_ratio", lambda train: float(np.median(train) * 0.90))
# The clinical criterion applied frame by frame, without EMA smoothing
LATERAL_STATIC_RULE = StaticRule("Static Rigid Threshold (CVA, unsmoothed)", "craniovertebral_angle_deg", lambda train: ErgoConfig().LATERAL_CVA_FHP_THRESH_DEG)

def extract_participant(subject_str: str) -> str:
    """Extracts clean human subject name or participant ID from subject string."""
    s = str(subject_str).strip()
    if s.startswith("Participant"):
        return s.split("_")[0]
    if " - " in s:
        return s.split(" - ")[-1].strip()
    return s

def with_person(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the "person" column used to group frames by participant."""
    df = df.copy()
    df["person"] = df["participant_id"] if "participant_id" in df.columns else df["subject_id"].apply(extract_participant)
    return df

def poor_posture_labels(df: pd.DataFrame) -> np.ndarray:
    """1 = poor posture (slouch or forward head), 0 = upright."""
    return df["label"].isin(["slouch", "forward_head"]).astype(int).values

def build_baseline(pgroup: pd.DataFrame, person: str) -> BaselineProfile:
    """Calibrates a personalized baseline on the participant's first upright frames, with the live calibrator's statistics."""
    upright_frames = pgroup[pgroup["label"] == "upright"]
    if len(upright_frames) >= 15:
        calib_sample = upright_frames.iloc[:60]
    else:
        calib_sample = pgroup.iloc[:min(60, len(pgroup))]

    def mean(col: str) -> float:
        return float(calib_sample[col].mean())

    def std(col: str, fallback: float) -> float:
        return float(np.std(calib_sample[col].values)) if len(calib_sample) > 1 else fallback

    # The upright craniovertebral angle is recorded only when calibration frames are mostly side-on, as in the live calibrator
    views = calib_sample["detected_view_angle"] if "detected_view_angle" in calib_sample.columns else pd.Series("frontal", index=calib_sample.index)
    lateral = calib_sample[views == "lateral"]
    mean_cva = float(lateral["craniovertebral_angle_deg"].mean()) if len(lateral) > len(calib_sample) / 2 else None

    return BaselineProfile(
        mean_h2s_ratio=mean("h2s_ratio"),
        std_h2s_ratio=std("h2s_ratio", 0.04),
        mean_shoulder_tilt_deg=mean("shoulder_tilt_deg"),
        std_shoulder_tilt_deg=std("shoulder_tilt_deg", 1.0),
        mean_head_roll_deg=mean("head_roll_deg"),
        std_head_roll_deg=std("head_roll_deg", 1.0),
        mean_forward_head_z=mean("forward_head_z"),
        std_forward_head_z=std("forward_head_z", 0.05),
        mean_shoulder_width_norm=mean("shoulder_width_norm"),
        mean_craniovertebral_angle_deg=mean_cva,
        sample_count=len(calib_sample),
        user_id=person
    )

def row_to_features(row, timestamp: float) -> PostureFeatures:
    """Builds PostureFeatures from one dataset row (as yielded by DataFrame.itertuples)."""
    return PostureFeatures(
        head_to_shoulder_ratio=float(row.h2s_ratio),
        shoulder_tilt_deg=float(row.shoulder_tilt_deg),
        head_roll_deg=float(row.head_roll_deg),
        forward_head_z=float(row.forward_head_z),
        shoulder_width_norm=float(row.shoulder_width_norm),
        neck_lateral_flexion_deg=float(row.neck_flexion_deg),
        timestamp=timestamp,
        detected_view_angle=str(getattr(row, "detected_view_angle", "frontal")),
        ear_shoulder_offset_x=float(getattr(row, "ear_shoulder_offset_x", 0.0)),
        nose_shoulder_angle_deg=float(getattr(row, "nose_shoulder_angle_deg", 0.0)),
        craniovertebral_angle_deg=float(getattr(row, "craniovertebral_angle_deg", 0.0))
    )

def smooth_participant(pgroup: pd.DataFrame, alpha: float, lateral_alpha: float) -> List[PostureFeatures]:
    """EMA-smoothed features of one participant's frames; side-on frames use their own smoothing factor, as in the live app."""
    ema_filter = EMAFilter(alpha=alpha)
    smoothed = []
    for i, row in enumerate(pgroup.itertuples(index=False)):
        feat = row_to_features(row, (i + 1) * FRAME_INTERVAL_SEC)
        ema_filter.set_alpha(lateral_alpha if feat.is_lateral else alpha)
        smoothed.append(ema_filter.update(feat))
    return smoothed

def make_classifier(params: Dict[str, Any], sustained_sec: float) -> PostureClassifier:
    """Classifier for one parameter set (the smoothing factors in `params` belong to the EMA filter)."""
    settings = {k: v for k, v in params.items() if k not in ("alpha", "lateral_alpha")}
    return PostureClassifier(sustained_window_sec=sustained_sec, **settings)

def classify_frames(classifier: PostureClassifier, smoothed: Sequence[PostureFeatures], baseline: BaselineProfile) -> np.ndarray:
    """1 = any non-upright state, 0 = upright, for each smoothed frame."""
    return np.array([
        int(classifier.evaluate_adaptive(feat, baseline, feat.timestamp).state != PostureState.GOOD_POSTURE)
        for feat in smoothed
    ])

def params_key(params: Dict[str, Any]) -> Tuple:
    return tuple(sorted(params.items()))

def run_ergoema_simulation(df: pd.DataFrame, params: Dict[str, Any], sustained_sec: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Simulates personalized online ErgoEMA monitoring on participant time-series data with one parameter set.
    Returns (y_true, y_pred, per_subject_metrics).
    """
    df = with_person(df)
    y_true_all, y_pred_all, per_person_metrics = [], [], {}
    for person, pgroup in df.groupby("person", sort=False):
        baseline = build_baseline(pgroup, person)
        smoothed = smooth_participant(pgroup, params["alpha"], params.get("lateral_alpha", params["alpha"]))
        y_true = poor_posture_labels(pgroup)
        y_pred = classify_frames(make_classifier(params, sustained_sec), smoothed, baseline)
        per_person_metrics[person] = compute_metrics(y_true, y_pred)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
    return np.concatenate(y_true_all), np.concatenate(y_pred_all), per_person_metrics

def simulate_grid(df: pd.DataFrame, grid: Sequence[Dict[str, Any]], sustained_sec: float) -> Dict[Tuple, Dict[str, Dict[str, float]]]:
    """
    Per-participant metrics for every parameter set in the grid. Each participant is calibrated once and smoothed
    once per smoothing setting, so the thresholds of a smoothing setting are scored on the same smoothed frames.
    """
    results = {params_key(p): {} for p in grid}
    by_smoothing: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for p in grid:
        by_smoothing.setdefault((p["alpha"], p.get("lateral_alpha", p["alpha"])), []).append(p)

    for person, pgroup in df.groupby("person", sort=False):
        baseline = build_baseline(pgroup, person)
        y_true = poor_posture_labels(pgroup)
        for (alpha, lateral_alpha), param_sets in by_smoothing.items():
            smoothed = smooth_participant(pgroup, alpha, lateral_alpha)
            for p in param_sets:
                results[params_key(p)][person] = compute_metrics(y_true, classify_frames(make_classifier(p, sustained_sec), smoothed, baseline))
    return results

def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    """Computes Accuracy, Precision, Recall, F1-Score, Specificity, and False Alarm Rate from confusion counts."""
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    specificity = tn / max(1, (tn + fp))
    f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))
    far = fp / max(1, (fp + tn))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "false_alarm_rate": float(far),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Accuracy, Precision, Recall, F1-Score, Specificity, and False Alarm Rate."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return metrics_from_counts(tp, tn, fp, fn)

def pool_metrics(person_metrics: Dict[str, Dict[str, float]], persons: Sequence[str]) -> Dict[str, float]:
    """Pools the confusion counts of the given participants and recomputes the metrics over all their frames."""
    return metrics_from_counts(*(sum(person_metrics[p][k] for p in persons) for k in ("tp", "tn", "fp", "fn")))

def macro_metrics(person_metrics: Dict[str, Dict[str, float]], persons: Sequence[str]) -> Dict[str, float]:
    """Mean of the per-participant scores."""
    return {
        k: float(np.mean([person_metrics[p][k] for p in persons]))
        for k in ("accuracy", "precision", "recall", "specificity", "f1_score", "false_alarm_rate")
    }

def select_best_params(
    grid_person_metrics: Dict[Tuple, Dict[str, Dict[str, float]]],
    persons: Sequence[str]
) -> Tuple[Tuple, Dict[str, float]]:
    """Returns the parameter set (as a key) with the highest pooled F1-score over the given participants."""
    best_key, best_metrics = None, None
    for key, person_metrics in grid_person_metrics.items():
        metrics = pool_metrics(person_metrics, persons)
        if best_metrics is None or metrics["f1_score"] > best_metrics["f1_score"]:
            best_key, best_metrics = key, metrics
    return best_key, best_metrics

def loso_from_grid(grid_person_metrics: Dict[Tuple, Dict[str, Dict[str, float]]], persons: Sequence[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tuple]]:
    """Each participant scored with the parameter set that is best on the other participants: (metrics, chosen key) per participant."""
    held_out, chosen = {}, {}
    for held in persons:
        key, _ = select_best_params(grid_person_metrics, [p for p in persons if p != held])
        held_out[held], chosen[held] = grid_person_metrics[key][held], key
    return held_out, chosen

def measure_latency_ms(step: Callable, inputs: Sequence) -> float:
    """Median wall-clock time (ms) of one call to `step` per input frame, measured on this machine."""
    timings = []
    for item in inputs:
        t0 = time.perf_counter()
        step(item)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(timings))

def measure_ergoema_latency_ms(df: pd.DataFrame, params: Dict[str, Any], sustained_sec: float) -> float:
    """Median per-frame time (ms) of the EMA update + adaptive classification on this machine."""
    person = df["person"].iloc[0]
    pgroup = df[df["person"] == person].iloc[:LATENCY_SAMPLE_FRAMES]

    alpha = params["alpha"]
    lateral_alpha = params.get("lateral_alpha", alpha)
    ema_filter = EMAFilter(alpha=alpha)
    classifier = make_classifier(params, sustained_sec)
    baseline = build_baseline(pgroup, person)
    feats = [row_to_features(row, (i + 1) * FRAME_INTERVAL_SEC) for i, row in enumerate(pgroup.itertuples(index=False))]

    def step(feat: PostureFeatures):
        ema_filter.set_alpha(lateral_alpha if feat.is_lateral else alpha)
        return classifier.evaluate_adaptive(ema_filter.update(feat), baseline, feat.timestamp)

    return measure_latency_ms(step, feats)

def benchmark_row(name: str, m: Dict[str, float], latency_ms: float) -> Dict:
    """Formats one model's metrics as a row of the benchmark comparison table."""
    return {
        "Model Architecture": name,
        "Accuracy (%)": round(m["accuracy"]*100, 2),
        "Precision (%)": round(m["precision"]*100, 2),
        "Recall (%)": round(m["recall"]*100, 2),
        "F1-Score (%)": round(m["f1_score"]*100, 2),
        "FAR (%)": round(m["false_alarm_rate"]*100, 2),
        "Latency (ms)": round(latency_ms, 4)
    }

def metric_columns(m: Dict[str, float]) -> Dict[str, float]:
    """Per-participant metric columns of a LOSO table."""
    return {
        "Accuracy (%)": round(m["accuracy"] * 100, 2),
        "Precision (%)": round(m["precision"] * 100, 2),
        "Recall (%)": round(m["recall"] * 100, 2),
        "Specificity (%)": round(m["specificity"] * 100, 2),
        "F1-Score (%)": round(m["f1_score"] * 100, 2)
    }

def evaluate_view(
    df: pd.DataFrame,
    grid: Sequence[Dict[str, Any]],
    feature_cols: Sequence[str],
    static_rule: StaticRule,
    alert_window: float,
    params_heading: str,
    tag: str = "",
    rule_variants: Sequence[Tuple[str, Sequence[Dict[str, Any]]]] = ()
) -> Dict[str, Any]:
    """
    Grid-searches the ErgoEMA hyperparameters, scores ErgoEMA under Leave-One-Subject-Out cross-validation, and
    benchmarks the rule variants and the ML and static baselines on the same folds.
    Every row of `df` must have a "person".
    """
    persons = sorted(df["person"].unique())
    tuned = [k for k in grid[0] if len({p[k] for p in grid}) > 1]

    # 1. Grid Search Optimization for ErgoEMA Hyperparameters
    print(f"\n[OPTIMIZATION{tag}] Grid search over {', '.join(tuned)} ({len(grid)} settings)...")
    grid_person_metrics = simulate_grid(df, grid, alert_window)
    grid_results = []
    for key, person_metrics in grid_person_metrics.items():
        m = pool_metrics(person_metrics, persons)
        grid_results.append(dict(dict(key), **{k: m[k] for k in ("accuracy", "precision", "recall", "f1_score", "false_alarm_rate")}))

    # Tuned on all participants (in-sample, not a validation result)
    best_key, in_sample_metrics = select_best_params(grid_person_metrics, persons)
    best_params = dict(best_key)

    print(f"\n{params_heading.format(n=len(persons))}")
    for k in tuned:
        print(f"  * {TUNED_COLUMNS[k][0].replace('Tuned ', ''):<34}: {TUNED_COLUMNS[k][1](best_params[k])}")
    print(f"  * {'Sustained Alert Window (config)':<34}: {alert_window} sec")
    print(f"  * {'In-Sample Accuracy':<34}: {in_sample_metrics['accuracy'] * 100:.2f}%")
    print(f"  * {'In-Sample F1-Score':<34}: {in_sample_metrics['f1_score'] * 100:.2f}%")
    print(f"  * {'In-Sample False Alarm Rate (FAR)':<34}: {in_sample_metrics['false_alarm_rate'] * 100:.2f}%")

    # 2. Leave-One-Subject-Out (LOSO) Cross-Validation
    print(f"\n[LOSO CROSS-VALIDATION{tag}] Tuned on N-1 participants, scored on the held-out participant:")
    held_out_metrics, chosen = loso_from_grid(grid_person_metrics, persons)
    loso_records = []
    for held_out in persons:
        sub_df = df[df["person"] == held_out]
        fold_params = dict(chosen[held_out])
        record = {"Participant": held_out}
        if "somatotype" in sub_df.columns:
            record["Somatotype"] = sub_df["somatotype"].iloc[0]
        if "bmi_category" in sub_df.columns:
            record["BMI Category"] = sub_df["bmi_category"].iloc[0]
        record["Frames"] = len(sub_df)
        for k in tuned:
            record[TUNED_COLUMNS[k][0]] = TUNED_COLUMNS[k][1](fold_params[k])
        record.update(metric_columns(held_out_metrics[held_out]))
        loso_records.append(record)

    loso_df = pd.DataFrame(loso_records)
    print(loso_df.to_string(index=False))

    loso_metrics = pool_metrics(held_out_metrics, persons)
    loso_macro = macro_metrics(held_out_metrics, persons)
    print(f"\n  * LOSO Held-Out Accuracy (pooled)   : {loso_metrics['accuracy'] * 100:.2f}%")
    print(f"  * LOSO Held-Out F1-Score (pooled)   : {loso_metrics['f1_score'] * 100:.2f}%")
    print(f"  * LOSO Sensitivity / Recall         : {loso_metrics['recall'] * 100:.2f}%")
    print(f"  * LOSO Specificity                  : {loso_metrics['specificity'] * 100:.2f}%")
    print(f"  * LOSO False Alarm Rate (FAR)       : {loso_metrics['false_alarm_rate'] * 100:.2f}%")

    bench_rows = [benchmark_row(ERGOEMA_NAME, loso_metrics, measure_ergoema_latency_ms(df, best_params, alert_window))]

    # Rule variants, tuned and scored with the same LOSO protocol
    for name, variant_grid in rule_variants:
        variant_metrics = simulate_grid(df, variant_grid, alert_window)
        variant_held_out, _ = loso_from_grid(variant_metrics, persons)
        variant_params = dict(select_best_params(variant_metrics, persons)[0])
        bench_rows.append(benchmark_row(name, pool_metrics(variant_held_out, persons), measure_ergoema_latency_ms(df, variant_params, alert_window)))

    # 3. Benchmark Comparisons Against ML Models (same LOSO protocol)
    print(f"\n[MODEL BENCHMARK{tag}] Comparing ErgoEMA vs Machine Learning Baselines (LOSO, measured latency)...")
    X = df[list(feature_cols)].values
    y = poor_posture_labels(df)
    groups = df["person"].values
    logo = LeaveOneGroupOut()
    latency_idx = np.linspace(0, len(df) - 1, min(LATENCY_SAMPLE_FRAMES, len(df))).astype(int)

    ml_models = [
        ("Random Forest (100 Trees)", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Support Vector Machine (RBF)", SVC(kernel="rbf", random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42))
    ]
    for name, model in ml_models:
        # Accuracy: each participant is predicted by a model trained on the other participants
        y_ml = cross_val_predict(model, X, y, groups=groups, cv=logo)
        # Latency: single-frame inference of the model fitted on the full dataset
        model.fit(X, y)
        latency_ml = measure_latency_ms(model.predict, [X[i:i + 1] for i in latency_idx])
        bench_rows.append(benchmark_row(name, compute_metrics(y, y_ml), latency_ml))

    # Static Rigid Threshold Baseline (non-adaptive; threshold fitted on the training participants only)
    values = df[static_rule.column].values
    y_static = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        y_static[test_idx] = (values[test_idx] < static_rule.fit_threshold(values[train_idx])).astype(int)
    static_thresh = static_rule.fit_threshold(values)
    latency_static = measure_latency_ms(lambda value: value < static_thresh, [float(values[i]) for i in latency_idx])
    bench_rows.append(benchmark_row(static_rule.name, compute_metrics(y, y_static), latency_static))

    bench_df = pd.DataFrame(bench_rows)
    print(bench_df.to_string(index=False))

    return {
        "best_params": best_params,
        "tuned": tuned,
        "in_sample_metrics": in_sample_metrics,
        "grid_results": grid_results,
        "loso_records": loso_records,
        "loso_df": loso_df,
        "loso_metrics": loso_metrics,
        "loso_macro": loso_macro,
        "bench_df": bench_df
    }

def ablation_row(title: str, df_view: pd.DataFrame, params: Dict[str, Any], alert_window: float) -> Dict[str, Any]:
    """One row of the viewpoint ablation; "Frames" counts the frames actually scored (rows without a participant ID are not simulated)."""
    y_true, y_pred, _ = run_ergoema_simulation(df_view, params, alert_window)
    m = compute_metrics(y_true, y_pred)
    return {"Camera Perspective": title, "Frames": len(y_true), "Accuracy (%)": round(m["accuracy"]*100, 2), "Precision (%)": round(m["precision"]*100, 2), "Recall (%)": round(m["recall"]*100, 2), "F1-Score (%)": round(m["f1_score"]*100, 2), "FAR (%)": round(m["false_alarm_rate"]*100, 2)}

def train_and_optimize(dataset_path: str = "data/processed/combined_dataset.csv", output_dir: str = "results", run_ablation: bool = True):
    """Runs full model training, parameter optimization, LOSO cross-validation, and ML comparisons."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 72)
    print("      ErgoEMA Model Training, Parameter Optimization & LOSO Validation")
    print("=" * 72)

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        return

    df = with_person(pd.read_csv(dataset_path))
    persons = sorted(df["person"].unique())
    if len(persons) < 2:
        print(f"[ERROR] LOSO cross-validation needs at least 2 participants, found {len(persons)}.")
        return

    print(f"\n[DATASET LOADED] {len(df):,} total frames across {len(persons)} study participants:")
    for person in persons:
        sub_df = df[df["person"] == person]
        count = len(sub_df)
        n_upright = (sub_df["label"] == "upright").sum()
        n_slouch = (sub_df["label"] == "slouch").sum()
        somatotype = sub_df["somatotype"].iloc[0] if "somatotype" in sub_df.columns else "N/A"
        bmi = sub_df["bmi_category"].iloc[0] if "bmi_category" in sub_df.columns else "N/A"
        print(f"  * {person:<14} ({somatotype}, {bmi}): {count:>5} frames ({n_upright:>4} upright, {n_slouch:>4} slouch)")

    # Frames are scored on the classified posture state, which does not depend on the
    # sustained alert window, so the window is taken from config rather than tuned.
    alert_window = ErgoConfig().SUSTAINED_ALERT_WINDOW_SEC

    # 1-3. Grid search for the deployed hyperparameters, LOSO cross-validation and model benchmark (primary dataset)
    front = evaluate_view(
        df, FRONTAL_GRID, FRONTAL_FEATURES, FRONTAL_STATIC_RULE, alert_window,
        "[DEPLOYED HYPERPARAMETERS] Tuned on all {n} participants:"
    )
    deployed = dict(front["best_params"])

    df_all = None
    all_angles_path = "data/processed/all_angles_dataset.csv"
    if run_ablation and os.path.exists(all_angles_path):
        df_all = pd.read_csv(all_angles_path, low_memory=False)
        if "view_angle" not in df_all.columns:
            df_all = None

    # 4. The LOSO benchmark of steps 1-3 repeated on the oblique (45°) and lateral (90°) recordings
    view_results = []
    lateral_params = {}
    if df_all is not None:
        view_specs = [
            # (key, title, view_angle value, ErgoEMA grid, ML baseline features, static baseline, rule variants)
            ("oblique_45", "Oblique Profile (45°)", "Side", FRONTAL_GRID, FRONTAL_FEATURES, FRONTAL_STATIC_RULE, []),
            ("lateral_90", "Lateral Profile (90°)", "90deg", lateral_grid(deployed, CVA_DROP_THRESHOLDS_DEG), LATERAL_FEATURES, LATERAL_STATIC_RULE,
             [(CLINICAL_VARIANT_NAME, lateral_grid(deployed, [None]))]),
        ]
        for key, title, view_value, grid, feature_cols, static_rule, variants in view_specs:
            in_view = df_all["view_angle"] == view_value
            # Rows without a participant ID (public benchmark images) cannot be assigned to a LOSO fold
            df_view = with_person(df_all[in_view & df_all["participant_id"].notna()])
            if df_view["person"].nunique() < 2:
                print(f"\n[WARN] {title} benchmark skipped: LOSO needs at least 2 participants.")
                continue

            print("\n" + "=" * 72)
            print(f"      {title} Benchmark: {len(df_view):,} frames")
            print("=" * 72)
            result = evaluate_view(
                df_view, grid, feature_cols, static_rule, alert_window,
                f"[{title.upper()} HYPERPARAMETERS] Tuned on all {{n}} participants (in-sample):",
                tag=f" - {title}", rule_variants=variants
            )
            result.update({"key": key, "title": title, "excluded_frames": int((in_view & df_all["participant_id"].isna()).sum())})
            view_results.append(result)
            if key == "lateral_90":
                lateral_params = {k: result["best_params"][k] for k in ("lateral_alpha", "lateral_cva_drop_thresh_deg")}

    # The configuration the live application runs: front-view settings for frontal and oblique frames, lateral settings side-on
    deployed_config = dict(deployed, **lateral_params)

    # 5. Viewpoint Ablation of the deployed configuration (Front 0° vs Oblique 45° vs Lateral 90° vs Combined)
    ablation_df = None
    if df_all is not None:
        ablation_rows = []
        for title, view_value in [("Front View (0° - Primary Scope)", "Front"), ("Oblique Profile (45° View)", "Side"), ("Lateral Profile (90° View)", "90deg")]:
            df_v = df_all[df_all["view_angle"] == view_value]
            if len(df_v) > 0:
                ablation_rows.append(ablation_row(title, df_v, deployed_config, alert_window))
        ablation_rows.append(ablation_row("All Combined Perspectives", df_all, deployed_config, alert_window))
        ablation_df = pd.DataFrame(ablation_rows)
        print("\n[ABLATION] Camera Perspective Sensitivity Study (deployed configuration):")
        print(ablation_df.to_string(index=False))

    # 6. Save Model Parameters & Evaluation Artifacts
    output = {
        "optimal_hyperparameters": {
            "alpha": deployed["alpha"],
            "slouch_thresh": deployed["slouch_ratio_drop_thresh"],
            "forward_head_z_thresh": deployed["forward_head_z_thresh"]
        },
        "loso_validation_metrics": front["loso_metrics"],
        "loso_macro_average": front["loso_macro"],
        "in_sample_metrics": front["in_sample_metrics"],
        "total_training_frames": len(df),
        "participant_count": len(persons),
        "loso_cross_validation": front["loso_records"],
        "benchmark_comparison": front["bench_df"].to_dict(orient="records"),
        "ablation_viewpoints": ablation_df.to_dict(orient="records") if ablation_df is not None else [],
        "view_benchmarks": {
            r["key"]: {
                "tuned_hyperparameters_in_sample": r["best_params"],
                "in_sample_metrics": r["in_sample_metrics"],
                "loso_validation_metrics": r["loso_metrics"],
                "loso_macro_average": r["loso_macro"],
                "total_frames": sum(rec["Frames"] for rec in r["loso_records"]),
                "participant_count": len(r["loso_records"]),
                "excluded_frames_without_participant_id": r["excluded_frames"],
                "loso_cross_validation": r["loso_records"],
                "benchmark_comparison": r["bench_df"].to_dict(orient="records")
            }
            for r in view_results
        }
    }
    if lateral_params:
        output["lateral_hyperparameters"] = {"alpha": lateral_params["lateral_alpha"], "cva_drop_thresh_deg": lateral_params["lateral_cva_drop_thresh_deg"]}

    weights_path = os.path.join(output_dir, "optimized_parameters.json")
    with open(weights_path, "w") as f:
        json.dump(output, f, indent=4)

    # 7. Generate Thesis Markdown Report
    report_path = os.path.join(output_dir, "training_thesis_report.md")
    generate_markdown_report(report_path, front, deployed_config, alert_window, len(df), ablation_df, view_results)
    print(f"\n[SAVED] Comprehensive thesis evaluation report written to: {report_path}")

    # 8. Generate Figures
    plot_training_results(front["grid_results"], deployed, front["loso_df"], front["bench_df"], output_dir)
    for result in view_results:
        plot_view_benchmark(result, output_dir)

def loso_table_with_summary(loso_df: pd.DataFrame, loso_macro: Dict, loso_metrics: Dict) -> pd.DataFrame:
    """LOSO table with summary rows (macro = mean of per-participant scores, pooled = all held-out frames together)."""
    metric_keys = {"Accuracy (%)": "accuracy", "Precision (%)": "precision", "Recall (%)": "recall", "Specificity (%)": "specificity", "F1-Score (%)": "f1_score"}
    summary_rows = []
    for label, source in [("Macro Average", loso_macro), ("Pooled (All Held-Out Frames)", loso_metrics)]:
        row = {col: "—" for col in loso_df.columns}
        row["Participant"] = label
        row["Frames"] = int(loso_df["Frames"].sum())
        row.update({col: round(source[key] * 100, 2) for col, key in metric_keys.items()})
        summary_rows.append(row)
    return pd.concat([loso_df, pd.DataFrame(summary_rows)], ignore_index=True)

def view_benchmark_section(number: int, result: Dict[str, Any]) -> str:
    """Markdown section for the LOSO benchmark of one camera view other than the primary front view."""
    loso_df, bench_df, m = result["loso_df"], result["bench_df"], result["loso_metrics"]
    n = len(loso_df)
    frames = int(loso_df["Frames"].sum())
    f1_min = loso_df.loc[loso_df["F1-Score (%)"].idxmin()]
    f1_max = loso_df.loc[loso_df["F1-Score (%)"].idxmax()]
    top_f1 = bench_df.loc[bench_df["F1-Score (%)"].idxmax()]

    if result["key"] == "lateral_90":
        excluded = f" (the {result['excluded_frames']} public benchmark images carry no participant ID and are excluded)" if result["excluded_frames"] else ""
        protocol = (
            f"The protocol of Sections 2 and 3, run on the {frames:,} lateral-profile frames of the {n} participants recorded side-on{excluded}. "
            "In this view ErgoEMA raises an alert when the EMA-smoothed craniovertebral angle (CVA) falls more than a tuned number of degrees below the participant's calibrated upright CVA; "
            f"the smoothing factor $\\alpha$ and that drop are tuned on the other {n - 1} participants. "
            "The state is named Forward Head Posture only when the CVA is also below the clinical $50^\\circ$ criterion; a smaller CVA with a normal angle is reported as a head-forward shift from the participant's upright. "
            f"The row *{CLINICAL_VARIANT_NAME}* scores the clinical criterion (CVA below $50^\\circ$) as the alert instead, with only $\\alpha$ tuned. "
            "Frames labelled slouch or forward head both count as poor posture, as in the other views. "
            "The ML baselines are trained on the side-view measurements (CVA, ear-to-shoulder offset, nose-to-shoulder angle), and the static baseline applies the $50^\\circ$ criterion to each frame without smoothing."
        )
    else:
        protocol = (
            f"The protocol of Sections 2 and 3, run on the {frames:,} oblique-profile frames (cameras $45^\\circ$ to the left and right) of all {n} participants. "
            f"ErgoEMA's hyperparameters are tuned on the other {n - 1} participants' $45^\\circ$ frames, and each participant's baseline is calibrated on their own $45^\\circ$ upright frames, so this measures a system set up for a $45^\\circ$ camera. "
            "Applying the front-view hyperparameters to these frames instead is the Oblique row of Section 4. The ML and static baselines use the same frontal features as in Section 3."
        )

    return f"""
---

## {number}. {result['title']} Benchmark

{protocol}

### {number}.1 Leave-One-Subject-Out Cross-Validation

{loso_table_with_summary(loso_df, result['loso_macro'], m).to_markdown(index=False)}

### {number}.2 Benchmark Architecture Comparison

{bench_df.to_markdown(index=False)}

On held-out participants, ErgoEMA reaches **{m['accuracy']*100:.2f}% accuracy**, **{m['f1_score']*100:.2f}% F1-score** and a **{m['false_alarm_rate']*100:.2f}% False Alarm Rate**; per-participant F1 ranges from {f1_min['F1-Score (%)']:.2f}% ({f1_min['Participant']}) to {f1_max['F1-Score (%)']:.2f}% ({f1_max['Participant']}). The highest held-out F1-score in the comparison is {top_f1['F1-Score (%)']:.2f}%, from {top_f1['Model Architecture']}.
"""

def generate_markdown_report(report_path: str, front: Dict[str, Any], deployed: Dict[str, Any], alert_window: float, total_frames: int, ablation_df: Optional[pd.DataFrame] = None, view_results: Sequence[Dict[str, Any]] = ()):
    """Writes formatted Markdown report for academic thesis."""
    loso_df, bench_df = front["loso_df"], front["bench_df"]
    loso_metrics, in_sample_metrics = front["loso_metrics"], front["in_sample_metrics"]

    ablation_section = ""
    if ablation_df is not None:
        lateral_note = (
            "The Lateral row uses the side-on settings tuned on those frames (Section 6), so it is also in-sample; the Oblique row uses the front-view settings, which were not tuned on its frames."
            if "lateral_alpha" in deployed else
            "The Oblique and Lateral rows use the front-view settings, which were not tuned on their frames."
        )
        ablation_section = f"""
---

## 4. Camera Viewpoint Sensitivity & Ablation Study (Chapter 4 Discussion)

The table below demonstrates the effect of camera placement geometry on biometric normalization. Every row runs the configuration the live application uses: the front-view hyperparameters from Section 1 for frontal and oblique frames, and the side-on settings for frames detected as lateral. The Front View row is an in-sample figure (those frames were used for tuning). {lateral_note}

{ablation_df.to_markdown(index=False)}

### Geometric Analysis:
* **Front View ($0^\\circ$)**: The 2D biacromial shoulder span ($\\|\\text{{Shoulder}}_R - \\text{{Shoulder}}_L\\|$) provides an invariant denominator, yielding **{ablation_df.iloc[0]['Accuracy (%)']}% Accuracy** and a **{ablation_df.iloc[0]['FAR (%)']}% False Alarm Rate**.
* **Oblique View ($45^\\circ$)**: Perspective foreshortening compresses the visible shoulder width, causing mild accuracy degradation compared to direct frontal placement.
* **Lateral View ($90^\\circ$)**: The alert tracks the drop of the estimated craniovertebral angle (CVA) below the participant's calibrated upright CVA, with C7 located on the back-of-neck silhouette; Forward Head Posture is named only when the CVA is also below the clinical $50^\\circ$ criterion. Thoracic kyphosis cannot be measured from video, so slouch is not assessed in this view.
"""

    first_view_section = 5 if ablation_section else 4
    view_sections = "".join(view_benchmark_section(first_view_section + i, result) for i, result in enumerate(view_results))

    loso_table = loso_table_with_summary(loso_df, front["loso_macro"], loso_metrics)
    n_participants = len(loso_df)
    f1_min = loso_df.loc[loso_df["F1-Score (%)"].idxmin()]
    f1_max = loso_df.loc[loso_df["F1-Score (%)"].idxmax()]
    top_f1 = bench_df.loc[bench_df["F1-Score (%)"].idxmax()]
    ergo_latency = bench_df.iloc[0]["Latency (ms)"]

    lateral_rows = ""
    if "lateral_alpha" in deployed:
        lateral_rows = f"""| **Side-On EMA Smoothing Factor** | $\\alpha_{{lat}}$ | **{deployed['lateral_alpha']}** | Used on frames detected as lateral; tuned on the lateral recordings (Section {first_view_section + 1}) |
| **CVA Drop Alert Threshold** | $\\Delta_{{CVA}}$ | **{deployed['lateral_cva_drop_thresh_deg']}°** | Alert when the CVA is this far below the calibrated upright CVA; tuned on the lateral recordings |
| **Clinical FHP Criterion** | — | **50°** | CVA below this is named Forward Head Posture (expert criterion, not tuned) |
"""

    content = f"""# ErgoEMA Model Training & Empirical Validation Report

**Dataset**: {total_frames:,} Real-World Front-Facing Video Frames
**Participants**: {n_participants} Study Participants ({', '.join(loso_df['Participant'])})
**Methodology**: Online Adaptive Calibration + Exponential Moving Average (EMA)
**Validation Protocol**: Leave-One-Subject-Out (hyperparameters tuned on {n_participants - 1} participants, scored on the held-out participant)

---

## 1. Deployed Hyperparameters

Tuned on all {n_participants} participants and saved to `optimized_parameters.json` for the live application:

| Hyperparameter | Symbol | Value | Description |
| :--- | :---: | :---: | :--- |
| **EMA Smoothing Factor** | $\\alpha$ | **{deployed['alpha']}** | Multi-channel noise suppression factor ($O(1)$ constant time) |
| **Slouch Ratio Drop Threshold** | $\\delta_{{slouch}}$ | **{deployed['slouch_ratio_drop_thresh'] * 100:.1f}%** | Percentage compression in $R_{{H2S}}$ below calibrated baseline |
| **Forward Head Depth Threshold** | $\\delta_{{FHP}}$ | **{deployed['forward_head_z_thresh']}** | Forward shift of the head depth $\\Delta Z$ beyond the calibrated baseline |
| **Sustained Alert Window** | $W_{{alert}}$ | **{alert_window} s** | Temporal buffer before raising alert. Set in `config.py`, not tuned: frames are scored on the classified posture state, which does not depend on this window |
{lateral_rows}
Scoring the front-view hyperparameters on the same {total_frames:,} frames they were tuned on gives {in_sample_metrics['accuracy']*100:.2f}% accuracy, {in_sample_metrics['f1_score']*100:.2f}% F1-score and a {in_sample_metrics['false_alarm_rate']*100:.2f}% False Alarm Rate. This is an in-sample figure, not a validation result.

---

## 2. Leave-One-Subject-Out (LOSO) Cross-Validation Results

Each row is scored on a participant whose frames were excluded from hyperparameter tuning; the "Tuned" columns show the values selected from the other {n_participants - 1} participants. The personalized baseline is still calibrated on the held-out participant's own first upright frames, as in live use.

{loso_table.to_markdown(index=False)}

---

## 3. Benchmark Architecture Comparison (Chapter 4 Results)

All models are scored under the same LOSO protocol: each participant is predicted by a model fitted on the other {n_participants - 1} participants. Latency is the median per-frame classification time over {LATENCY_SAMPLE_FRAMES:,} frames, measured on the machine that generated this report; it excludes MediaPipe pose estimation, which every model shares.

{bench_df.to_markdown(index=False)}

### Key Findings:
1. **Held-Out Performance**: On participants excluded from tuning, ErgoEMA reaches **{loso_metrics['accuracy']*100:.2f}% accuracy** and **{loso_metrics['f1_score']*100:.2f}% F1-score**. Per-participant F1 ranges from {f1_min['F1-Score (%)']:.2f}% ({f1_min['Participant']}) to {f1_max['F1-Score (%)']:.2f}% ({f1_max['Participant']}).
2. **False Alarm Rate**: The held-out False Alarm Rate (FAR) is **{loso_metrics['false_alarm_rate']*100:.2f}%**.
3. **Baseline Comparison**: The highest held-out F1-score in the table is {top_f1['F1-Score (%)']:.2f}%, from {top_f1['Model Architecture']}.
4. **Measured Latency**: The ErgoEMA EMA update and classification step takes a median of **{ergo_latency} ms** per frame.
{ablation_section}{view_sections}
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

# Short labels and colours for the benchmark figures
SHORT_MODEL_NAMES = {
    ERGOEMA_NAME: ("ErgoEMA", "#10B981"),
    CLINICAL_VARIANT_NAME: ("Clinical CVA", "#F59E0B"),
    "Random Forest (100 Trees)": ("Random Forest", "#3B82F6"),
    "Support Vector Machine (RBF)": ("SVM", "#6366F1"),
    "Logistic Regression": ("Logistic Reg.", "#8B5CF6"),
}

def plot_loso_panel(ax, loso_df: pd.DataFrame, title: str):
    """Bar chart of held-out accuracy and F1-score per participant."""
    sub_names = [name.replace("Participant", "P") for name in loso_df["Participant"]]
    x = np.arange(len(loso_df))
    width = 0.35
    ax.bar(x - width/2, loso_df["Accuracy (%)"], width, label="Accuracy", color="#2563EB")
    ax.bar(x + width/2, loso_df["F1-Score (%)"], width, label="F1-Score", color="#10B981")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sub_names, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

def plot_benchmark_panel(ax, bench_df: pd.DataFrame, title: str):
    """Bar chart of the held-out F1-score of each model in the benchmark table."""
    labels = [SHORT_MODEL_NAMES.get(name, ("Static Rule", "#EF4444")) for name in bench_df["Model Architecture"]]
    bars = ax.bar([label for label, _ in labels], bench_df["F1-Score (%)"].values, color=[color for _, color in labels], width=0.55)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("F1-Score (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='x', rotation=20)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.6)

def plot_training_results(grid_results: List[Dict], deployed: Dict[str, Any], loso_df: pd.DataFrame, bench_df: pd.DataFrame, output_dir: str):
    """Generates 3-panel publication figure for Chapter 4."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel 1: Hyperparameter Sensitivity Curves (in-sample, all participants) at the deployed forward-head threshold
    grid_df = pd.DataFrame(grid_results)
    grid_df = grid_df[grid_df["forward_head_z_thresh"] == deployed["forward_head_z_thresh"]]
    pivot = grid_df.groupby(["alpha", "slouch_ratio_drop_thresh"])["f1_score"].mean().unstack()
    for alpha_val in pivot.index:
        axes[0].plot(pivot.columns * 100, pivot.loc[alpha_val] * 100, marker='o', label=f"EMA $\\alpha$ = {alpha_val}")
    axes[0].set_title(f"(a) In-Sample F1 vs. Slouch Drop (FHP depth {deployed['forward_head_z_thresh']})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Slouch Ratio Drop $\\delta_{slouch}$ (%)", fontsize=10)
    axes[0].set_ylabel("F1-Score (%)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: LOSO Cross-Validation per Held-Out Participant
    plot_loso_panel(axes[1], loso_df, f"(b) LOSO Held-Out Validation ({len(loso_df)} Participants)")

    # Panel 3: Architecture Benchmark Comparison
    plot_benchmark_panel(axes[2], bench_df, "(c) Benchmark Architecture Comparison (LOSO F1-Score)")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "training_optimization_curves.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[SAVED] Publication figures saved to: {chart_path}")

def plot_view_benchmark(result: Dict[str, Any], output_dir: str):
    """Two-panel figure (LOSO per participant, model comparison) for the benchmark of one camera view."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_loso_panel(axes[0], result["loso_df"], f"(a) {result['title']}: LOSO Held-Out Validation ({len(result['loso_df'])} Participants)")
    plot_benchmark_panel(axes[1], result["bench_df"], f"(b) {result['title']}: Benchmark Comparison (LOSO F1-Score)")
    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"benchmark_{result['key']}.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[SAVED] {result['title']} benchmark figure saved to: {chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize ErgoEMA model on processed dataset")
    parser.add_argument("--data", type=str, default="data/processed/combined_dataset.csv", help="Path to cleaned dataset CSV")
    parser.add_argument("--output", type=str, default="results", help="Output directory for reports & plots")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the camera viewpoint ablation and the 45°/90° benchmarks (e.g. when evaluating a subset such as the pilot dataset)")
    args = parser.parse_args()

    train_and_optimize(dataset_path=args.data, output_dir=args.output, run_ablation=not args.skip_ablation)
