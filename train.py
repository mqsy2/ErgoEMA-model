"""
ErgoEMA Model Training, Hyperparameter Optimization, and Benchmark Suite.

Performs:
1. Personalized Calibration & Adaptive EMA Simulation grouped by participant.
2. Grid Search Optimization of ErgoEMA hyperparameters (alpha_EMA, delta_slouch).
3. Leave-One-Subject-Out (LOSO) Cross-Validation: hyperparameters are tuned on N-1 participants and scored on the held-out one.
4. Machine Learning Benchmarking under the same LOSO protocol (ErgoEMA vs. Random Forest vs. SVM vs. Logistic Regression vs. Static Baseline), with measured per-frame latency.
5. Generates thesis publication tables, JSON parameters, and visual figures for Chapter 4.
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Sequence, Tuple, Optional

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

def extract_participant(subject_str: str) -> str:
    """Extracts clean human subject name or participant ID from subject string."""
    s = str(subject_str).strip()
    if s.startswith("Participant"):
        return s.split("_")[0]
    if " - " in s:
        return s.split(" - ")[-1].strip()
    return s

def build_baseline(pgroup: pd.DataFrame, person: str) -> BaselineProfile:
    """Calibrates a personalized baseline on the participant's first upright frames."""
    upright_frames = pgroup[pgroup["label"] == "upright"]
    if len(upright_frames) >= 15:
        calib_sample = upright_frames.iloc[:60]
    else:
        calib_sample = pgroup.iloc[:min(60, len(pgroup))]

    return BaselineProfile(
        mean_h2s_ratio=float(calib_sample["h2s_ratio"].mean()),
        std_h2s_ratio=float(calib_sample["h2s_ratio"].std() if len(calib_sample) > 1 else 0.04),
        mean_shoulder_tilt_deg=float(calib_sample["shoulder_tilt_deg"].mean()),
        std_shoulder_tilt_deg=float(calib_sample["shoulder_tilt_deg"].std() if len(calib_sample) > 1 else 1.0),
        mean_head_roll_deg=float(calib_sample["head_roll_deg"].mean()),
        std_head_roll_deg=float(calib_sample["head_roll_deg"].std() if len(calib_sample) > 1 else 1.0),
        mean_forward_head_z=float(calib_sample["forward_head_z"].mean()),
        std_forward_head_z=float(calib_sample["forward_head_z"].std() if len(calib_sample) > 1 else 0.05),
        mean_shoulder_width_norm=float(calib_sample["shoulder_width_norm"].mean()),
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
        nose_shoulder_angle_deg=float(getattr(row, "nose_shoulder_angle_deg", 0.0))
    )

def run_ergoema_simulation(
    df: pd.DataFrame,
    alpha: float,
    slouch_thresh: float,
    sustained_sec: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Simulates personalized online ErgoEMA monitoring on participant time-series data.
    Returns (y_true, y_pred, per_subject_metrics).
    """
    df = df.copy()
    if "participant_id" in df.columns:
        df["person"] = df["participant_id"]
    else:
        df["person"] = df["subject_id"].apply(extract_participant)

    y_true_all = []
    y_pred_all = []
    per_person_metrics = {}

    for person, pgroup in df.groupby("person", sort=False):
        ema_filter = EMAFilter(alpha=alpha)
        classifier = PostureClassifier(
            slouch_ratio_drop_thresh=slouch_thresh,
            sustained_window_sec=sustained_sec
        )

        # 1. Calibrate on the participant's upright frames
        baseline = build_baseline(pgroup, person)

        y_true_p = []
        y_pred_p = []
        current_t = 0.0

        for row in pgroup.itertuples(index=False):
            current_t += 0.033
            smoothed = ema_filter.update(row_to_features(row, current_t))

            assessment = classifier.evaluate_adaptive(
                smoothed_feat=smoothed,
                baseline=baseline,
                current_time=current_t
            )

            # Map: 1 = Poor Posture / Violation (Slouch, FHP), 0 = Good / Upright
            pred_binary = 1 if assessment.state != PostureState.GOOD_POSTURE else 0
            true_binary = 1 if getattr(row, "label") in ["slouch", "forward_head"] else 0

            y_pred_p.append(pred_binary)
            y_true_p.append(true_binary)

        per_person_metrics[person] = compute_metrics(np.array(y_true_p), np.array(y_pred_p))
        y_true_all.extend(y_true_p)
        y_pred_all.extend(y_pred_p)

    return np.array(y_true_all), np.array(y_pred_all), per_person_metrics

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

def select_best_params(
    grid_person_metrics: Dict[Tuple[float, float], Dict[str, Dict[str, float]]],
    persons: Sequence[str]
) -> Tuple[Tuple[float, float], Dict[str, float]]:
    """Returns the (alpha, slouch_thresh) with the highest pooled F1-score over the given participants."""
    best_key, best_metrics = None, None
    for key, person_metrics in grid_person_metrics.items():
        metrics = pool_metrics(person_metrics, persons)
        if best_metrics is None or metrics["f1_score"] > best_metrics["f1_score"]:
            best_key, best_metrics = key, metrics
    return best_key, best_metrics

def measure_latency_ms(step: Callable, inputs: Sequence) -> float:
    """Median wall-clock time (ms) of one call to `step` per input frame, measured on this machine."""
    timings = []
    for item in inputs:
        t0 = time.perf_counter()
        step(item)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(timings))

def measure_ergoema_latency_ms(df: pd.DataFrame, alpha: float, slouch_thresh: float, sustained_sec: float) -> float:
    """Median per-frame time (ms) of the EMA update + adaptive classification on this machine."""
    person = df["person"].iloc[0]
    pgroup = df[df["person"] == person].iloc[:LATENCY_SAMPLE_FRAMES]

    ema_filter = EMAFilter(alpha=alpha)
    classifier = PostureClassifier(slouch_ratio_drop_thresh=slouch_thresh, sustained_window_sec=sustained_sec)
    baseline = build_baseline(pgroup, person)
    feats = [row_to_features(row, (i + 1) * 0.033) for i, row in enumerate(pgroup.itertuples(index=False))]

    return measure_latency_ms(
        lambda feat: classifier.evaluate_adaptive(ema_filter.update(feat), baseline, feat.timestamp),
        feats
    )

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

def train_and_optimize(dataset_path: str = "data/processed/combined_dataset.csv", output_dir: str = "results", run_ablation: bool = True):
    """Runs full model training, parameter optimization, LOSO cross-validation, and ML comparisons."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 72)
    print("      ErgoEMA Model Training, Parameter Optimization & LOSO Validation")
    print("=" * 72)

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    if "participant_id" in df.columns:
        df["person"] = df["participant_id"]
    else:
        df["person"] = df["subject_id"].apply(extract_participant)

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

    # 1. Grid Search Optimization for ErgoEMA Hyperparameters
    print("\n[OPTIMIZATION] Running Grid Search on EMA smoothing factor & slouch threshold...")
    alphas = [0.08, 0.12, 0.15, 0.20, 0.25]
    slouch_thresholds = [0.08, 0.10, 0.12, 0.15, 0.18]

    # Frames are scored on the classified posture state, which does not depend on the
    # sustained alert window, so the window is taken from config rather than tuned.
    alert_window = ErgoConfig().SUSTAINED_ALERT_WINDOW_SEC

    # Each participant is simulated independently, so one pass per grid point yields
    # the per-participant counts needed for both the pooled fit and every LOSO fold.
    grid_person_metrics = {}
    grid_results = []

    for alpha in alphas:
        for sthresh in slouch_thresholds:
            _, _, person_metrics = run_ergoema_simulation(df, alpha, sthresh, alert_window)
            grid_person_metrics[(alpha, sthresh)] = person_metrics
            metrics = pool_metrics(person_metrics, persons)

            grid_results.append({
                "alpha": alpha,
                "slouch_thresh": sthresh,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "false_alarm_rate": metrics["false_alarm_rate"]
            })

    # Deployed hyperparameters: tuned on all participants (in-sample, not a validation result)
    (best_alpha, best_sthresh), in_sample_metrics = select_best_params(grid_person_metrics, persons)
    best_params = {"alpha": best_alpha, "slouch_thresh": best_sthresh}

    print(f"\n[DEPLOYED HYPERPARAMETERS] Tuned on all {len(persons)} participants:")
    print(f"  * EMA Smoothing (alpha)             : {best_params['alpha']}")
    print(f"  * Slouch Ratio Drop (delta)         : {best_params['slouch_thresh'] * 100:.1f}%")
    print(f"  * Sustained Alert Window (config)   : {alert_window} sec")
    print(f"  * In-Sample Accuracy                : {in_sample_metrics['accuracy'] * 100:.2f}%")
    print(f"  * In-Sample F1-Score                : {in_sample_metrics['f1_score'] * 100:.2f}%")
    print(f"  * In-Sample False Alarm Rate (FAR)  : {in_sample_metrics['false_alarm_rate'] * 100:.2f}%")

    # 2. Leave-One-Subject-Out (LOSO) Cross-Validation
    print("\n[LOSO CROSS-VALIDATION] Tuned on N-1 participants, scored on the held-out participant:")
    loso_records = []
    held_out_metrics = {}
    for held_out in persons:
        train_persons = [p for p in persons if p != held_out]
        fold_key, _ = select_best_params(grid_person_metrics, train_persons)
        m = grid_person_metrics[fold_key][held_out]
        held_out_metrics[held_out] = m

        sub_df = df[df["person"] == held_out]
        record = {"Participant": held_out}
        if "somatotype" in sub_df.columns:
            record["Somatotype"] = sub_df["somatotype"].iloc[0]
        if "bmi_category" in sub_df.columns:
            record["BMI Category"] = sub_df["bmi_category"].iloc[0]
        record.update({
            "Frames": len(sub_df),
            "Tuned alpha": fold_key[0],
            "Tuned Slouch Drop (%)": round(fold_key[1] * 100, 1),
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Precision (%)": round(m["precision"] * 100, 2),
            "Recall (%)": round(m["recall"] * 100, 2),
            "Specificity (%)": round(m["specificity"] * 100, 2),
            "F1-Score (%)": round(m["f1_score"] * 100, 2)
        })
        loso_records.append(record)

    loso_df = pd.DataFrame(loso_records)
    print(loso_df.to_string(index=False))

    loso_metrics = pool_metrics(held_out_metrics, persons)
    loso_macro = {
        k: float(np.mean([held_out_metrics[p][k] for p in persons]))
        for k in ("accuracy", "precision", "recall", "specificity", "f1_score", "false_alarm_rate")
    }
    print(f"\n  * LOSO Held-Out Accuracy (pooled)   : {loso_metrics['accuracy'] * 100:.2f}%")
    print(f"  * LOSO Held-Out F1-Score (pooled)   : {loso_metrics['f1_score'] * 100:.2f}%")
    print(f"  * LOSO Sensitivity / Recall         : {loso_metrics['recall'] * 100:.2f}%")
    print(f"  * LOSO Specificity                  : {loso_metrics['specificity'] * 100:.2f}%")
    print(f"  * LOSO False Alarm Rate (FAR)       : {loso_metrics['false_alarm_rate'] * 100:.2f}%")

    # 3. Benchmark Comparisons Against ML Models for Chapter 4 (same LOSO protocol)
    print("\n[MODEL BENCHMARK] Comparing ErgoEMA vs Machine Learning Baselines (LOSO, measured latency)...")
    feature_cols = ["h2s_ratio", "shoulder_tilt_deg", "head_roll_deg", "forward_head_z", "neck_flexion_deg", "shoulder_width_norm"]
    X = df[feature_cols].values
    y = df["label"].isin(["slouch", "forward_head"]).astype(int).values
    groups = df["person"].values
    logo = LeaveOneGroupOut()
    latency_idx = np.linspace(0, len(df) - 1, min(LATENCY_SAMPLE_FRAMES, len(df))).astype(int)

    bench_rows = [benchmark_row(
        "ErgoEMA (Adaptive Time-Series)",
        loso_metrics,
        measure_ergoema_latency_ms(df, best_params["alpha"], best_params["slouch_thresh"], alert_window)
    )]

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

    # Static Rigid Threshold Baseline (Non-adaptive; threshold derived from the training participants only)
    h2s = df["h2s_ratio"].values
    y_static = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        y_static[test_idx] = (h2s[test_idx] < np.median(h2s[train_idx]) * 0.90).astype(int)
    static_thresh = float(np.median(h2s) * 0.90)
    latency_static = measure_latency_ms(lambda ratio: ratio < static_thresh, [float(h2s[i]) for i in latency_idx])
    bench_rows.append(benchmark_row("Static Rigid Threshold", compute_metrics(y, y_static), latency_static))

    bench_df = pd.DataFrame(bench_rows)
    print(bench_df.to_string(index=False))

    # 5. Optional Viewpoint Ablation Analysis (Front 0° vs Oblique 45° vs Lateral 90° vs Combined)
    ablation_df = None
    all_angles_path = "data/processed/all_angles_dataset.csv"
    if run_ablation and os.path.exists(all_angles_path):
        try:
            df_all = pd.read_csv(all_angles_path)
            if "view_angle" in df_all.columns:
                df_f = df_all[df_all["view_angle"] == "Front"]
                df_s = df_all[df_all["view_angle"] == "Side"]
                df_90 = df_all[df_all["view_angle"] == "90deg"]

                ablation_rows = []

                # Front View (0°)
                if len(df_f) > 0:
                    y_tf, y_pf, _ = run_ergoema_simulation(df_f, best_params["alpha"], best_params["slouch_thresh"], alert_window)
                    mf = compute_metrics(y_tf, y_pf)
                    ablation_rows.append({"Camera Perspective": "Front View (0° - Primary Scope)", "Frames": len(df_f), "Accuracy (%)": round(mf["accuracy"]*100, 2), "Precision (%)": round(mf["precision"]*100, 2), "Recall (%)": round(mf["recall"]*100, 2), "F1-Score (%)": round(mf["f1_score"]*100, 2), "FAR (%)": round(mf["false_alarm_rate"]*100, 2)})

                # Oblique Profile (45°)
                if len(df_s) > 0:
                    y_ts, y_ps, _ = run_ergoema_simulation(df_s, best_params["alpha"], best_params["slouch_thresh"], alert_window)
                    ms = compute_metrics(y_ts, y_ps)
                    ablation_rows.append({"Camera Perspective": "Oblique Profile (45° View)", "Frames": len(df_s), "Accuracy (%)": round(ms["accuracy"]*100, 2), "Precision (%)": round(ms["precision"]*100, 2), "Recall (%)": round(ms["recall"]*100, 2), "F1-Score (%)": round(ms["f1_score"]*100, 2), "FAR (%)": round(ms["false_alarm_rate"]*100, 2)})

                # Lateral Profile (90° View)
                if len(df_90) > 0:
                    y_t90, y_p90, _ = run_ergoema_simulation(df_90, best_params["alpha"], best_params["slouch_thresh"], alert_window)
                    m90 = compute_metrics(y_t90, y_p90)
                    ablation_rows.append({"Camera Perspective": "Lateral Profile (90° View)", "Frames": len(df_90), "Accuracy (%)": round(m90["accuracy"]*100, 2), "Precision (%)": round(m90["precision"]*100, 2), "Recall (%)": round(m90["recall"]*100, 2), "F1-Score (%)": round(m90["f1_score"]*100, 2), "FAR (%)": round(m90["false_alarm_rate"]*100, 2)})

                # All Combined
                y_ta, y_pa, _ = run_ergoema_simulation(df_all, best_params["alpha"], best_params["slouch_thresh"], alert_window)
                ma = compute_metrics(y_ta, y_pa)
                ablation_rows.append({"Camera Perspective": "All Combined Perspectives", "Frames": len(df_all), "Accuracy (%)": round(ma["accuracy"]*100, 2), "Precision (%)": round(ma["precision"]*100, 2), "Recall (%)": round(ma["recall"]*100, 2), "F1-Score (%)": round(ma["f1_score"]*100, 2), "FAR (%)": round(ma["false_alarm_rate"]*100, 2)})

                ablation_df = pd.DataFrame(ablation_rows)
                print("\n[ABLATION] Camera Perspective Sensitivity Study (deployed hyperparameters, in-sample):")
                print(ablation_df.to_string(index=False))
        except Exception as e:
            print(f"[WARN] Viewpoint ablation skipped: {e}")

    # 6. Save Model Parameters & Evaluation Artifacts
    weights_path = os.path.join(output_dir, "optimized_parameters.json")
    with open(weights_path, "w") as f:
        json.dump({
            "optimal_hyperparameters": best_params,
            "loso_validation_metrics": loso_metrics,
            "loso_macro_average": loso_macro,
            "in_sample_metrics": in_sample_metrics,
            "total_training_frames": len(df),
            "participant_count": len(persons),
            "loso_cross_validation": loso_records,
            "benchmark_comparison": bench_df.to_dict(orient="records"),
            "ablation_viewpoints": ablation_df.to_dict(orient="records") if ablation_df is not None else []
        }, f, indent=4)

    # 7. Generate Thesis Markdown Report
    report_path = os.path.join(output_dir, "training_thesis_report.md")
    generate_markdown_report(report_path, best_params, alert_window, loso_metrics, loso_macro, in_sample_metrics, loso_df, bench_df, len(df), ablation_df)
    print(f"\n[SAVED] Comprehensive thesis evaluation report written to: {report_path}")

    # 8. Generate Figures
    plot_training_results(grid_results, loso_df, bench_df, output_dir)

def generate_markdown_report(report_path: str, best_params: Dict, alert_window: float, loso_metrics: Dict, loso_macro: Dict, in_sample_metrics: Dict, loso_df: pd.DataFrame, bench_df: pd.DataFrame, total_frames: int, ablation_df: Optional[pd.DataFrame] = None):
    """Writes formatted Markdown report for academic thesis."""
    ablation_section = ""
    if ablation_df is not None:
        ablation_section = f"""
---

## 4. Camera Viewpoint Sensitivity & Ablation Study (Chapter 4 Discussion)

The table below demonstrates the effect of camera placement geometry on biometric normalization. Every row uses the deployed hyperparameters from Section 1, so the Front View row is an in-sample figure (those frames were used for tuning), not a held-out result:

{ablation_df.to_markdown(index=False)}

### Geometric Analysis:
* **Front View ($0^\\circ$)**: The 2D biacromial shoulder span ($\\|\\text{{Shoulder}}_R - \\text{{Shoulder}}_L\\|$) provides an invariant denominator, yielding **{ablation_df.iloc[0]['Accuracy (%)']}% Accuracy** and a **{ablation_df.iloc[0]['FAR (%)']}% False Alarm Rate**.
* **Oblique View ($45^\\circ$)**: Perspective foreshortening compresses the visible shoulder width, causing mild accuracy degradation compared to direct frontal placement.
* **Lateral View ($90^\\circ$)**: Utilizes invariant lateral biometrics (ear-to-shoulder horizontal displacement and nose-to-shoulder vertical inclination angle) calibrated to clinical hyperkyphosis (>55°) and FHP (>0.14) criteria, maintaining high accuracy across both left-facing and right-facing camera orientations.
"""

    # LOSO table with summary rows (macro = mean of per-participant scores, pooled = all held-out frames together)
    metric_keys = {"Accuracy (%)": "accuracy", "Precision (%)": "precision", "Recall (%)": "recall", "Specificity (%)": "specificity", "F1-Score (%)": "f1_score"}
    summary_rows = []
    for label, source in [("Macro Average", loso_macro), ("Pooled (All Held-Out Frames)", loso_metrics)]:
        row = {col: "—" for col in loso_df.columns}
        row["Participant"] = label
        row["Frames"] = total_frames
        row.update({col: round(source[key] * 100, 2) for col, key in metric_keys.items()})
        summary_rows.append(row)
    loso_table = pd.concat([loso_df, pd.DataFrame(summary_rows)], ignore_index=True)

    n_participants = len(loso_df)
    f1_min = loso_df.loc[loso_df["F1-Score (%)"].idxmin()]
    f1_max = loso_df.loc[loso_df["F1-Score (%)"].idxmax()]
    top_f1 = bench_df.loc[bench_df["F1-Score (%)"].idxmax()]
    ergo_latency = bench_df.iloc[0]["Latency (ms)"]

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
| **EMA Smoothing Factor** | $\\alpha$ | **{best_params['alpha']}** | Multi-channel noise suppression factor ($O(1)$ constant time) |
| **Slouch Ratio Drop Threshold** | $\\delta_{{slouch}}$ | **{best_params['slouch_thresh'] * 100:.1f}%** | Percentage compression in $R_{{H2S}}$ below calibrated baseline |
| **Sustained Alert Window** | $W_{{alert}}$ | **{alert_window} s** | Temporal buffer before raising alert. Set in `config.py`, not tuned: frames are scored on the classified posture state, which does not depend on this window |

Scoring these hyperparameters on the same {total_frames:,} frames they were tuned on gives {in_sample_metrics['accuracy']*100:.2f}% accuracy, {in_sample_metrics['f1_score']*100:.2f}% F1-score and a {in_sample_metrics['false_alarm_rate']*100:.2f}% False Alarm Rate. This is an in-sample figure, not a validation result.

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
{ablation_section}
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

def plot_training_results(grid_results: List[Dict], loso_df: pd.DataFrame, bench_df: pd.DataFrame, output_dir: str):
    """Generates 3-panel publication figure for Chapter 4."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel 1: Hyperparameter Sensitivity Curves (in-sample, all participants)
    grid_df = pd.DataFrame(grid_results)
    pivot = grid_df.groupby(["alpha", "slouch_thresh"])["f1_score"].mean().unstack()
    for alpha_val in pivot.index:
        axes[0].plot(pivot.columns * 100, pivot.loc[alpha_val] * 100, marker='o', label=f"EMA $\\alpha$ = {alpha_val}")
    axes[0].set_title("(a) In-Sample F1-Score vs. Slouch Drop Threshold", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Slouch Ratio Drop $\\delta_{slouch}$ (%)", fontsize=10)
    axes[0].set_ylabel("F1-Score (%)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: LOSO Cross-Validation per Held-Out Participant
    sub_names = [name.replace("Participant", "P") for name in loso_df["Participant"]]
    x = np.arange(len(loso_df))
    width = 0.35
    axes[1].bar(x - width/2, loso_df["Accuracy (%)"], width, label="Accuracy", color="#2563EB")
    axes[1].bar(x + width/2, loso_df["F1-Score (%)"], width, label="F1-Score", color="#10B981")
    axes[1].set_title(f"(b) LOSO Held-Out Validation ({len(loso_df)} Participants)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sub_names, fontsize=10)
    axes[1].set_ylabel("Score (%)", fontsize=10)
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Architecture Benchmark Comparison
    models_short = ["ErgoEMA", "Random Forest", "SVM", "Logistic Reg.", "Static Rule"]
    f1_scores = bench_df["F1-Score (%)"].values
    colors = ["#10B981", "#3B82F6", "#6366F1", "#8B5CF6", "#EF4444"]
    bars = axes[2].bar(models_short, f1_scores, color=colors, width=0.55)
    axes[2].set_title("(c) Benchmark Architecture Comparison (LOSO F1-Score)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("F1-Score (%)", fontsize=10)
    axes[2].set_ylim(0, 105)
    axes[2].tick_params(axis='x', rotation=20)
    for bar in bars:
        height = bar.get_height()
        axes[2].annotate(f'{height:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "training_optimization_curves.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[SAVED] Publication figures saved to: {chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize ErgoEMA model on processed dataset")
    parser.add_argument("--data", type=str, default="data/processed/combined_dataset.csv", help="Path to cleaned dataset CSV")
    parser.add_argument("--output", type=str, default="results", help="Output directory for reports & plots")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the camera viewpoint ablation (e.g. when evaluating a subset such as the pilot dataset)")
    args = parser.parse_args()

    train_and_optimize(dataset_path=args.data, output_dir=args.output, run_ablation=not args.skip_ablation)
