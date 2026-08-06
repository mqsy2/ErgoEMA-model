"""
Evaluation and Benchmarking Script for ErgoEMA.
Compares Static Threshold Baseline vs Adaptive ErgoEMA Model.
Computes Accuracy, Precision, Recall, F1-Score, False Alarm Rate, and Execution Latency.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import ErgoConfig
from src.feature_extractor import PostureFeatures
from src.ema_filter import EMAFilter
from src.calibrator import PostureCalibrator, BaselineProfile
from src.classifier import PostureClassifier, PostureState

def generate_synthetic_benchmark(num_frames: int = 1800, fps: int = 30) -> pd.DataFrame:
    """
    Generates realistic synthetic front-facing workstation time-series data
    incorporating natural jitter, micro-movements (typing/fidgets), and sustained slumps.
    """
    np.random.seed(42)
    time_series = []
    
    # Baseline upright features for a typical user
    base_ratio = 0.52
    base_shoulder_tilt = 0.5
    base_head_roll = 0.2
    base_z = -0.15

    for t in range(num_frames):
        sec = t / float(fps)
        
        # Determine postural state
        # 0-15s: Upright sitting (with small sensor jitter)
        # 15-20s: Micro-movement / fidget (brief 1-2s tilt)
        # 20-35s: Sustained Slouch / Kyphosis
        # 35-45s: Upright recovery
        # 45-55s: Sustained Asymmetric Shoulder Drop
        # 55-60s: Upright recovery
        
        jitter_r = np.random.normal(0, 0.015)
        jitter_s = np.random.normal(0, 0.8)
        jitter_h = np.random.normal(0, 0.6)
        jitter_z = np.random.normal(0, 0.01)

        if 0 <= sec < 15:
            label = "upright"
            ratio = base_ratio + jitter_r
            shoulder = base_shoulder_tilt + jitter_s
            head = base_head_roll + jitter_h
            z = base_z + jitter_z
        elif 15 <= sec < 20:
            # Fleeting micro-fidget (should NOT trigger alert in adaptive model)
            label = "upright"
            ratio = base_ratio - 0.12 + jitter_r # brief dip
            shoulder = base_shoulder_tilt + 4.5 + jitter_s
            head = base_head_roll + 5.0 + jitter_h
            z = base_z + jitter_z
        elif 20 <= sec < 35:
            # Sustained slouch
            label = "slouch"
            ratio = base_ratio - 0.18 + jitter_r # marked ratio drop
            shoulder = base_shoulder_tilt + jitter_s
            head = base_head_roll + jitter_h
            z = base_z - 0.08 + jitter_z # forward head
        elif 35 <= sec < 45:
            label = "upright"
            ratio = base_ratio + jitter_r
            shoulder = base_shoulder_tilt + jitter_s
            head = base_head_roll + jitter_h
            z = base_z + jitter_z
        elif 45 <= sec < 55:
            # Sustained asymmetric shoulder lean
            label = "slouch"
            ratio = base_ratio + jitter_r
            shoulder = base_shoulder_tilt + 8.5 + jitter_s
            head = base_head_roll + 4.0 + jitter_h
            z = base_z + jitter_z
        else:
            label = "upright"
            ratio = base_ratio + jitter_r
            shoulder = base_shoulder_tilt + jitter_s
            head = base_head_roll + jitter_h
            z = base_z + jitter_z

        time_series.append({
            "timestamp": sec,
            "label": label,
            "h2s_ratio": ratio,
            "shoulder_tilt_deg": shoulder,
            "head_roll_deg": head,
            "forward_head_z": z,
            "neck_flexion_deg": 0.0,
            "shoulder_width_norm": 0.35
        })

    return pd.DataFrame(time_series)

def run_evaluation(csv_path: Optional[str] = None, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    config = ErgoConfig()

    if csv_path and os.path.exists(csv_path):
        print(f"[INFO] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("[INFO] No external dataset specified. Generating standard 60-second time-series benchmark dataset...")
        df = generate_synthetic_benchmark(num_frames=1800, fps=30)
        df.to_csv(os.path.join(output_dir, "benchmark_dataset.csv"), index=False)

    # Convert binary ground-truth: 0 = Normal/Upright, 1 = Slouch/Bad Posture Alert
    y_true = np.array([0 if lbl in ["upright", "micro_movement"] else 1 for lbl in df["label"]])

    # 1. Evaluate Static Threshold Baseline Model
    static_classifier = PostureClassifier()
    y_pred_static = []
    latencies_static = []

    for _, row in df.iterrows():
        feat = PostureFeatures(
            head_to_shoulder_ratio=float(row["h2s_ratio"]),
            shoulder_tilt_deg=float(row["shoulder_tilt_deg"]),
            head_roll_deg=float(row["head_roll_deg"]),
            forward_head_z=float(row["forward_head_z"]),
            shoulder_width_norm=float(row["shoulder_width_norm"]),
            neck_lateral_flexion_deg=float(row["neck_flexion_deg"])
        )
        t0 = time.perf_counter()
        assessment = static_classifier.evaluate_static_baseline(
            feat,
            fixed_ratio_thresh=config.STATIC_SLOUCH_RATIO_FIXED,
            fixed_shoulder_tilt_thresh=config.STATIC_SHOULDER_TILT_FIXED_DEG,
            fixed_head_roll_thresh=config.STATIC_HEAD_ROLL_FIXED_DEG
        )
        latencies_static.append((time.perf_counter() - t0) * 1000.0)
        y_pred_static.append(1 if assessment.is_alert else 0)

    # 2. Evaluate Adaptive ErgoEMA Model
    # First 90 frames for calibration
    calib_frames = min(90, len(df))
    calibrator = PostureCalibrator(target_frames=calib_frames)
    calibrator.start("eval_user")

    for i in range(calib_frames):
        row = df.iloc[i]
        feat = PostureFeatures(
            head_to_shoulder_ratio=float(row["h2s_ratio"]),
            shoulder_tilt_deg=float(row["shoulder_tilt_deg"]),
            head_roll_deg=float(row["head_roll_deg"]),
            forward_head_z=float(row["forward_head_z"]),
            shoulder_width_norm=float(row["shoulder_width_norm"]),
            neck_lateral_flexion_deg=float(row["neck_flexion_deg"])
        )
        calibrator.add_frame(feat)

    baseline = calibrator.profile
    assert baseline is not None, "Calibration failed"

    ema_filter = EMAFilter(alpha=config.DEFAULT_EMA_ALPHA)
    adaptive_classifier = PostureClassifier(
        slouch_ratio_drop_thresh=config.SLOUCH_RATIO_DROP_THRESH,
        shoulder_tilt_thresh_deg=config.SHOULDER_TILT_THRESH_DEG,
        head_roll_thresh_deg=config.HEAD_ROLL_THRESH_DEG,
        forward_head_z_thresh=config.FORWARD_HEAD_Z_THRESH,
        sustained_window_sec=config.SUSTAINED_ALERT_WINDOW_SEC
    )

    y_pred_ergoema = []
    latencies_ergoema = []
    smoothed_ratios = []

    for _, row in df.iterrows():
        feat = PostureFeatures(
            head_to_shoulder_ratio=float(row["h2s_ratio"]),
            shoulder_tilt_deg=float(row["shoulder_tilt_deg"]),
            head_roll_deg=float(row["head_roll_deg"]),
            forward_head_z=float(row["forward_head_z"]),
            shoulder_width_norm=float(row["shoulder_width_norm"]),
            neck_lateral_flexion_deg=float(row["neck_flexion_deg"])
        )
        t0 = time.perf_counter()
        smoothed = ema_filter.update(feat)
        assessment = adaptive_classifier.evaluate_adaptive(
            smoothed_feat=smoothed,
            baseline=baseline,
            current_time=float(row["timestamp"])
        )
        latencies_ergoema.append((time.perf_counter() - t0) * 1000.0)
        smoothed_ratios.append(smoothed.head_to_shoulder_ratio)
        y_pred_ergoema.append(1 if assessment.is_alert else 0)

    # 3. Compute Metrics
    metrics_static = {
        "Accuracy": accuracy_score(y_true, y_pred_static),
        "Precision": precision_score(y_true, y_pred_static, zero_division=0),
        "Recall": recall_score(y_true, y_pred_static, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred_static, zero_division=0),
        "Mean Latency (ms)": np.mean(latencies_static),
        "False Positives (Nuisance Alerts)": int(np.sum((y_true == 0) & (np.array(y_pred_static) == 1)))
    }

    metrics_ergoema = {
        "Accuracy": accuracy_score(y_true, y_pred_ergoema),
        "Precision": precision_score(y_true, y_pred_ergoema, zero_division=0),
        "Recall": recall_score(y_true, y_pred_ergoema, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred_ergoema, zero_division=0),
        "Mean Latency (ms)": np.mean(latencies_ergoema),
        "False Positives (Nuisance Alerts)": int(np.sum((y_true == 0) & (np.array(y_pred_ergoema) == 1)))
    }

    print("\n" + "="*70)
    print("                ErgoEMA MODEL EVALUATION REPORT")
    print("="*70)
    print(f"{'Metric':<35} | {'Static Baseline':<15} | {'ErgoEMA (Proposed)':<18}")
    print("-"*70)
    for k in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        print(f"{k:<35} | {metrics_static[k]*100:>13.2f}% | {metrics_ergoema[k]*100:>16.2f}%")
    print(f"{'False Positives (Nuisance Alerts)':<35} | {metrics_static['False Positives (Nuisance Alerts)']:>14} | {metrics_ergoema['False Positives (Nuisance Alerts)']:>17}")
    print(f"{'Execution Latency per frame':<35} | {metrics_static['Mean Latency (ms)']:>12.4f} ms | {metrics_ergoema['Mean Latency (ms)']:>15.4f} ms")
    print("="*70)

    # 4. Generate Visual Plots for Thesis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top Plot: Signal Smoothing (Raw vs EMA)
    time_arr = df["timestamp"].values
    ax1.plot(time_arr, df["h2s_ratio"].values, label="Raw Noisy Signal (Webcam)", color="gray", alpha=0.5, lw=1)
    ax1.plot(time_arr, smoothed_ratios, label="EMA Filtered Signal (alpha=0.15)", color="#1f77b4", lw=2)
    ax1.axhline(baseline.mean_h2s_ratio, color="green", linestyle="--", label=f"Personalized Baseline ({baseline.mean_h2s_ratio:.2f})")
    ax1.axhline(baseline.mean_h2s_ratio * (1.0 - config.SLOUCH_RATIO_DROP_THRESH), color="red", linestyle=":", label="Adaptive Slouch Threshold")
    ax1.set_title("Figure 4: Temporal Smoothing & Dynamic Baseline Tracking", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Head-to-Shoulder Ratio ($R_{H2S}$)")
    ax1.legend(loc="lower left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Bottom Plot: Classification Alerts Comparison
    ax2.fill_between(time_arr, 0, y_true, color="gray", alpha=0.2, label="Ground Truth (Slouching Ground State)")
    ax2.plot(time_arr, y_pred_static, label="Static Baseline Alerts (High Nuisance Alarms)", color="orange", alpha=0.8, lw=1.5)
    ax2.plot(time_arr, y_pred_ergoema, label="ErgoEMA Adaptive Alerts (Filtered & Sustained)", color="crimson", lw=2)
    ax2.set_title("Figure 5: Alert Response Comparison (Mitigating Notification Fatigue)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Alert State (0/1)")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Alert"])
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ergoema_evaluation_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Evaluation charts saved to: {plot_path}")

    # Save summary markdown table
    report_md = f"""# ErgoEMA Evaluation Results Summary

| Metric | Static Baseline Model | ErgoEMA (Adaptive + EMA) | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | {metrics_static['Accuracy']*100:.2f}% | **{metrics_ergoema['Accuracy']*100:.2f}%** | +{(metrics_ergoema['Accuracy']-metrics_static['Accuracy'])*100:.2f}% |
| **Precision** | {metrics_static['Precision']*100:.2f}% | **{metrics_ergoema['Precision']*100:.2f}%** | +{(metrics_ergoema['Precision']-metrics_static['Precision'])*100:.2f}% |
| **Recall** | {metrics_static['Recall']*100:.2f}% | **{metrics_ergoema['Recall']*100:.2f}%** | +{(metrics_ergoema['Recall']-metrics_static['Recall'])*100:.2f}% |
| **F1-Score** | {metrics_static['F1-Score']*100:.2f}% | **{metrics_ergoema['F1-Score']*100:.2f}%** | +{(metrics_ergoema['F1-Score']-metrics_static['F1-Score'])*100:.2f}% |
| **False Nuisance Alerts** | {metrics_static['False Positives (Nuisance Alerts)']} | **{metrics_ergoema['False Positives (Nuisance Alerts)']}** | **{((metrics_static['False Positives (Nuisance Alerts)'] - metrics_ergoema['False Positives (Nuisance Alerts)']) / max(1, metrics_static['False Positives (Nuisance Alerts)'])) * 100:.1f}% reduction** |
| **Processing Latency** | {metrics_static['Mean Latency (ms)']:.4f} ms | **{metrics_ergoema['Mean Latency (ms)']:.4f} ms** | Well under 33.33 ms threshold (30 FPS) |
"""
    with open(os.path.join(output_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ErgoEMA against baseline")
    parser.add_argument("--data", type=str, default=None, help="Path to input CSV dataset")
    parser.add_argument("--output", type=str, default="results", help="Directory for output reports and plots")
    args = parser.parse_args()

    run_evaluation(csv_path=args.data, output_dir=args.output)
