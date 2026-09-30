# ErgoEMA: Front-Facing Adaptive Posture Monitoring Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose%203D-orange.svg)](https://developers.google.com/mediapipe)
[![Classifier Latency](https://img.shields.io/badge/Classifier%20Latency-0.008%20ms%2Fframe%20(measured)-green.svg)](results/training_thesis_report.md)
[![False Alarm Rate](https://img.shields.io/badge/FAR%20(LOSO%2C%209%20participants)-25.08%25-blue.svg)](results/training_thesis_report.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**ErgoEMA** is a real-time, lightweight, front-facing ergonomic posture monitoring model powered by **Google MediaPipe (BlazePose 3D)**, **$O(1)$ Exponential Moving Average (EMA) signal filtering**, and **online personalized baseline calibration**.

Designed specifically for standard desktop and laptop computer workstations (~60 cm viewing distance at eye level, adhering to **ISO 9241-5** ergonomic standards), ErgoEMA eliminates morphological bias across diverse body types and drastically mitigates notification fatigue without requiring specialized wearable sensors or heavy neural network inference.

---

## 🌟 Key Architectural Innovations

```
[Webcam RGB Stream (30 FPS)]
             │
             ▼
 [MediaPipe 33-Landmark Pose]  ──> (Nose, Acromion Shoulders, 3D Z-Depth)
             │
             ▼
  [Biomechanical Extraction]   ──> (Head-to-Shoulder Ratio R_H2S, Anterior Depth ΔZ_FHP)
             │
             ▼
    [Recursive EMA Filter]     ──> S_t = α·X_t + (1-α)·S_{t-1} (Jitter & Fidget Noise Suppression)
             │
             ▼
  [Personalized Calibrator]    ──> 3-Second Upright Resting Baseline (μ_baseline, σ_baseline)
             │
             ▼
   [Adaptive State Machine]    ──> Upright vs. Slouch (R_H2S drop) vs. Forward Head (FHP ΔZ)
             │
             ▼
  [Explainable AI (XAI) HUD]   ──> Transparent Real-Time Ergonomic Diagnostics Overlay
```

1. **Front-Facing Scale-Invariant Geometry**:
   - **$R_{H2S}$ (Head-to-Shoulder Vertical Compression Ratio)**: Normalizes vertical distance between nose and midpoint shoulder by the 2D biacromial shoulder span ($\frac{\Delta Y_{nose, mid\_shoulder}}{\|\text{Shoulder}_R - \text{Shoulder}_L\|}$), providing mathematical scale and distance invariance.
   - **$\Delta Z_{FHP}$ (Sagittal Forward Head Displacement)**: Measures anterior cranial shift relative to the shoulder torso plane in 3D camera space.
2. **Online Personalized Baseline Calibration ($3\text{ s}$ / $90\text{ frames}$)**:
   - Captures each user's unique resting upright posture ($\boldsymbol{\mu}_{baseline}, \boldsymbol{\sigma}_{baseline}$) in real time.
   - Eliminates morphological classification bias across somatotypes (**Ectomorph**, **Mesomorph**, **Endomorph**) and **BMI categories** (Underweight, Normal, Overweight).
3. **$O(1)$ Multi-Channel Exponential Moving Average (EMA)**:
   - Smooths high-frequency skeletal jitter, typing fidgets, and natural breathing micro-movements with constant time and memory complexity ($O(1)$):
     $$\mathbf{S}_t = \alpha \cdot \mathbf{X}_t + (1 - \alpha) \cdot \mathbf{S}_{t-1}$$
4. **Focused Ergonomic Posture Scope**:
   - **Upright / Good Posture**: Neutral cranial and spinal alignment within calibrated tolerances.
   - **Slouch (Thoracic Kyphosis)**: Significant drop in head-to-shoulder vertical ratio ($R_{H2S}$).
   - **Forward Head Posture (FHP)**: Anterior cranial displacement ($\Delta Z_{FHP}$).
5. **Explainable AI (XAI) HUD Engine**:
   - Transparent on-screen diagnostic cards and plain-English feedback combat notification fatigue (e.g., *"Head compressed 18.5% below baseline | FHP shift: +0.082"*).
6. **Lateral Profile ($90^\circ$) Continuous Spine Biomechanics**:
   - In addition to frontal tracking, includes dedicated $90^\circ$ sagittal processing with 4-segment spine landmark estimation: **Cervical (C7)**, **Thoracic (T-spine dorsal curvature)**, **Lumbar (L-spine lordosis)**, and **Sacral (S1)**.
   - Applies an anatomical dorsal back offset (aligning spine keypoints along the true posterior curve rather than anterior chest) with clinical hyperkyphosis ($>55^\circ$) and forward head displacement ($>0.14$) thresholding.

---

## 📊 Empirical Thesis Benchmark Results (Chapter 4)

Evaluated across **14,726 real-world front-facing participant video frames** across diverse somatotypes and BMIs (with multi-perspective evaluation totaling **49,113 frames** across 9 de-identified study participants):

All accuracy figures below come from **Leave-One-Subject-Out (LOSO) cross-validation**: each participant is scored by a model tuned or trained on the other participants only. Latency is the median per-frame classification time measured by `train.py` on the machine that generated [`results/training_thesis_report.md`](results/training_thesis_report.md); it excludes MediaPipe pose estimation, which every model shares, and will differ on other hardware.

### Model Architecture Comparison (Primary Front View — 14,726 Frames, LOSO)
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`70.62%`** | **`68.41%`** | **`65.44%`** | **`66.90%`** | **`25.08%`** | **`0.0077 ms`** |
| Random Forest (100 Trees) | 44.00% | 39.57% | 44.56% | 41.92% | 56.47% | 6.4877 ms |
| Support Vector Machine (RBF) | 60.59% | 56.66% | 55.76% | 56.21% | 35.39% | 0.8140 ms |
| Logistic Regression | 49.54% | 43.65% | 38.70% | 41.03% | 41.47% | 0.1290 ms |
| Static Rigid Threshold | 47.99% | 42.07% | 38.94% | 40.45% | 44.50% | 0.0001 ms |

### Leave-One-Subject-Out (LOSO) Cross-Validation (Full 9-Participant Cohort)
Each participant is scored with the hyperparameters that maximize F1 on the other eight participants (all nine folds selected $\alpha = 0.08$, $\delta_{slouch} = 8\%$). The personalized baseline is calibrated on the held-out participant's own first upright frames, as in live use.

| Participant | Somatotype | BMI Category | Frames | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Participant 1** | Mesomorph | Normal | 1,389 | **98.92%** | **97.76%** | **100.00%** | **97.96%** | **98.87%** |
| **Participant 2** | Mesomorph | Normal | 1,286 | **71.62%** | **65.10%** | **96.36%** | **45.53%** | **77.70%** |
| **Participant 3** | Endomorph | Normal | 4,958 | **60.99%** | **62.18%** | **53.01%** | **68.73%** | **57.23%** |
| **Participant 4** | Endomorph | Normal | 3,415 | **63.54%** | **32.31%** | **9.29%** | **90.37%** | **14.43%** |
| **Participant 5** | Endomorph | Normal | 1,146 | **88.74%** | **95.98%** | **78.38%** | **97.29%** | **86.29%** |
| **Participant 6** | Mesomorph | Normal | 753 | **67.73%** | **62.15%** | **100.00%** | **31.36%** | **76.66%** |
| **Participant 7** | Ectomorph | Normal | 317 | **78.23%** | **70.00%** | **100.00%** | **55.77%** | **82.35%** |
| **Participant 8** | Mesomorph | Normal | 745 | **77.32%** | **67.93%** | **100.00%** | **56.33%** | **80.90%** |
| **Participant 9** | Endomorph | Overweight | 717 | **78.10%** | **69.51%** | **100.00%** | **56.27%** | **82.02%** |
| **Macro Average** | — | — | **14,726** | **`76.13%`** | **`69.21%`** | **`81.89%`** | **`66.62%`** | **`72.94%`** |
| **Pooled (All Held-Out Frames)** | — | — | **14,726** | **`70.62%`** | **`68.41%`** | **`65.44%`** | **`74.92%`** | **`66.90%`** |

### Camera Viewpoint Sensitivity & Perspective Ablation
Every row uses the deployed hyperparameters ($\alpha = 0.08$, $\delta_{slouch} = 8\%$), which were tuned on the front-view frames, so the Front View row is an in-sample figure:

| Camera Perspective | Angle | Frames | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Front View (Primary Scope)** | $0^\circ$ | **14,726** | **70.62%** | **68.41%** | **65.44%** | **66.90%** | **25.08%** |
| **Oblique Profile** | $45^\circ$ | 30,827 | 58.11% | 79.46% | 22.28% | 34.80% | 5.80% |
| **Lateral Profile (Spine Biometrics)** | $90^\circ$ | 3,560 | **96.84%** | **94.04%** | **100.00%** | **96.93%** | **6.28%** |
| **All Combined Perspectives** | All | 49,113 | 57.04% | 64.59% | 26.11% | 37.18% | 13.59% |

### Pilot Benchmark Reference (4-Participant Subset — 3,168 Frames)
*Initial proof-of-concept benchmark on [`data/processed/pilot_dataset.csv`](data/processed/pilot_dataset.csv): the earlier feature extraction of Participants 1, 7, 8, and 9. Reproduce with `python train.py --data data/processed/pilot_dataset.csv --output results/pilot --skip-ablation` (full report: [`results/pilot/training_thesis_report.md`](results/pilot/training_thesis_report.md)).*

| Protocol | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) | FAR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOSO (held-out participants)** | **`93.06%`** | **`95.30%`** | **`90.07%`** | **`95.85%`** | **`92.61%`** | **`4.15%`** |
| In-sample (tuned and scored on the same frames) | 94.79% | 96.21% | 92.88% | 96.58% | 94.52% | 3.42% |

The in-sample row is the figure quoted in earlier versions of this README. It is not a validation result and is kept only so the earlier number can be traced.

---

## 💻 System Requirements

### Hardware Requirements
* **Processor**: Modern Intel / AMD CPU with AVX2 instruction support (Intel Core i3 8th Gen+ or AMD Ryzen 3+).
* **Camera**: Standard integrated or USB RGB Webcam (720p or 1080p @ 30 FPS).
* **RAM**: Minimum 4 GB (8 GB recommended).
* **Workstation Placement**: Webcam placed directly in front of the user at eye level (~50–70 cm distance, $0^\circ$ viewing angle).

### Software Requirements
* **Operating System**: Windows 10/11, macOS 11+, or Ubuntu 20.04+.
* **Python**: `3.9`, `3.10`, or `3.11` (64-bit).

---

## 🛠️ Step-by-Step Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/mqsy2/ErgoEMA-model.git
cd ErgoEMA-model
```

### Step 2: Create and Activate a Virtual Environment
* **On Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Procedures to Run the System

### Procedure 1: Real-Time Live Webcam Posture Monitoring

Launch the live application with real-time MediaPipe skeletal tracking and XAI overlay:

```powershell
python main.py
```

#### Live Keyboard Controls:
| Key | Action | Description |
| :---: | :--- | :--- |
| **`C`** | **Calibrate Baseline** | Starts a **3-second upright posture calibration**. Sit comfortably upright facing the webcam at eye level. |
| **`S`** | **Save Baseline** | Saves your current calibrated profile to [`user_baseline.json`](user_baseline.json) for instant reuse. |
| **`L`** | **Load Baseline** | Loads an existing baseline profile from [`user_baseline.json`](user_baseline.json). |
| **`A`** | **Toggle Mode** | Toggles between **Adaptive ErgoEMA Mode** and **Static Rigid Baseline Mode** (for thesis demonstration). |
| **`Q`** / **`ESC`** | **Quit** | Gracefully closes video stream and cleans up resources. |

#### Optional CLI Arguments for `main.py`:
```powershell
# Run with a secondary external webcam (index 1)
python main.py --camera 1

# Launch with preloaded baseline and custom EMA smoothing factor
python main.py --baseline user_baseline.json --alpha 0.15
```

#### Hyperparameters Used by the Live App:
At startup `main.py` loads the EMA smoothing factor and slouch ratio drop threshold tuned by `train.py` from [`results/optimized_parameters.json`](results/optimized_parameters.json) and prints the values in use. `--alpha` overrides the tuned smoothing factor, `--params <path>` points to a different tuned-parameters file, and if no file is found the defaults in [`config.py`](config.py) apply. The sustained alert window (1.0 s) and forward-head threshold are not tuned and always come from `config.py`.

---

### Procedure 2: Extracting & Adding Video Recordings to the Dataset

Use **[`video_processor.py`](video_processor.py)** to process recorded participant videos into standardized, cleaned CSV datasets with automatic Z-score outlier filtering and interpolation:

#### Option A: Process a Single Video Recording
```powershell
python video_processor.py --video "path/to/participant_upright.mp4" --subject "Participant Name" --somatotype Mesomorph --bmi Normal --label upright --view_angle Front
```

#### Option B: Process an Entire Directory of Videos
```powershell
python video_processor.py --dir "path/to/videos_folder" --somatotype Endomorph --bmi Overweight --label auto --view_angle Front --output data/processed
```

#### Option C: Process Dedicated 90° Lateral Profile Videos
Use **[`lateral_video_processor.py`](lateral_video_processor.py)** to process lateral video recordings with 4-segment continuous spine landmark estimation:
```powershell
python lateral_video_processor.py
```

#### Video Processor Parameters:
| Argument | Options | Description |
| :--- | :--- | :--- |
| `--video` | File Path | Path to a single raw video file (`.mp4`, `.mov`, `.avi`, `.mkv`). |
| `--dir` | Directory Path | Path to a folder containing video recordings. |
| `--subject` | String | Participant identifier (e.g. `Participant1`, used for LOSO cross-validation grouping). |
| `--somatotype` | `Ectomorph`, `Mesomorph`, `Endomorph` | Body somatotype classification. |
| `--bmi` | `Underweight`, `Normal`, `Overweight` | BMI category. |
| `--label` | `upright`, `slouch`, `fhp`, `auto` | Ground-truth posture label. |
| `--view_angle` | `Front`, `Side` | Camera viewing perspective (`Front` for primary thesis dataset). |
| `--save_frames`| Flag | Exports individual extracted JPG frames to `data/frames/`. |
| `--output` | Directory Path | Destination folder for processed CSV files (default: `data/processed`). |

> [!NOTE]
> Processed frames are automatically merged and deduplicated into [`data/processed/combined_dataset.csv`](data/processed/combined_dataset.csv) (14,726 frames) and [`data/processed/front_view_dataset.csv`](data/processed/front_view_dataset.csv) (14,726 frames). Batch extraction across all subfolders is handled via [`batch_extract.py`](batch_extract.py).

---

### Procedure 3: Model Training, Parameter Optimization & LOSO Validation

Execute the complete thesis optimization and evaluation suite:

```powershell
python train.py
```

To reproduce the pilot benchmark on the 3,168-frame subset:

```powershell
python train.py --data data/processed/pilot_dataset.csv --output results/pilot --skip-ablation
```

#### What `train.py` Performs:
1. **Grid Search Optimization**: Sweeps EMA smoothing factors ($\alpha \in \{0.08, 0.12, 0.15, 0.20, 0.25\}$) and slouch ratio drop thresholds ($\delta_{slouch} \in \{8, 10, 12, 15, 18\}\%$). The values that maximize F1 over all participants are saved as the deployed hyperparameters.
2. **Leave-One-Subject-Out (LOSO) Cross-Validation**: For each participant, selects the hyperparameters that maximize F1 on the other participants and scores them on the held-out participant (9 folds on 14,726 front-facing frames).
3. **Machine Learning Benchmarking**: Compares ErgoEMA against Random Forest, SVM (RBF), Logistic Regression, and Static Rigid Threshold baselines under the same LOSO protocol, and measures each model's per-frame classification latency.
4. **Camera Viewpoint Ablation Study**: Compares Front View ($0^\circ$), Oblique Profile ($45^\circ$), and Lateral Profile ($90^\circ$) across 49,113 multi-perspective frames using the deployed hyperparameters (skip with `--skip-ablation`).
5. **Generates Academic Artifacts**:
   - 📄 **[`results/training_thesis_report.md`](results/training_thesis_report.md)**: Markdown tables formatted for Chapter 4.
   - 📈 **[`results/training_optimization_curves.png`](results/training_optimization_curves.png)**: High-resolution 3-panel publication figure.
   - ⚙️ **[`results/optimized_parameters.json`](results/optimized_parameters.json)**: Deployed hyperparameters (loaded by `main.py` at startup) plus LOSO and in-sample metrics.

---

### Procedure 4: Running Unit Tests

Run the automated test suite to verify geometric calculations, EMA filtering, and calibration logic:

```powershell
python -m unittest discover -s tests
```

---

## 📁 Repository Structure

```
ErgoEMA-model/
├── config.py                 # Hyperparameters, camera settings, and baseline thresholds
├── main.py                   # Real-time webcam monitoring app with XAI HUD overlay
├── train.py                  # Optimization, LOSO cross-validation, and benchmark suite
├── video_processor.py        # Video extraction, data cleaning, and CSV generation pipeline
├── lateral_video_processor.py # 90° lateral profile processing with 4-segment spine biometrics
├── batch_extract.py          # Batch automated video processing pipeline across participant folders
├── clean_datasets.py         # Automated data cleaning, de-identification, and CSV deduplication
├── dataset_collector.py      # Live webcam data collection and annotation tool
├── paper_revisions.md        # Academic thesis manuscript text revisions (Chapters 1–4)
├── requirements.txt          # Python dependencies
├── user_baseline.json        # Calibrated user baseline profile (generated via main.py)
│
├── src/                      # Core ErgoEMA Architecture Modules
│   ├── __init__.py
│   ├── pose_detector.py      # MediaPipe 33-landmark 3D skeletal extractor
│   ├── feature_extractor.py  # Normalized biacromial compression and Z-depth calculator
│   ├── ema_filter.py         # Multi-channel recursive O(1) EMA temporal filter
│   ├── calibrator.py         # 3-second online personalized baseline fitting module
│   ├── classifier.py         # Adaptive threshold state machine (Upright, Slouch, FHP)
│   └── xai_explainer.py      # Transparent XAI HUD renderer and diagnostic cards
│
├── data/
│   └── processed/            # Cleaned, de-identified frame datasets
│       ├── combined_dataset.csv      # Primary training dataset (Front view — 14,726 frames)
│       ├── pilot_dataset.csv         # Pilot subset: earlier extraction of Participants 1, 7, 8, 9 (3,168 frames)
│       ├── front_view_dataset.csv    # Front-facing camera recordings (14,726 frames)
│       ├── side_view_dataset.csv     # Oblique profile recordings for viewpoint ablation (30,827 frames)
│       ├── side_90deg_dataset.csv    # Dedicated 90° lateral profile spine dataset (3,560 frames)
│       └── all_angles_dataset.csv    # Full multi-perspective dataset (49,113 frames)
│
├── results/                  # Generated benchmark artifacts
│   ├── optimized_parameters.json     # Deployed hyperparameters (loaded by main.py) and LOSO metrics
│   ├── training_thesis_report.md     # Formatted Chapter 4 academic evaluation report
│   ├── evaluation_report.md          # Multi-model evaluation summary
│   ├── training_optimization_curves.png # 3-panel publication figures
│   └── pilot/                        # Same artifacts for the 3,168-frame pilot dataset
│
└── tests/
    └── test_ergoema.py       # Unit tests verifying geometry, EMA, and calibration logic
```

---

## 📄 Academic Thesis Documentation

* **Manuscript Revisions**: See [`paper_revisions.md`](paper_revisions.md) for textual revisions, mathematical derivations, and updated figure descriptions for Chapters 1, 2, and 3 of the thesis paper.
* **Empirical Report**: See [`results/training_thesis_report.md`](results/training_thesis_report.md) for performance tables and ablation studies for Chapter 4.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
