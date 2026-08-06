# ErgoEMA: Front-Facing Adaptive Posture Monitoring Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose%203D-orange.svg)](https://developers.google.com/mediapipe)
[![Latency](https://img.shields.io/badge/Inference%20Latency-1.2%20ms%20(%3E800%20FPS)-green.svg)](results/training_thesis_report.md)
[![False Alarm Rate](https://img.shields.io/badge/FAR-3.42%25-brightgreen.svg)](results/training_thesis_report.md)
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

---

## 📊 Empirical Thesis Benchmark Results (Chapter 4)

Evaluated across **3,168 real-world front-facing participant video frames** across diverse somatotypes:

### Model Comparison
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) 🏆 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`94.79%`** | **`96.21%`** | **`92.88%`** | **`94.52%`** | **`3.42%`** | **`1.2 ms`** |
| Random Forest (100 Trees) | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 14.8 ms |
| Support Vector Machine (RBF) | 59.44% | 68.30% | 29.98% | 41.67% | 13.01% | 28.5 ms |
| Logistic Regression | 88.10% | 90.63% | 84.06% | 87.22% | 8.12% | 0.8 ms |
| Static Rigid Threshold | 79.67% | 85.34% | 69.95% | 76.88% | 11.24% | 0.4 ms |

### Leave-One-Subject-Out (LOSO) Cross-Validation
| Participant | Somatotype | BMI Category | Frames | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bigid lagger** | Endomorph | Overweight | 717 | **99.02%** | **100.00%** | **98.04%** | **99.01%** |
| **Jasper Valdez** | Mesomorph | Normal | 745 | **97.99%** | **98.04%** | **97.77%** | **97.90%** |
| **Margot Antoinette Gabon** | Mesomorph | Normal | 1,389 | **96.11%** | **92.98%** | **99.24%** | **96.01%** |
| **Gab Villanueva** | Mesomorph | Normal | 317 | **71.92%** | **100.00%** | **44.72%** | **61.80%** |

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

#### Video Processor Parameters:
| Argument | Options | Description |
| :--- | :--- | :--- |
| `--video` | File Path | Path to a single raw video file (`.mp4`, `.mov`, `.avi`, `.mkv`). |
| `--dir` | Directory Path | Path to a folder containing video recordings. |
| `--subject` | String | Participant identifier / name (used for LOSO cross-validation grouping). |
| `--somatotype` | `Ectomorph`, `Mesomorph`, `Endomorph` | Body somatotype classification. |
| `--bmi` | `Underweight`, `Normal`, `Overweight` | BMI category. |
| `--label` | `upright`, `slouch`, `fhp`, `auto` | Ground-truth posture label. |
| `--view_angle` | `Front`, `Side` | Camera viewing perspective (`Front` for primary thesis dataset). |
| `--save_frames`| Flag | Exports individual extracted JPG frames to `data/frames/`. |
| `--output` | Directory Path | Destination folder for processed CSV files (default: `data/processed`). |

> [!NOTE]
> Processed frames are automatically merged and deduplicated into [`data/processed/combined_dataset.csv`](data/processed/combined_dataset.csv) and [`data/processed/front_view_dataset.csv`](data/processed/front_view_dataset.csv).

---

### Procedure 3: Model Training, Parameter Optimization & LOSO Validation

Execute the complete thesis optimization and evaluation suite:

```powershell
python train.py
```

#### What `train.py` Performs:
1. **Grid Search Optimization**: Sweeps EMA smoothing factors ($\alpha \in [0.05, 0.40]$) and slouch ratio drop thresholds ($\delta_{slouch} \in [5\%, 25\%]$) to discover optimal hyperparameters.
2. **Leave-One-Subject-Out (LOSO) Cross-Validation**: Validates cross-participant generalizability across all 4 participants.
3. **Machine Learning Benchmarking**: Compares ErgoEMA against Random Forest, SVM (RBF), Logistic Regression, and Static Rigid Threshold baselines.
4. **Camera Viewpoint Ablation Study**: Compares Front View ($0^\circ$) vs. Side Profile ($90^\circ$).
5. **Generates Academic Artifacts**:
   - 📄 **[`results/training_thesis_report.md`](results/training_thesis_report.md)**: Markdown tables formatted for Chapter 4.
   - 📈 **[`results/training_optimization_curves.png`](results/training_optimization_curves.png)**: High-resolution 3-panel publication figure.
   - ⚙️ **[`results/optimized_parameters.json`](results/optimized_parameters.json)**: Optimal hyperparameters loaded by runtime apps.

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
│   └── processed/            # Cleaned frame datasets
│       ├── combined_dataset.csv      # Primary training dataset (Front view)
│       ├── front_view_dataset.csv    # Front-facing camera recordings (3,168 frames)
│       ├── side_view_dataset.csv     # Profile recordings for viewpoint ablation (6,372 frames)
│       └── all_angles_dataset.csv    # Full multi-perspective dataset (9,540 frames)
│
├── results/                  # Generated benchmark artifacts
│   ├── optimized_parameters.json     # Saved optimal hyperparameters
│   ├── training_thesis_report.md     # Formatted Chapter 4 academic evaluation report
│   └── training_optimization_curves.png # 3-panel publication figures
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
