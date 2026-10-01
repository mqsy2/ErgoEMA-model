# ErgoEMA: Front-Facing Adaptive Posture Monitoring Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose%203D-orange.svg)](https://developers.google.com/mediapipe)
[![Classifier Latency](https://img.shields.io/badge/Classifier%20Latency-0.01%20ms%2Fframe%20(measured)-green.svg)](results/training_thesis_report.md)
[![False Alarm Rate](https://img.shields.io/badge/FAR%20(LOSO%2C%209%20participants)-15.67%25-blue.svg)](results/training_thesis_report.md)
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
6. **Lateral Profile ($90^\circ$) Craniovertebral Angle**:
   - In addition to frontal tracking, a side-on camera is assessed with the clinical measure of forward head posture: the **craniovertebral angle (CVA)** between the horizontal and the line from C7 to the ear. C7 is located on the back-of-neck edge of the body silhouette.
   - After a side-on calibration, the alert is raised when the CVA falls more than **$5^\circ$** below your calibrated upright CVA. It is named **Forward Head Posture** only when the CVA is also below the clinical **$50^\circ$** criterion; otherwise it reads *Head Forward of Your Upright*. The HUD always shows whether the current CVA meets the clinical criterion. Without a side-on calibration, the $50^\circ$ criterion raises the alert on its own.
   - The overlay draws what is measured: the back contour, C7, the CVA, and approximate thoracic, lumbar and sacral levels where the torso is in view.
   - The estimate needs a true side-on view with the ear and the back of the neck uncovered. Headphones push the ear landmark forward onto the cheek and make the angle read low; long hair or a hood hides the neck, in which case C7 is placed from body proportions alone.
   - Slouch is not assessed in this view. It is defined clinically by thoracic kyphosis above $40^\circ$ (normal: $20^\circ$–$40^\circ$), which cannot be measured from pose landmarks or the body outline.

---

## 📊 Empirical Thesis Benchmark Results (Chapter 4)

Evaluated across **13,580 real-world front-facing participant video frames** across diverse somatotypes and BMIs (with multi-perspective evaluation totaling **47,983 frames** across 9 de-identified study participants):

All accuracy figures below come from **Leave-One-Subject-Out (LOSO) cross-validation**: each participant is scored by a model tuned or trained on the other participants only. Latency is the median per-frame classification time measured by `train.py` on the machine that generated [`results/training_thesis_report.md`](results/training_thesis_report.md); it excludes MediaPipe pose estimation, which every model shares, and will differ on other hardware.

### Model Architecture Comparison (Primary Front View — 13,580 Frames, LOSO)
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`79.38%`** | **`76.00%`** | **`72.19%`** | **`74.05%`** | **`15.67%`** | **`0.0127 ms`** |
| Random Forest (100 Trees) | 46.09% | 37.87% | 50.42% | 43.25% | 56.89% | 6.3170 ms |
| Support Vector Machine (RBF) | 55.33% | 45.09% | 44.23% | 44.65% | 37.03% | 0.7080 ms |
| Logistic Regression | 66.61% | 66.45% | 36.44% | 47.06% | 12.65% | 0.1257 ms |
| Static Rigid Threshold | 52.12% | 42.12% | 46.81% | 44.34% | 44.23% | 0.0001 ms |

The forward-head depth threshold $\delta_{FHP}$ is tuned with $\alpha$ and $\delta_{slouch}$; with it fixed at its former value of 0.06, the same protocol gives 75.15% accuracy, 71.20% F1 and a 25.03% FAR.

### Leave-One-Subject-Out (LOSO) Cross-Validation (Full 9-Participant Cohort)
Each participant is scored with the hyperparameters that maximize F1 on the other eight participants. Seven folds selected $\alpha = 0.08$, $\delta_{slouch} = 8\%$, $\delta_{FHP} = 0.20$; Participant 5's fold selected $\alpha = 0.12$ and $\delta_{slouch} = 10\%$, and Participant 6's fold $\delta_{FHP} = 0.30$. The personalized baseline is calibrated on the held-out participant's own first upright frames, as in live use.

| Participant | Somatotype | BMI Category | Frames | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Participant 1** | Mesomorph | Normal | 1,389 | **99.42%** | **99.24%** | **99.54%** | **99.32%** | **99.39%** |
| **Participant 2** | Mesomorph | Normal | 1,286 | **73.02%** | **70.51%** | **81.52%** | **64.06%** | **75.61%** |
| **Participant 3** | Endomorph | Normal | 3,812 | **93.23%** | **83.43%** | **99.92%** | **89.79%** | **90.93%** |
| **Participant 4** | Endomorph | Normal | 3,415 | **63.54%** | **32.31%** | **9.29%** | **90.37%** | **14.43%** |
| **Participant 5** | Endomorph | Normal | 1,146 | **72.60%** | **97.22%** | **40.54%** | **99.04%** | **57.22%** |
| **Participant 6** | Mesomorph | Normal | 753 | **69.06%** | **67.58%** | **79.95%** | **56.78%** | **73.25%** |
| **Participant 7** | Ectomorph | Normal | 317 | **78.23%** | **70.00%** | **100.00%** | **55.77%** | **82.35%** |
| **Participant 8** | Mesomorph | Normal | 745 | **77.32%** | **67.93%** | **100.00%** | **56.33%** | **80.90%** |
| **Participant 9** | Endomorph | Overweight | 717 | **78.10%** | **69.51%** | **100.00%** | **56.27%** | **82.02%** |
| **Macro Average** | — | — | **13,580** | **`78.28%`** | **`73.08%`** | **`78.97%`** | **`74.19%`** | **`72.90%`** |
| **Pooled (All Held-Out Frames)** | — | — | **13,580** | **`79.38%`** | **`76.00%`** | **`72.19%`** | **`84.33%`** | **`74.05%`** |

Participant 3's *Full Slouch* front-view recording was filmed side-on and is excluded (1,146 frames; listed in `EXCLUDED_RECORDINGS` in `clean_datasets.py`), so their front-view data comes from their three other recordings. Participant 4's slouch recordings show no change in the head-to-shoulder ratio from their upright recordings, which limits every method for that participant.

### Camera Viewpoint Sensitivity & Perspective Ablation
Every row runs the configuration the live app uses: the front-view hyperparameters ($\alpha = 0.08$, $\delta_{slouch} = 8\%$, $\delta_{FHP} = 0.20$) for frontal and oblique frames, and the side-on settings ($\alpha = 0.25$, alert at a CVA drop of $5^\circ$) for frames detected as lateral. The Front and Lateral rows are in-sample (tuned on those frames); the Oblique row is not.

| Camera Perspective | Angle | Frames | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Front View (Primary Scope)** | $0^\circ$ | 13,580 | 81.25% | 76.91% | 77.16% | 77.03% | 15.93% |
| **Oblique Profile** | $45^\circ$ | 30,827 | 53.53% | 82.36% | 9.42% | 16.90% | 2.03% |
| **Lateral Profile** | $90^\circ$ | 3,279 | 96.92% | 98.10% | 95.63% | 96.85% | 1.81% |
| **All Combined Perspectives** | All | 47,686 | 61.98% | 70.83% | 33.77% | 45.74% | 12.56% |

With the front-view settings, a $45^\circ$ camera catches only 9.42% of poor-posture frames: the tuned forward-head threshold rarely triggers at that angle. The All Combined row calibrates each participant once on their first upright frames, so side-on frames without a side-on calibration fall back to the clinical $50^\circ$ criterion. The 297 public benchmark images in the lateral dataset carry no participant ID and are not scored.

### Per-View Benchmarks: Oblique ($45^\circ$) and Lateral ($90^\circ$), LOSO
The LOSO protocol above, re-run on each camera view's own recordings: ErgoEMA is tuned on the other participants' frames from that view, and each participant's baseline is calibrated in that view. In the lateral view the tuned settings are $\alpha$ and the CVA drop that raises the alert, and the ML baselines are trained on the side-view measurements (CVA, ear-to-shoulder offset, nose-to-shoulder angle). Per-participant tables are in Sections 5 and 6 of [`results/training_thesis_report.md`](results/training_thesis_report.md), with figures [`benchmark_oblique_45.png`](results/benchmark_oblique_45.png) and [`benchmark_lateral_90.png`](results/benchmark_lateral_90.png).

**Oblique Profile ($45^\circ$): 30,827 frames, 9 participants**

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`56.00%`** | **`66.92%`** | **`24.36%`** | **`35.72%`** | **`12.13%`** | **`0.0123 ms`** |
| Random Forest (100 Trees) | 55.83% | 56.79% | 50.15% | 53.26% | 38.44% | 6.3923 ms |
| Support Vector Machine (RBF) | 58.59% | 61.87% | 45.53% | 52.46% | 28.27% | 2.0630 ms |
| Logistic Regression | 50.19% | 50.45% | 40.65% | 45.02% | 40.21% | 0.1261 ms |
| Static Rigid Threshold | 51.91% | 88.70% | 4.77% | 9.05% | 0.61% | 0.0001 ms |

**Lateral Profile ($90^\circ$): 3,279 frames, 8 participants**

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`96.13%`** | **`96.64%`** | **`95.50%`** | **`96.07%`** | **`3.26%`** | **`0.0144 ms`** |
| ErgoEMA, Clinical CVA Criterion Only | 83.14% | 76.11% | 96.12% | 84.95% | 29.61% | 0.0143 ms |
| Random Forest (100 Trees) | 77.62% | 74.56% | 83.19% | 78.64% | 27.85% | 6.5153 ms |
| Support Vector Machine (RBF) | 86.52% | 86.04% | 86.88% | 86.46% | 13.84% | 0.1636 ms |
| Logistic Regression | 86.79% | 85.30% | 88.61% | 86.92% | 14.98% | 0.1250 ms |
| Static Rigid Threshold (CVA, unsmoothed) | 84.02% | 77.17% | 96.18% | 85.64% | 27.92% | 0.0001 ms |

- At $45^\circ$ the threshold rule flags only 24.36% of the poor-posture frames, because the head-to-shoulder ratio barely changes when most participants slouch at this angle. Every fold selected the most sensitive forward-head threshold (0.04), trading accuracy for F1 (58.11% accuracy and 34.80% F1 with the former fixed 0.06). Random Forest and SVM reach 53.26% and 52.46% F1, at 38.44% and 28.27% false alarm rates.
- At $90^\circ$ ErgoEMA alerts when the CVA falls more than a tuned number of degrees below the participant's calibrated upright CVA (the folds selected 3°–6°, with $\alpha = 0.25$). On held-out participants this reaches 96.13% accuracy, 96.07% F1 and a 3.26% FAR, against 83.14%, 84.95% and 29.61% when the clinical criterion (CVA below $50^\circ$) raises the alert, and 86.92% F1 for the best machine-learning baseline (Logistic Regression). Forward Head Posture is still named only when the CVA is below $50^\circ$; a drop that leaves it above is reported as a head-forward shift from the participant's upright. Most remaining errors are missed frames of Participants 1 and 4 (recall 56.60% and 57.89%).

### Pilot Benchmark Reference (4-Participant Subset — 3,168 Frames)
*Initial proof-of-concept benchmark on [`data/processed/pilot_dataset.csv`](data/processed/pilot_dataset.csv): the earlier feature extraction of Participants 1, 7, 8, and 9. Reproduce with `python train.py --data data/processed/pilot_dataset.csv --output results/pilot --skip-ablation` (full report: [`results/pilot/training_thesis_report.md`](results/pilot/training_thesis_report.md)).*

| Protocol | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) | FAR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOSO (held-out participants)** | **`94.26%`** | **`97.20%`** | **`90.73%`** | **`97.56%`** | **`93.85%`** | **`2.44%`** |
| In-sample (tuned and scored on the same frames) | 96.34% | 99.30% | 93.08% | 99.39% | 96.09% | 0.61% |

The in-sample row is not a validation result. Earlier versions of this README quoted 94.79%, an in-sample figure from before the forward-head threshold was tuned; it is kept here only so that number can be traced.

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

# Show the person's Heath-Carter somatotype, computed from their body measurements
# (first copy anthropometry_template.json to my_measurements.json and fill in the values)
python main.py --anthropometry my_measurements.json
```

#### Hyperparameters Used by the Live App:
At startup `main.py` loads the hyperparameters tuned by `train.py` from [`results/optimized_parameters.json`](results/optimized_parameters.json) and prints the values in use: the EMA smoothing factor, slouch ratio drop and forward-head depth threshold for frontal frames, and the smoothing factor and CVA drop for side-on frames. `--alpha` overrides the frontal smoothing factor, `--params <path>` points to a different tuned-parameters file, and if no file is found the defaults in [`config.py`](config.py) apply. The sustained alert window (1.0 s) and the clinical $50^\circ$ CVA criterion are not tuned and always come from `config.py`.

#### Somatotype in the Live App (Heath-Carter Method):
`--anthropometry <file>` computes the person's **Heath-Carter anthropometric somatotype** ([`src/somatotype.py`](src/somatotype.py)) and shows it under the mode banner as the endomorphy–mesomorphy–ectomorphy rating and its category, for example `3.0-4.9-2.5` / `BALANCED MESOMORPH`. Pressing **`S`** stores the category in the `somatotype` field of [`user_baseline.json`](user_baseline.json).

The Heath-Carter method is defined on measurements taken on the body, so the somatotype is calculated from values you enter; it is not detected from the video and does not change how posture is classified. Copy [`anthropometry_template.json`](anthropometry_template.json) and replace each `null` with the person's value. If the file cannot be read, the HUD shows `SOMATOTYPE: NOT LOADED` and the console says why. A filled-in file looks like this:

```json
{
    "height_cm": 175.0,
    "weight_kg": 70.0,
    "triceps_skinfold_mm": 10.0,
    "subscapular_skinfold_mm": 12.0,
    "supraspinale_skinfold_mm": 8.0,
    "medial_calf_skinfold_mm": 10.0,
    "humerus_breadth_cm": 7.0,
    "femur_breadth_cm": 9.5,
    "flexed_arm_girth_cm": 32.0,
    "calf_girth_cm": 37.0
}
```

| Component | Measurements needed | Equation |
| :--- | :--- | :--- |
| **Endomorphy** | Height; triceps, subscapular and supraspinale skinfolds | $-0.7182 + 0.1451X - 0.00068X^2 + 0.0000014X^3$, with $X = \text{skinfold sum} \times 170.18 / \text{height}$ |
| **Mesomorphy** | Height; humerus and femur breadths; flexed arm and calf girths; triceps and medial calf skinfolds | $0.858\,\text{humerus} + 0.601\,\text{femur} + 0.188\,\text{corrected arm girth} + 0.161\,\text{corrected calf girth} - 0.131\,\text{height} + 4.5$ |
| **Ectomorphy** | Height and weight | From $\text{HWR} = \text{height} / \sqrt[3]{\text{weight}}$: $0.732\,\text{HWR} - 28.58$ if $\text{HWR} \ge 40.75$; $0.463\,\text{HWR} - 17.63$ if $38.25 < \text{HWR} < 40.75$; otherwise $0.1$ |

Corrected girths subtract the skinfold (in cm) from the girth. A component with missing measurements is shown as `n/a`, and the category is named only when all three are rated. Values outside a plausible adult range are rejected to catch unit mix-ups.

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
Use **[`lateral_video_processor.py`](lateral_video_processor.py)** to process lateral video recordings into craniovertebral-angle features:
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
> Processed frames are automatically merged and deduplicated into [`data/processed/combined_dataset.csv`](data/processed/combined_dataset.csv) (13,580 frames) and [`data/processed/front_view_dataset.csv`](data/processed/front_view_dataset.csv) (13,580 frames). Recordings listed in `EXCLUDED_RECORDINGS` in [`clean_datasets.py`](clean_datasets.py) are skipped by `batch_extract.py` and dropped by `clean_datasets.py`. Batch extraction across all subfolders is handled via [`batch_extract.py`](batch_extract.py).

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
1. **Grid Search Optimization**: Sweeps EMA smoothing factors ($\alpha \in \{0.08, 0.12, 0.15, 0.20, 0.25\}$), slouch ratio drop thresholds ($\delta_{slouch} \in \{8, 10, 12, 15, 18\}\%$) and forward-head depth thresholds ($\delta_{FHP} \in \{0.04, 0.06, 0.10, 0.15, 0.20, 0.30\}$). The values that maximize F1 over all participants are saved as the deployed hyperparameters.
2. **Leave-One-Subject-Out (LOSO) Cross-Validation**: For each participant, selects the hyperparameters that maximize F1 on the other participants and scores them on the held-out participant (9 folds on 13,580 front-facing frames).
3. **Machine Learning Benchmarking**: Compares ErgoEMA against Random Forest, SVM (RBF), Logistic Regression, and Static Rigid Threshold baselines under the same LOSO protocol, and measures each model's per-frame classification latency.
4. **Per-View Benchmarks**: Repeats steps 1–3 on the oblique ($45^\circ$) and lateral ($90^\circ$) recordings. In the lateral view the grid covers the side-on smoothing factor and the CVA drop that raises the alert ($2^\circ$–$15^\circ$); the tuned values are saved for side-on frames.
5. **Camera Viewpoint Ablation Study**: Runs the live app's configuration on the Front View ($0^\circ$), Oblique Profile ($45^\circ$), and Lateral Profile ($90^\circ$) recordings (47,686 scored frames). Steps 4 and 5 are skipped with `--skip-ablation`. A full run takes about 8 minutes.
6. **Generates Academic Artifacts**:
   - 📄 **[`results/training_thesis_report.md`](results/training_thesis_report.md)**: Markdown tables formatted for Chapter 4.
   - 📈 **[`results/training_optimization_curves.png`](results/training_optimization_curves.png)**: High-resolution 3-panel publication figure.
   - 📈 **[`results/benchmark_oblique_45.png`](results/benchmark_oblique_45.png)**, **[`results/benchmark_lateral_90.png`](results/benchmark_lateral_90.png)**: LOSO and model comparison figures for the $45^\circ$ and $90^\circ$ views.
   - ⚙️ **[`results/optimized_parameters.json`](results/optimized_parameters.json)**: Deployed hyperparameters (loaded by `main.py` at startup) plus LOSO, in-sample and per-view benchmark metrics.

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
├── lateral_video_processor.py # 90° lateral profile processing (craniovertebral angle features)
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
│   ├── sagittal_geometry.py  # Side-profile craniovertebral angle from landmarks and body silhouette
│   ├── ema_filter.py         # Multi-channel recursive O(1) EMA temporal filter
│   ├── calibrator.py         # 3-second online personalized baseline fitting module
│   ├── classifier.py         # Adaptive threshold state machine (Upright, Slouch, FHP)
│   ├── somatotype.py         # Heath-Carter anthropometric somatotype from entered body measurements
│   └── xai_explainer.py      # Transparent XAI HUD renderer and diagnostic cards
│
├── data/
│   └── processed/            # Cleaned, de-identified frame datasets
│       ├── combined_dataset.csv      # Primary training dataset (Front view — 13,580 frames)
│       ├── pilot_dataset.csv         # Pilot subset: earlier extraction of Participants 1, 7, 8, 9 (3,168 frames)
│       ├── front_view_dataset.csv    # Front-facing camera recordings (13,580 frames)
│       ├── side_view_dataset.csv     # Oblique profile recordings for viewpoint ablation (30,827 frames)
│       ├── side_90deg_dataset.csv    # Dedicated 90° lateral profile dataset (3,576 frames)
│       └── all_angles_dataset.csv    # Full multi-perspective dataset (47,983 frames)
│
├── results/                  # Generated benchmark artifacts
│   ├── optimized_parameters.json     # Deployed hyperparameters (loaded by main.py) and LOSO metrics
│   ├── training_thesis_report.md     # Formatted Chapter 4 academic evaluation report
│   ├── evaluation_report.md          # Multi-model evaluation summary
│   ├── training_optimization_curves.png # 3-panel publication figures
│   ├── benchmark_oblique_45.png      # 45° LOSO and model comparison figure
│   ├── benchmark_lateral_90.png      # 90° LOSO and model comparison figure
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
