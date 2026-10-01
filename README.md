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

Evaluated across **14,757 real-world front-facing participant video frames** across diverse somatotypes and BMIs (with multi-perspective evaluation totaling **52,318 frames** across 11 de-identified study participants):

All accuracy figures below come from **Leave-One-Subject-Out (LOSO) cross-validation**: each participant is scored by a model tuned or trained on the other participants only. Latency is the median per-frame classification time measured by `train.py` on the machine that generated [`results/training_thesis_report.md`](results/training_thesis_report.md); it excludes MediaPipe pose estimation, which every model shares, and will differ on other hardware.

### Model Architecture Comparison (Primary Front View — 14,757 Frames, LOSO)
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`74.96%`** | **`68.44%`** | **`71.37%`** | **`69.87%`** | **`22.58%`** | **`0.0128 ms`** |
| Random Forest (100 Trees) | 51.18% | 42.14% | 53.56% | 47.17% | 50.45% | 6.3361 ms |
| Support Vector Machine (RBF) | 53.64% | 42.86% | 41.89% | 42.37% | 38.31% | 1.0897 ms |
| Logistic Regression | 65.30% | 63.34% | 34.91% | 45.01% | 13.86% | 0.1279 ms |
| Static Rigid Threshold | 53.11% | 42.96% | 46.47% | 44.64% | 42.33% | 0.0001 ms |

The forward-head depth threshold $\delta_{FHP}$ is tuned with $\alpha$ and $\delta_{slouch}$. With it fixed at its former value of 0.06, the same protocol gives 75.49% accuracy, 71.80% F1 and a 25.33% FAR: tuning it lowers the FAR but, on this cohort, also slightly lowers accuracy and F1.

### Leave-One-Subject-Out (LOSO) Cross-Validation (Full 11-Participant Cohort)
Each participant is scored with the hyperparameters that maximize F1 on the other 10 participants. Eight folds selected $\alpha = 0.08$, $\delta_{slouch} = 8\%$, $\delta_{FHP} = 0.20$; Participant 3's fold selected $\delta_{FHP} = 0.06$, Participant 5's fold $\alpha = 0.12$ and $\delta_{slouch} = 10\%$, and Participant 6's fold $\delta_{FHP} = 0.30$. The personalized baseline is calibrated on the held-out participant's own first upright frames, as in live use.

| Participant | Somatotype | BMI Category | Frames | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Participant 1** | Mesomorph | Normal | 1,389 | **99.42%** | **99.24%** | **99.54%** | **99.32%** | **99.39%** |
| **Participant 2** | Mesomorph | Normal | 1,286 | **73.02%** | **70.51%** | **81.52%** | **64.06%** | **75.61%** |
| **Participant 3** | Endomorph | Normal | 3,812 | **79.33%** | **62.18%** | **99.92%** | **68.73%** | **76.66%** |
| **Participant 4** | Endomorph | Normal | 3,415 | **63.54%** | **32.31%** | **9.29%** | **90.37%** | **14.43%** |
| **Participant 5** | Endomorph | Normal | 1,146 | **72.60%** | **97.22%** | **40.54%** | **99.04%** | **57.22%** |
| **Participant 6** | Mesomorph | Normal | 753 | **69.06%** | **67.58%** | **79.95%** | **56.78%** | **73.25%** |
| **Participant 7** | Ectomorph | Normal | 317 | **78.23%** | **70.00%** | **100.00%** | **55.77%** | **82.35%** |
| **Participant 8** | Mesomorph | Normal | 745 | **77.32%** | **67.93%** | **100.00%** | **56.33%** | **80.90%** |
| **Participant 9** | Endomorph | Overweight | 717 | **78.10%** | **69.51%** | **100.00%** | **56.27%** | **82.02%** |
| **Participant 10** | Ectomorph | Normal | 560 | **66.96%** | **51.57%** | **100.00%** | **49.04%** | **68.05%** |
| **Participant 11** | Ectomorph | Normal | 617 | **70.83%** | **100.00%** | **34.31%** | **100.00%** | **51.09%** |
| **Macro Average** | — | — | **14,757** | **`75.31%`** | **`71.64%`** | **`76.82%`** | **`72.34%`** | **`69.18%`** |
| **Pooled (All Held-Out Frames)** | — | — | **14,757** | **`74.96%`** | **`68.44%`** | **`71.37%`** | **`77.42%`** | **`69.87%`** |

Participant 3's *Full Slouch* front-view recording was filmed side-on and is excluded (1,146 frames; listed in `EXCLUDED_RECORDINGS` in `clean_datasets.py`), so their front-view data comes from their three other recordings. Participant 4's slouch recordings show no change in the head-to-shoulder ratio from their upright recordings, which limits every method for that participant. Of the two ectomorph participants added from a separate recording session (Participants 10 and 11, Normal BMI), Participant 10's two upright recordings differ (median head-to-shoulder ratio 0.354 and 0.294), so the second is partly flagged against a baseline calibrated on the first, and Participant 11's first slouch recording shows no drop in the ratio.

### Camera Viewpoint Sensitivity & Perspective Ablation
Every row runs the configuration the live app uses: the front-view hyperparameters ($\alpha = 0.08$, $\delta_{slouch} = 8\%$, $\delta_{FHP} = 0.20$) for frontal and oblique frames, and the side-on settings ($\alpha = 0.25$, alert at a CVA drop of $5^\circ$) for frames detected as lateral. The Front and Lateral rows are in-sample (tuned on those frames); the Oblique row is not.

| Camera Perspective | Angle | Frames | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Front View (Primary Scope)** | $0^\circ$ | 14,757 | 80.27% | 75.66% | 75.95% | 75.80% | 16.76% |
| **Oblique Profile** | $45^\circ$ | 32,752 | 53.06% | 82.50% | 8.90% | 16.06% | 1.92% |
| **Lateral Profile** | $90^\circ$ | 4,512 | 97.67% | 98.52% | 96.95% | 97.73% | 1.56% |
| **All Combined Perspectives** | All | 52,021 | 63.26% | 74.99% | 34.70% | 47.45% | 10.60% |

With the front-view settings, a $45^\circ$ camera catches only 8.90% of poor-posture frames: the tuned forward-head threshold rarely triggers at that angle. In the All Combined row each front-view participant is calibrated once on their first upright frames, and that calibration also serves their oblique frames; the side-on participants, who are numbered separately, are calibrated on their own side-on upright frames. The 297 public benchmark images in the lateral dataset carry no participant ID and are not scored.

### Per-View Benchmarks: Oblique ($45^\circ$) and Lateral ($90^\circ$), LOSO
The LOSO protocol above, re-run on each camera view's own recordings: ErgoEMA is tuned on the other participants' frames from that view, and each participant's baseline is calibrated in that view. In the lateral view the tuned settings are $\alpha$ and the CVA drop that raises the alert, and the ML baselines are trained on the side-view measurements (CVA, ear-to-shoulder offset, nose-to-shoulder angle). The lateral recordings come from a separate group of 10 participants with their own numbering, so lateral Participant 1 is not front Participant 1; lateral Participants 9 and 10 are front Participants 10 and 11. Per-participant tables are in Sections 5 and 6 of [`results/training_thesis_report.md`](results/training_thesis_report.md), with figures [`benchmark_oblique_45.png`](results/benchmark_oblique_45.png) and [`benchmark_lateral_90.png`](results/benchmark_lateral_90.png).

**Oblique Profile ($45^\circ$): 32,752 frames, 11 participants**

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`55.57%`** | **`65.95%`** | **`24.76%`** | **`36.01%`** | **`13.03%`** | **`0.0127 ms`** |
| Random Forest (100 Trees) | 54.91% | 55.87% | 50.75% | 53.19% | 40.85% | 6.5527 ms |
| Support Vector Machine (RBF) | 64.65% | 75.54% | 44.31% | 55.86% | 14.62% | 2.2382 ms |
| Logistic Regression | 49.88% | 50.46% | 39.15% | 44.09% | 39.19% | 0.1264 ms |
| Static Rigid Threshold | 51.49% | 88.70% | 4.46% | 8.50% | 0.58% | 0.0001 ms |

**Lateral Profile ($90^\circ$): 4,512 frames, 10 participants**

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`97.61%`** | **`98.39%`** | **`96.95%`** | **`97.66%`** | **`1.69%`** | **`0.0149 ms`** |
| ErgoEMA, Clinical CVA Criterion Only | 83.58% | 76.97% | 97.29% | 85.95% | 31.06% | 0.0129 ms |
| Random Forest (100 Trees) | 81.85% | 82.97% | 81.58% | 82.27% | 17.87% | 6.6058 ms |
| Support Vector Machine (RBF) | 86.10% | 89.29% | 83.04% | 86.05% | 10.63% | 0.1762 ms |
| Logistic Regression | 88.90% | 92.39% | 85.53% | 88.83% | 7.51% | 0.1259 ms |
| Static Rigid Threshold (CVA, unsmoothed) | 84.13% | 77.61% | 97.34% | 86.36% | 29.96% | 0.0002 ms |

- At $45^\circ$ the threshold rule flags only 24.76% of the poor-posture frames, because the head-to-shoulder ratio barely changes when most participants slouch at this angle. Every fold selected the most sensitive forward-head threshold (0.04), trading accuracy for F1 (57.89% accuracy and 35.27% F1 with the former fixed 0.06). Random Forest and SVM reach 53.19% and 55.86% F1, at 40.85% and 14.62% false alarm rates.
- At $90^\circ$ ErgoEMA alerts when the CVA falls more than a tuned number of degrees below the participant's calibrated upright CVA (the folds selected 4°–5°, with $\alpha = 0.25$). On held-out participants this reaches 97.61% accuracy, 97.66% F1 and a 1.69% FAR, against 83.58%, 85.95% and 31.06% when the clinical criterion (CVA below $50^\circ$) raises the alert, and 88.83% F1 for the best machine-learning baseline (Logistic Regression). Forward Head Posture is still named only when the CVA is below $50^\circ$; a drop that leaves it above is reported as a head-forward shift from the participant's upright. Most remaining errors are missed frames of Participants 1 and 4 (recall 60.38% and 57.89%).

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

#### Option D: Add a Recording Session (Front, Oblique and Lateral Folders)
Use **[`add_participant_videos.py`](add_participant_videos.py)** for a folder with `Front Facing`, `Oblique` and `Lateral` subfolders and files named like `Participant 10-Upright 2.mp4` or `Participant 11-Slouch Left.mp4`. Add the front/oblique participants to `PARTICIPANT_PROFILES` and the lateral ones to `LATERAL_PARTICIPANT_PROFILES` in [`lateral_video_processor.py`](lateral_video_processor.py) first: the lateral recordings number their participants separately, so a lateral file's number is its lateral ID. `--dry-run` lists the clips and their IDs without extracting anything, and `--lateral-map` can rename lateral IDs if needed. The new rows are appended without changing any existing row, and recordings already in the datasets are refused.
```powershell
python add_participant_videos.py --dir "C:\Users\...\Ectomorph Posture"
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
> Processed frames are automatically merged and deduplicated into [`data/processed/combined_dataset.csv`](data/processed/combined_dataset.csv) (14,757 frames) and [`data/processed/front_view_dataset.csv`](data/processed/front_view_dataset.csv) (14,757 frames). Recordings listed in `EXCLUDED_RECORDINGS` in [`clean_datasets.py`](clean_datasets.py) are skipped by `batch_extract.py` and dropped by `clean_datasets.py`. Batch extraction across all subfolders is handled via [`batch_extract.py`](batch_extract.py).

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
2. **Leave-One-Subject-Out (LOSO) Cross-Validation**: For each participant, selects the hyperparameters that maximize F1 on the other participants and scores them on the held-out participant (11 folds on 14,757 front-facing frames).
3. **Machine Learning Benchmarking**: Compares ErgoEMA against Random Forest, SVM (RBF), Logistic Regression, and Static Rigid Threshold baselines under the same LOSO protocol, and measures each model's per-frame classification latency.
4. **Per-View Benchmarks**: Repeats steps 1–3 on the oblique ($45^\circ$) and lateral ($90^\circ$) recordings. In the lateral view the grid covers the side-on smoothing factor and the CVA drop that raises the alert ($2^\circ$–$15^\circ$); the tuned values are saved for side-on frames.
5. **Camera Viewpoint Ablation Study**: Runs the live app's configuration on the Front View ($0^\circ$), Oblique Profile ($45^\circ$), and Lateral Profile ($90^\circ$) recordings (52,021 scored frames). Steps 4 and 5 are skipped with `--skip-ablation`. A full run takes about 8 minutes.
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
├── add_participant_videos.py  # Adds a session's front, oblique and lateral videos to every dataset
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
│       ├── combined_dataset.csv      # Primary training dataset (Front view — 14,757 frames)
│       ├── pilot_dataset.csv         # Pilot subset: earlier extraction of Participants 1, 7, 8, 9 (3,168 frames)
│       ├── front_view_dataset.csv    # Front-facing camera recordings (14,757 frames)
│       ├── side_view_dataset.csv     # Oblique profile recordings for viewpoint ablation (32,752 frames)
│       ├── side_90deg_dataset.csv    # Dedicated 90° lateral profile dataset (4,809 frames)
│       └── all_angles_dataset.csv    # Full multi-perspective dataset (52,318 frames)
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
