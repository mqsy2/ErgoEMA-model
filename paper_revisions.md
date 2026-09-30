# ErgoEMA Academic Manuscript Revisions: Front-Facing Adaptive Posture Model

> **Title of Study**: ErgoEMA: An Adaptive Posture Detection Model Using Time-Series Skeletal Data  
> **Target Camera Geometry**: Front-facing / Desktop Monocular Webcam (1080p RGB, Eye-level, ~60 cm distance, ISO 9241-5 compliant)  
> **Core Innovation**: Scale-invariant front-facing geometric ratios ($R_{H2S}$), personalized online baseline calibration ($\boldsymbol{\mu}_{base}, \boldsymbol{\sigma}_{base}$), and recursive $O(1)$ Exponential Moving Average (EMA) filtering for low-FAR, real-time posture monitoring.

---

## 1. RATIONALE & METHODOLOGICAL FOUNDATION

In modern desktop computer workstations, webcams are mounted atop display monitors or built into laptop bezels, capturing users from an **anterior (front-facing) perspective**.

### Methodological Shift from Lateral CVA to Anterior Scale-Invariant Geometry
* **The Lateral Craniovertebral Angle (CVA) Limitation**: Traditional ergonomic studies measure forward head posture via the sagittal CVA (angle between the tragus of the ear and the C7 vertebra relative to horizontal). This strictly requires a $90^\circ$ side-profile camera. A single front-facing webcam cannot measure sagittal 2D planar angles without severe perspective projection errors.
* **The Front-Facing Solution**: In anterior viewing geometry, upper-body posture degradation (thoracic slouching and forward head slump) manifests through:
  1. **Vertical cranial compression relative to biacromial shoulder span ($R_{H2S}$)**.
  2. **MediaPipe 3D sagittal depth differentials ($\Delta Z_{FHP}$)**.
* **Streamlined Clinical Scope**: Focused on **Upright Posture**, **Vertical Slouch (Thoracic Kyphosis)**, and **Forward Head Posture (FHP)** to maximize clinical relevance and eliminate confounding multi-label noise.

---

## 2. REVISED CHAPTER SECTIONS (EXACT MANUSCRIPT REPLACEMENTS)

### Section 1.5: Revised Scope and Limitations
```markdown
1.5 SCOPE AND LIMITATIONS

This study focuses on developing and validating an adaptive software-based posture detection model (ErgoEMA) that leverages Exponential Moving Average (EMA) temporal filtering and online personalized baseline calibration to analyze time-series skeletal data from a standard front-facing desktop webcam. The operational scope is tailored to desk-based computer workstations where the user is captured from an anterior upper-body perspective at a standardized viewing distance (~50–70 cm) and eye-level height in compliance with ISO 9241-5 workstation ergonomic standards.

The classification scope is specifically focused on primary upper-body postural states:
1. Upright / Ergonomic Sitting (Calibrated Natural Baseline)
2. Vertical Slouch / Thoracic Kyphosis (Spinal compression causing cranial-to-shoulder ratio drop)
3. Forward Head Posture / FHP (Anterior cranial displacement along the sagittal Z-axis)

The model is evaluated using continuous real-world participant video recordings spanning diverse morphological somatotypes (Ectomorph, Mesomorph, Endomorph) and BMI classifications (Underweight, Normal, Overweight). The system operates exclusively on upper-body skeletal landmarks (nose, eyes, ears, and acromion shoulder joints), ensuring uninterrupted tracking even when lower extremities (hips, knees) are occluded by standard office desks.
```

---

### Section 2.3: Theoretical Background — Anterior Biomechanical Formulation
```markdown
2.3 THEORETICAL BACKGROUND

Computer Vision and Anterior Spatial Skeletal Mapping
The ErgoEMA framework extracts 33 spatial skeletal landmarks $(x_i, y_i, z_i)$ from monocular RGB video feeds using Google MediaPipe Pose (BlazePose). For an anterior (front-facing) workstation configuration, the key upper-body landmarks utilized include:
- Left and Right Acromion Processes (Shoulders: $S_L = (x_{11}, y_{11}, z_{11})$, $S_R = (x_{12}, y_{12}, z_{12})$)
- Cranial Reference Points (Nose: $N = (x_0, y_0, z_0)$; Left Ear: $E_L = (x_7, y_7, z_7)$; Right Ear: $E_R = (x_8, y_8, z_8)$)

Anatomical Reference Points:
- Mid-Shoulder Centroid ($M_S$):
  $$M_S = \left( \frac{x_{11} + x_{12}}{2}, \frac{y_{11} + y_{12}}{2}, \frac{z_{11} + z_{12}}{2} \right)$$
- 2D Biacromial Shoulder Span ($W_{shoulder}$):
  $$W_{shoulder} = \|\mathbf{S}_R - \mathbf{S}_L\|_2 = \sqrt{(x_{12} - x_{11})^2 + (y_{12} - y_{11})^2}$$

Biomechanical Formulations:

1. Normalized Head-to-Shoulder Vertical Compression Ratio ($R_{H2S}$):
To detect spinal slump and thoracic kyphosis in a scale- and distance-invariant manner, the vertical pixel distance between the cranial landmark (nose) and the mid-shoulder centroid is normalized by the user's instantaneous 2D biacromial shoulder span:
$$\Delta Y_{cranial} = y_{M_S} - y_N$$
$$R_{H2S} = \frac{\Delta Y_{cranial}}{W_{shoulder}} = \frac{y_{M_S} - y_N}{\sqrt{(x_{12} - x_{11})^2 + (y_{12} - y_{11})^2}}$$
Because both $\Delta Y_{cranial}$ and $W_{shoulder}$ scale proportionally with user distance from the camera, $R_{H2S}$ provides true distance invariance. When the thoracic spine flexes into a slouch, $\Delta Y_{cranial}$ drops while $W_{shoulder}$ remains stable, resulting in a marked percentage drop in $R_{H2S}$.

2. Sagittal Forward-Head Z-Displacement ($\Delta Z_{FHP}$):
MediaPipe provides camera-relative depth coordinates ($z_i$), where smaller/more negative values indicate proximity to the camera. Anterior forward head shift relative to the shoulder torso plane is calculated as:
$$\Delta Z_{FHP} = z_N - z_{M_S}$$
A significant negative decrease ($\Delta Z_{FHP} < \mu_{Z, base} - \delta_{Z}$) indicates anterior cranial translation toward the computer display.
```

---

### Section 3.2: Methodology, Data Cleaning, and Experimental Dataset
```markdown
3.2 EXPERIMENTAL DATASET AND DATA CLEANING PIPELINE

3.2.1 Data Collection Protocol
To evaluate front-facing posture monitoring under real-world conditions, continuous high-definition video recordings were captured at 30 frames per second (fps) at 1080p resolution. In accordance with ISO 9241-5 guidelines, the camera was positioned at eye level atop the primary workstation display at a standardized viewing distance of ~60 cm.

The full experimental dataset comprises **14,726 primary front-facing frames** across 9 de-identified participants representing diverse morphological somatotypes and body compositions:
- **Participant 1**: Mesomorph, Normal BMI (1,389 frames)
- **Participant 2**: Mesomorph, Normal BMI (1,286 frames)
- **Participant 3**: Endomorph, Normal BMI (4,958 frames)
- **Participant 4**: Endomorph, Normal BMI (3,415 frames)
- **Participant 5**: Endomorph, Normal BMI (1,146 frames)
- **Participant 6**: Mesomorph, Normal BMI (753 frames)
- **Participant 7**: Ectomorph, Normal BMI (317 frames)
- **Participant 8**: Mesomorph, Normal BMI (745 frames)
- **Participant 9**: Endomorph, Overweight BMI (717 frames)

An additional **30,827 oblique profile frames** ($45^\circ$ angle) and **3,560 dedicated lateral profile frames** ($90^\circ$ Left/Right views with 4-segment spine biometrics) were collected to serve as an empirical ablation benchmark for viewpoint sensitivity analysis (Total dataset corpus: **49,113 frames**). Controlled preliminary testing was also benchmarked on an initial 3,168-frame 4-participant pilot subset (an earlier feature extraction of Participants 1, 7, 8, and 9).

3.2.2 Data Cleaning and Anomaly Filtering Pipeline
The raw video stream is processed through an automated cleaning pipeline:
1. Spatial Landmark Extraction: Decodes sequential frames and extracts 33 3D coordinates via MediaPipe Pose.
2. Visibility Confidence Thresholding: Frames with confidence $< 0.5$ on cranial or shoulder joints are flagged.
3. Linear Temporal Interpolation: Brief tracking dropouts ($< 15$ frames / 0.5 s) are filled using linear interpolation:
   $$\hat{\mathbf{x}}_t = \mathbf{x}_{t_0} + (t - t_0) \frac{\mathbf{x}_{t_1} - \mathbf{x}_{t_0}}{t_1 - t_0}$$
4. Statistical Z-Score Outlier Filtering ($Z > 3.0$): Coordinate spikes caused by transient sensor noise or lighting shifts are detected and replaced:
   $$Z_t = \frac{|f_t - \bar{f}|}{\sigma_f}$$
5. Complete De-Identification & Confidentiality: All participant identities are pseudonymized into standard identifiers (`Participant1`–`Participant9`), satisfying institutional research ethics requirements.
```

---

### Section 3.3: System Architecture & Adaptive Algorithms
```markdown
3.3 SYSTEM ARCHITECTURE AND TIME-SERIES ALGORITHMS

Stage 1: Online Personalized Calibration
To eliminate morphological bias stemming from anatomical differences (e.g., individual neck length, shoulder width, natural spinal curvature), ErgoEMA performs a 3-second online calibration (~90 frames at 30 fps) while the user sits comfortably upright:
$$\mu_{base} = \frac{1}{N} \sum_{t=1}^N R_{H2S}(t), \quad \sigma_{base} = \sqrt{\frac{1}{N} \sum_{t=1}^N \big(R_{H2S}(t) - \mu_{base}\big)^2}$$

Stage 2: Recursive $O(1)$ Exponential Moving Average (EMA) Filtering
To eliminate high-frequency keypoint jitter, typing fidgets, and respiratory thoracic motion without memory-intensive sliding window buffers, incoming features are filtered recursively:
$$S_t = \alpha \cdot X_t + (1 - \alpha) \cdot S_{t-1}$$
where $\alpha \in (0, 1]$ is the smoothing factor (empirically optimized to $\alpha = 0.08$ on the full 14,726-frame dataset, and $\alpha = 0.12$ on the pilot dataset). This yields constant time $O(1)$ and space $O(1)$ computational complexity.

Stage 3: Adaptive State Machine Classification
The current posture state is determined by evaluating smoothed features against personalized baseline relative drops:
1. Slouch Condition:
   $$\text{Drop}_{H2S}(t) = \frac{\mu_{base} - S_t^{H2S}}{\mu_{base}} \ge \delta_{slouch} \quad (\text{where } \delta_{slouch} = 8.0\% \text{ to } 10.0\%)$$
2. Forward Head Condition:
   $$S_t^Z - \mu_{Z, base} \le -0.06$$
3. Upright Condition: Neither slouch nor FHP thresholds are violated.

To avoid false alarms from momentary posture adjustments, an alert is only triggered if a non-upright state is sustained for at least $W_{alert} = 1.0\text{ second}$ (30 consecutive frames at 30 fps). $W_{alert}$ is a fixed design setting rather than a tuned hyperparameter: the evaluation in Chapter 4 scores the classified posture state of each frame, which does not depend on it.

Stage 4: Multi-Perspective Spine Extension (Lateral $90^\circ$ Profiling)
For lateral $90^\circ$ camera configurations, the framework integrates continuous 4-segment spine landmark estimation:
- **Cervical (C7)** at the shoulder centroid.
- **Thoracic (T-spine)** with posterior offset reflecting dorsal kyphotic curve.
- **Lumbar (L-spine)** reflecting normal lordosis.
- **Sacral (S1)** anchored at the hip centroid.
Spine coordinates are projected with a directional posterior normal offset to conform strictly to the subject's dorsal back profile rather than anterior chest landmarks.

Stage 5: Explainable AI (XAI) Diagnostic Feedback
Unlike opaque black-box deep learning models, ErgoEMA generates actionable, human-interpretable ergonomic diagnostics on a real-time Head-Up Display (HUD):
- Posture Status: Clear state indication (UPRIGHT / SLOUCH / FHP).
- Metric Telemetry: Current ratio vs. calibrated baseline (e.g., "Ratio: 0.395 (Base: 0.440 | -10.2%)").
- Diagnostic Reasoning: Explicit root-cause feedback preventing notification fatigue.
```

---

## 3. CHAPTER 4: EMPIRICAL RESULTS & DISCUSSION (TABLES & FINDINGS)

### 4.1 Overall Model Benchmark Results

All models are evaluated with Leave-One-Subject-Out (LOSO) cross-validation: each participant is predicted by a model tuned (ErgoEMA) or trained (machine-learning baselines, static threshold) on the other participants only. Latency is the median per-frame classification time measured by `train.py` over 1,000 frames on the test machine; it excludes MediaPipe pose estimation, which is common to all models.

#### A. Scaled Cohort Evaluation (Full 14,726 Primary Front-Facing Frames, 9 Participants)
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`70.62%`** | **`68.41%`** | **`65.44%`** | **`66.90%`** | **`25.08%`** | **`0.0077 ms`** |
| Random Forest (100 Trees) | 44.00% | 39.57% | 44.56% | 41.92% | 56.47% | 6.4877 ms |
| Support Vector Machine (RBF) | 60.59% | 56.66% | 55.76% | 56.21% | 35.39% | 0.8140 ms |
| Logistic Regression | 49.54% | 43.65% | 38.70% | 41.03% | 41.47% | 0.1290 ms |
| Static Rigid Threshold | 47.99% | 42.07% | 38.94% | 40.45% | 44.50% | 0.0001 ms |

#### B. Pilot Benchmark Reference (3,168-Frame 4-Participant Subset)
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`93.06%`** | **`95.30%`** | **`90.07%`** | **`92.61%`** | **`4.15%`** | **`0.0076 ms`** |
| Random Forest (100 Trees) | 63.23% | 63.28% | 56.96% | 59.95% | 30.91% | 6.4424 ms |
| Support Vector Machine (RBF) | 31.85% | 22.11% | 16.26% | 18.74% | 53.57% | 0.3074 ms |
| Logistic Regression | 62.09% | 59.73% | 66.17% | 62.78% | 41.72% | 0.1276 ms |
| Static Rigid Threshold | 72.73% | 78.53% | 59.96% | 68.00% | 15.33% | 0.0001 ms |

The pilot subset is the earlier feature extraction of Participants 1, 7, 8, and 9 (`data/processed/pilot_dataset.csv`). Tuning and scoring ErgoEMA on the same 3,168 frames gives 94.79% accuracy, 94.52% F1-score and a 3.42% FAR; that in-sample figure is not a validation result.

### 4.2 Leave-One-Subject-Out (LOSO) Cross-Validation (Full 9-Participant Cohort)
Each participant is scored with the hyperparameters that maximize F1 on the other eight participants (all nine folds selected $\alpha = 0.08$, $\delta_{slouch} = 8\%$). The personalized baseline is calibrated on the held-out participant's own first upright frames, as in live use:

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

### 4.3 Viewpoint Sensitivity & Camera Ablation Study (49,113 Total Frames)

Every row uses the deployed hyperparameters ($\alpha = 0.08$, $\delta_{slouch} = 8\%$), which were tuned on the front-view frames, so the Front View row is an in-sample figure:

| Camera Perspective | Viewing Angle | Total Frames | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | False Alarm Rate (FAR) (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Front View ($0^\circ$ — Primary Scope)** | $0^\circ$ | **14,726** | **`70.62%`** | **`68.41%`** | **`65.44%`** | **`66.90%`** | **`25.08%`** |
| **Oblique Profile ($45^\circ$ View)** | $45^\circ$ | 30,827 | 58.11% | 79.46% | 22.28% | 34.80% | 5.80% |
| **Lateral Profile ($90^\circ$ View — Spine Biometrics)** | $90^\circ$ | 3,560 | **`96.84%`** | **`94.04%`** | **`100.00%`** | **`96.93%`** | **`6.28%`** |
| **All Combined Perspectives** | All | 49,113 | 57.04% | 64.59% | 26.11% | 37.18% | 13.59% |

### 4.4 Key Academic Discussion Points
1. **False Alarm Rate and Adaptive Normalization**:
   - In computer ergonomics, excessive false alarms cause users to disable monitoring software. On held-out participants, ErgoEMA's FAR is $4.15\%$ on the pilot benchmark and $25.08\%$ on the scaled cohort. In both cases this is lower than every baseline evaluated under the same protocol ($15.33\%$–$53.57\%$ on the pilot, $35.39\%$–$56.47\%$ on the scaled cohort), which supports personalized calibration combined with EMA temporal smoothing over uncalibrated classification. The scaled-cohort FAR is uneven across participants: specificity ranges from $31.36\%$ (Participant 6) to $97.96\%$ (Participant 1).
2. **Generalization to Unseen Participants**:
   - The machine-learning baselines classify raw feature values and do not transfer to unseen participants: on the scaled cohort, Random Forest and Logistic Regression fall below $50\%$ held-out accuracy. ErgoEMA, which scores each frame relative to the participant's own calibrated baseline, has the highest held-out accuracy and F1-score in both benchmarks, although its per-participant F1-score ranges from $14.43\%$ (Participant 4) to $98.87\%$ (Participant 1).
3. **Computational Efficiency for Ubiquitous Deployment**:
   - The EMA update and classification step takes a measured median of **0.0077 ms** per frame, compared with 6.49 ms for Random Forest and 0.81 ms for the SVM. This figure covers the classification stage only; end-to-end frame time is dominated by MediaPipe pose estimation, which all models share.
4. **Geometric Basis for Front-View Webcam Superiority & Dedicated Lateral Profiling**:
   - Direct front-view ($0^\circ$) monocular webcam geometry provides the ideal biacromial span ($W_{shoulder}$) for scale-invariant distance normalization under standard desktop constraints.
   - For dedicated clinical sagittal assessment, the $90^\circ$ lateral pipeline leverages invariant lateral biometrics (ear-to-shoulder displacement and dorsal kyphotic spine curvature) achieving **96.84% accuracy**.
5. **Strict Research Ethics & Participant De-Identification**:
   - All experimental data and evaluations strictly adhere to ethical standards of participant confidentiality through complete de-identification (`Participant1` through `Participant9`), ensuring compliance with human subject research protocols.

---

## 4. SUMMARY OF CORE ADVANTAGES

| Dimension | Previous Literature / Generic Baseline | ErgoEMA (This Study) |
| :--- | :--- | :--- |
| **Camera Geometry** | Sagittal profile (requires side camera mount) | **Standard front-facing monitor webcam (ISO 9241-5)** |
| **Scale Invariance** | Unnormalized pixel distances | **Normalized by 2D biacromial shoulder span ($R_{H2S}$)** |
| **Morphological Bias** | Fixed population thresholds fail on varied somatotypes | **Personalized online baseline calibration ($\boldsymbol{\mu}_{base}, \boldsymbol{\sigma}_{base}$)** |
| **Temporal Filtering** | Heavy sliding-window queues ($O(W)$ memory) | **Recursive Exponential Moving Average ($O(1)$ time/space)** |
| **False Alarm Rate (LOSO)** | Static threshold: $\text{FAR} = 15.33\%$ (pilot), $44.50\%$ (scaled cohort) | **$\text{FAR} = 4.15\%$ (pilot), $25.08\%$ (scaled cohort)** |
| **Explainability** | Opaque black-box binary classification | **Transparent XAI HUD with quantified percentage drops** |
