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

The full experimental dataset comprises **14,757 primary front-facing frames** across 11 de-identified participants representing diverse morphological somatotypes and body compositions:
- **Participant 1**: Mesomorph, Normal BMI (1,389 frames)
- **Participant 2**: Mesomorph, Normal BMI (1,286 frames)
- **Participant 3**: Endomorph, Normal BMI (3,812 frames)
- **Participant 4**: Endomorph, Normal BMI (3,415 frames)
- **Participant 5**: Endomorph, Normal BMI (1,146 frames)
- **Participant 6**: Mesomorph, Normal BMI (753 frames)
- **Participant 7**: Ectomorph, Normal BMI (317 frames)
- **Participant 8**: Mesomorph, Normal BMI (745 frames)
- **Participant 9**: Endomorph, Overweight BMI (717 frames)
- **Participant 10**: Ectomorph, Normal BMI (560 frames)
- **Participant 11**: Ectomorph, Normal BMI (617 frames)

Participants 10 and 11 were recorded in a separate session (two upright and up to two slouch recordings per camera view). The lateral recordings come from a separate group of 10 participants with their own numbering, so lateral Participant 1 is not front Participant 1; lateral Participants 9 and 10 are front Participants 10 and 11. One front-view recording was excluded: Participant 3's full-slouch video was filmed side-on, so its 1,146 frames are side-view rather than front-view measurements.

An additional **32,752 oblique profile frames** ($45^\circ$ angle) and **4,809 dedicated lateral profile frames** ($90^\circ$ Left/Right views: 4,512 participant video frames and 297 public benchmark images) were collected to serve as an empirical ablation benchmark for viewpoint sensitivity analysis (Total dataset corpus: **52,318 frames**). Controlled preliminary testing was also benchmarked on an initial 3,168-frame 4-participant pilot subset (an earlier feature extraction of Participants 1, 7, 8, and 9).

3.2.2 Data Cleaning and Anomaly Filtering Pipeline
The raw video stream is processed through an automated cleaning pipeline:
1. Spatial Landmark Extraction: Decodes sequential frames and extracts 33 3D coordinates via MediaPipe Pose.
2. Visibility Confidence Thresholding: Frames with confidence $< 0.5$ on cranial or shoulder joints are flagged.
3. Linear Temporal Interpolation: Brief tracking dropouts ($< 15$ frames / 0.5 s) are filled using linear interpolation:
   $$\hat{\mathbf{x}}_t = \mathbf{x}_{t_0} + (t - t_0) \frac{\mathbf{x}_{t_1} - \mathbf{x}_{t_0}}{t_1 - t_0}$$
4. Statistical Z-Score Outlier Filtering ($Z > 3.0$): Coordinate spikes caused by transient sensor noise or lighting shifts are detected and replaced:
   $$Z_t = \frac{|f_t - \bar{f}|}{\sigma_f}$$
5. Complete De-Identification & Confidentiality: All participant identities are pseudonymized into standard identifiers (`Participant1`–`Participant11`), satisfying institutional research ethics requirements.
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
where $\alpha \in (0, 1]$ is the smoothing factor (empirically optimized to $\alpha = 0.08$ on the full 14,757-frame dataset, and $\alpha = 0.15$ on the pilot dataset). This yields constant time $O(1)$ and space $O(1)$ computational complexity.

Stage 3: Adaptive State Machine Classification
The current posture state is determined by evaluating smoothed features against personalized baseline relative drops:
1. Slouch Condition:
   $$\text{Drop}_{H2S}(t) = \frac{\mu_{base} - S_t^{H2S}}{\mu_{base}} \ge \delta_{slouch} \quad (\text{where } \delta_{slouch} = 8.0\% \text{ to } 10.0\%)$$
2. Forward Head Condition:
   $$S_t^Z - \mu_{Z, base} < -\delta_{FHP} \quad (\text{tuned with } \alpha \text{ and } \delta_{slouch}\text{: } \delta_{FHP} = 0.20 \text{ on the full dataset, } 0.15 \text{ on the pilot})$$
3. Upright Condition: Neither slouch nor FHP thresholds are violated.

To avoid false alarms from momentary posture adjustments, an alert is only triggered if a non-upright state is sustained for at least $W_{alert} = 1.0\text{ second}$ (30 consecutive frames at 30 fps). $W_{alert}$ is a fixed design setting rather than a tuned hyperparameter: the evaluation in Chapter 4 scores the classified posture state of each frame, which does not depend on it.

Stage 4: Lateral $90^\circ$ Extension (Craniovertebral Angle)
For lateral $90^\circ$ camera configurations, forward head posture is assessed with the clinical craniovertebral angle (CVA): the angle between the horizontal and the line from the C7 vertebra to the tragus of the ear.
- **Tragus**: the camera-facing ear landmark $E$.
- **C7**: MediaPipe provides no C7 landmark. Its expected position is set from the camera-facing shoulder $S$ using the ear-to-shoulder distance $U = \|E - S\|$ as the body-size unit ($0.35\,U$ behind and $0.30\,U$ above the shoulder), and is then moved to the nearest point on the back-of-neck edge of the person segmentation mask when that edge lies within $0.30\,U$ of it. When long hair, a hood or a headrest pushes the edge further out, the expected position is used unchanged.
- **Angle**: with all coordinates converted to pixels so that the frame's aspect ratio does not distort the result,
  $$\text{CVA} = \operatorname{atan2}\big(y_{C7} - y_E,\; d\,(x_E - x_{C7})\big), \quad d = \pm 1 \text{ for a subject facing right or left.}$$
- **Alert**: after a side-on calibration, the alert is raised when the EMA-smoothed CVA falls more than $\Delta_{CVA}$ below the calibrated upright CVA ($\Delta_{CVA} = 5^\circ$, tuned with the side-on smoothing factor $\alpha_{lat} = 0.25$ on the lateral recordings). Several participants' instructed upright posture already measures close to $50^\circ$, so the absolute criterion alone would flag much of their upright sitting.
- **Clinical label**: the state is named forward head posture only when the CVA is also below $50^\circ$, the clinical criterion, which is not tuned; a drop that leaves the CVA above $50^\circ$ is reported as a head-forward shift from the user's upright. Without a side-on calibration, the $50^\circ$ criterion raises the alert on its own.
Clinical slouching is defined by thoracic kyphosis above $40^\circ$ (normal range $20^\circ$–$40^\circ$). Kyphosis cannot be measured from pose landmarks, and estimates from the back outline were unstable on the study's recordings (chair backs and loose clothing hide the lower thoracic spine), so slouch is not assessed in the lateral view.

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

#### A. Scaled Cohort Evaluation (Full 14,757 Primary Front-Facing Frames, 11 Participants)
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`74.96%`** | **`68.44%`** | **`71.37%`** | **`69.87%`** | **`22.58%`** | **`0.0128 ms`** |
| Random Forest (100 Trees) | 51.18% | 42.14% | 53.56% | 47.17% | 50.45% | 6.3361 ms |
| Support Vector Machine (RBF) | 53.64% | 42.86% | 41.89% | 42.37% | 38.31% | 1.0897 ms |
| Logistic Regression | 65.30% | 63.34% | 34.91% | 45.01% | 13.86% | 0.1279 ms |
| Static Rigid Threshold | 53.11% | 42.96% | 46.47% | 44.64% | 42.33% | 0.0001 ms |

With the forward-head threshold fixed at its former 0.06 instead of tuned, the same protocol gives ErgoEMA 75.49% accuracy, 71.80% F1-score and a 25.33% FAR.

#### B. Pilot Benchmark Reference (3,168-Frame 4-Participant Subset)
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ErgoEMA (Adaptive Time-Series)** | **`94.26%`** | **`97.20%`** | **`90.73%`** | **`93.85%`** | **`2.44%`** | **`0.0127 ms`** |
| Random Forest (100 Trees) | 63.23% | 63.28% | 56.96% | 59.95% | 30.91% | 6.4664 ms |
| Support Vector Machine (RBF) | 31.85% | 22.11% | 16.26% | 18.74% | 53.57% | 0.3033 ms |
| Logistic Regression | 62.09% | 59.73% | 66.17% | 62.78% | 41.72% | 0.1266 ms |
| Static Rigid Threshold | 72.73% | 78.53% | 59.96% | 68.00% | 15.33% | 0.0001 ms |

The pilot subset is the earlier feature extraction of Participants 1, 7, 8, and 9 (`data/processed/pilot_dataset.csv`). Tuning and scoring ErgoEMA on the same 3,168 frames gives 96.34% accuracy, 96.09% F1-score and a 0.61% FAR; that in-sample figure is not a validation result.

### 4.2 Leave-One-Subject-Out (LOSO) Cross-Validation (Full 11-Participant Cohort)
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

### 4.3 Viewpoint Sensitivity & Camera Ablation Study (52,318 Total Frames, 52,021 Scored)

Every row runs the configuration the live app uses: the front-view hyperparameters ($\alpha = 0.08$, $\delta_{slouch} = 8\%$, $\delta_{FHP} = 0.20$) for frontal and oblique frames, and the side-on settings ($\alpha = 0.25$, alert at a CVA drop of $5^\circ$) for frames detected as lateral. The Front and Lateral rows are in-sample (tuned on those frames); the Oblique row is not.

| Camera Perspective | Angle | Frames | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Front View ($0^\circ$ — Primary Scope)** | $0^\circ$ | 14,757 | 80.27% | 75.66% | 75.95% | 75.80% | 16.76% |
| **Oblique Profile ($45^\circ$ View)** | $45^\circ$ | 32,752 | 53.06% | 82.50% | 8.90% | 16.06% | 1.92% |
| **Lateral Profile ($90^\circ$ View — Change from Calibrated Upright CVA)** | $90^\circ$ | 4,512 | 97.67% | 98.52% | 96.95% | 97.73% | 1.56% |
| **All Combined Perspectives** | All | 52,021 | 63.26% | 74.99% | 34.70% | 47.45% | 10.60% |

With the front-view settings, a $45^\circ$ camera catches only 8.90% of poor-posture frames: the tuned forward-head threshold rarely triggers at that angle. In the All Combined row each front-view participant is calibrated once on their first upright frames, and that calibration also serves their oblique frames; the side-on participants, who are numbered separately, are calibrated on their own side-on upright frames. The 297 public benchmark images in the lateral dataset carry no participant ID and are not scored.

### 4.4 Key Academic Discussion Points
1. **False Alarm Rate and Adaptive Normalization**:
   - In computer ergonomics, excessive false alarms cause users to disable monitoring software. On held-out participants, ErgoEMA's FAR is $2.44\%$ on the pilot benchmark and $22.58\%$ on the scaled cohort. On the pilot this is lower than every baseline evaluated under the same protocol ($15.33\%$–$53.57\%$). On the scaled cohort Logistic Regression has a lower FAR ($13.86\%$) but detects only $34.91\%$ of poor-posture frames, against $71.37\%$ for ErgoEMA; the other baselines range from $38.31\%$ to $50.45\%$. Tuning the forward-head threshold, rather than fixing it at 0.06, lowered the scaled-cohort FAR from $25.33\%$, but also lowered the F1-score from $71.80\%$ to $69.87\%$ and the accuracy from $75.49\%$ to $74.96\%$. The FAR is uneven across participants: specificity ranges from $49.04\%$ (Participant 10) to $100.00\%$ (Participant 11).
2. **Generalization to Unseen Participants**:
   - The machine-learning baselines classify raw feature values and transfer poorly to unseen participants: on the scaled cohort they reach $51.18\%$–$65.30\%$ held-out accuracy, the best being Logistic Regression. ErgoEMA's threshold rule, which scores each frame relative to the participant's own calibrated baseline, has the highest held-out accuracy and F1-score in both benchmarks. Per-participant F1 for the threshold rule ranges from $14.43\%$ (Participant 4) to $99.39\%$ (Participant 1).
3. **Computational Efficiency for Ubiquitous Deployment**:
   - The EMA update and classification step takes a measured median of **0.0128 ms** per frame, compared with 6.34 ms for the Random Forest baseline and 1.09 ms for the SVM. This figure covers the classification stage only; end-to-end frame time is dominated by MediaPipe pose estimation, which all models share.
4. **Geometric Basis for Front-View Webcam Superiority & Dedicated Lateral Profiling**:
   - Direct front-view ($0^\circ$) monocular webcam geometry provides the ideal biacromial span ($W_{shoulder}$) for scale-invariant distance normalization under standard desktop constraints.
   - For sagittal assessment, the $90^\circ$ pipeline alerts on the drop of the estimated craniovertebral angle below the user's calibrated upright CVA and names forward head posture only below the clinical $50^\circ$ criterion. On held-out participants this reaches $97.61\%$ accuracy with a $1.69\%$ FAR, against $83.58\%$ and $31.06\%$ when the $50^\circ$ criterion raises the alert, because several participants' upright posture measures close to $50^\circ$. The CVA estimate carries an uncertainty of a few degrees because C7 is located from the body outline rather than palpated. Thoracic kyphosis, which defines clinical slouching (above $40^\circ$), cannot be measured from video and is not assessed.
5. **Strict Research Ethics & Participant De-Identification**:
   - All experimental data and evaluations strictly adhere to ethical standards of participant confidentiality through complete de-identification (`Participant1` through `Participant11`), ensuring compliance with human subject research protocols.

---

## 4. SUMMARY OF CORE ADVANTAGES

| Dimension | Previous Literature / Generic Baseline | ErgoEMA (This Study) |
| :--- | :--- | :--- |
| **Camera Geometry** | Sagittal profile (requires side camera mount) | **Standard front-facing monitor webcam (ISO 9241-5)** |
| **Scale Invariance** | Unnormalized pixel distances | **Normalized by 2D biacromial shoulder span ($R_{H2S}$)** |
| **Morphological Bias** | Fixed population thresholds fail on varied somatotypes | **Personalized online baseline calibration ($\boldsymbol{\mu}_{base}, \boldsymbol{\sigma}_{base}$)** |
| **Temporal Filtering** | Heavy sliding-window queues ($O(W)$ memory) | **Recursive Exponential Moving Average ($O(1)$ time/space)** |
| **False Alarm Rate (LOSO)** | Static threshold: $\text{FAR} = 15.33\%$ (pilot), $42.33\%$ (scaled cohort) | **$\text{FAR} = 2.44\%$ (pilot), $22.58\%$ (scaled cohort)** |
| **Explainability** | Opaque black-box binary classification | **Transparent XAI HUD with quantified percentage drops** |
