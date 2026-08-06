# ErgoEMA Thesis Revisions: Front-Facing Posture Recognition Adaptation

> **Title of Study**: ErgoEMA: An Adaptive Posture Detection Model Using Time-Series Skeletal Data  
> **Target Camera Geometry**: Front-facing / Desktop Monocular Webcam (Ansen 1080p, Eye-level, ~60 cm distance, ISO 9241 compliant)  
> **Key Revision Rationale**: Transitioning from lateral/side-profile Craniovertebral Angle (CVA) assumptions to **scale-invariant front-facing geometric ratios and 3D pose landmarks**.

---

## 1. RATIONALE & SUMMARY OF METHODOLOGICAL REVISION

In typical desktop computing environments, webcams are mounted directly atop computer monitors or integrated into laptop bezels, capturing users from an **anterior (front-facing) perspective**. 

### Why the Previous Draft Needed Revision
* **The Lateral CVA Limitation**: Traditional ergonomic literature assesses forward head posture using the *Craniovertebral Angle (CVA)*, which requires a **sagittal (side-profile)** view (connecting the tragus of the ear to the C7 spinous process relative to the horizontal). A single front-facing camera cannot project sagittal depth onto 2D planar angles without severe perspective ambiguity.
* **The Front-Facing Solution**: In a front-facing viewing geometry, posture degradation (slouching, thoracic kyphosis, forward head slump, and asymmetrical leaning) manifests through specific **anterior 2D planar displacements, scale-invariant vertical ratios, and MediaPipe 3D sagittal depth differentials ($Z$-coordinates)**.

---

## 2. REVISED CHAPTER SECTIONS (EXACT TEXT REPLACEMENTS)

### Section 1.5: Revised Scope and Limitations (Page 7)
```markdown
1.5 SCOPE AND LIMITATIONS

This study focuses on developing a software-based posture detection model (ErgoEMA) that uses an Exponential Moving Average (EMA) algorithm to analyze time-series skeletal data from a standard front-facing desktop webcam. The scope is specifically tailored to desk-based computer workstations where the user is captured from an anterior (front-facing) upper-body perspective at a standardized viewing distance (~60 cm) and eye-level height per ISO 9241 guidelines.

The model classifies upper-body sitting posture into categories including:
1. Upright / Ergonomic Sitting (Calibrated Baseline)
2. Vertical Slouch / Spinal Compression (Thoracic Kyphosis & Forward Head Slump)
3. Asymmetric Shoulder Slump / Lateral Lean (Left/Right Shoulder Elevation or Depression)
4. Lateral Head Flexion (Left/Right Head Tilt)
5. Forward Head Jutting (Sagittal Z-axis Protrusion)

The study utilizes live video feeds for online personalized calibration and structured front-facing benchmark sequences for baseline comparison. The system is designed to operate on upper-body skeletal landmarks (shoulders, neck, nose, ears, and eyes), ensuring high tracking reliability even when lower extremities (hips, knees) are occluded by standard office desks.
```

---

### Section 2.3: Revised Theoretical Background — Frontal Skeletal Trigonometry (Pages 19–21)
```markdown
2.3 THEORETICAL BACKGROUND

Computer Vision and Anterior Spatial Mapping
The ErgoEMA framework extracts 33 spatial skeletal landmarks $(x_i, y_i, z_i)$ from monocular RGB video feeds using MediaPipe Pose (BlazePose). For an anterior (front-facing) desk configuration, the key anatomical landmarks utilized include:
- Left and Right Acromion Processes (Shoulders: $S_L = (x_{11}, y_{11}, z_{11})$, $S_R = (x_{12}, y_{12}, z_{12})$)
- Cranial Reference Points (Nose: $N = (x_0, y_0, z_0)$; Left Ear: $E_L = (x_7, y_7, z_7)$; Right Ear: $E_R = (x_8, y_8, z_8)$)
- Ocular Reference Points (Left Eye: $O_L = (x_2, y_2, z_2)$; Right Eye: $O_R = (x_5, y_5, z_5)$)

Midpoint References:
- Mid-Shoulder Point ($M_S$):
  $$M_S = \left( \frac{x_{11} + x_{12}}{2}, \frac{y_{11} + y_{12}}{2}, \frac{z_{11} + z_{12}}{2} \right)$$
- Mid-Ear / Cranial Center Point ($M_E$):
  $$M_E = \left( \frac{x_7 + x_8}{2}, \frac{y_7 + y_8}{2}, \frac{z_7 + z_8}{2} \right)$$

Biomechanical Metrics for Front-Facing Posture Assessment:

1. Normalized Head-to-Shoulder Vertical Compression Ratio ($R_{H2S}$):
To detect spinal slump and forward head slouching in a scale-invariant manner (independent of distance from the webcam), the vertical distance between the cranial center (or nose) and the mid-shoulder is normalized by the participant's biacromial (inter-shoulder) Euclidean pixel distance ($W_{shoulder}$):
$$W_{shoulder} = \sqrt{(x_{12} - x_{11})^2 + (y_{12} - y_{11})^2}$$
$$\Delta Y_{cranial} = y_{M_S} - y_{N}$$
$$R_{H2S} = \frac{\Delta Y_{cranial}}{W_{shoulder}}$$
When a user slouches forward or hunches their back, the vertical projection $\Delta Y_{cranial}$ decreases significantly relative to biacromial width, causing $R_{H2S}$ to drop below baseline.

2. Shoulder Alignment / Horizontal Tilt Angle ($\theta_{shoulder}$):
Measures lateral spinal leaning and asymmetric shoulder elevation against the horizontal camera axis:
$$\theta_{shoulder} = \arctan\left( \frac{y_{12} - y_{11}}{x_{12} - x_{11}} \right) \times \frac{180^\circ}{\pi}$$
An upright posture exhibits $\theta_{shoulder} \approx 0^\circ$. Significant deviations indicate unilateral leaning or improper armrest support.

3. Head Roll / Lateral Flexion Angle ($\theta_{head}$):
Calculates lateral tilting of the cervical spine:
$$\theta_{head} = \arctan\left( \frac{y_8 - y_7}{x_8 - x_7} \right) \times \frac{180^\circ}{\pi}$$

4. 3D Sagittal Forward-Head Displacement ($\Delta Z_{FHP}$):
Leveraging MediaPipe's depth estimation ($z$-coordinate relative to the mid-hip/shoulder plane), sagittal forward head drift is tracked as:
$$\Delta Z_{FHP} = z_{N} - z_{M_S}$$
A negative shift in $\Delta Z_{FHP}$ indicates that the user's head has protruded anteriorly toward the screen relative to their torso.
```

---

### Section 3.2.1: Revised Data Collection (Page 25)
```markdown
3.2.1 Data Collection

A critical challenge identified in existing posture research is the absence of an appropriate, publicly accessible benchmark dataset for front-facing (anterior) desktop workstation posture monitoring. Existing public datasets (such as the Posture Keypoints Detection Dataset or MPII/COCO pose sets) predominantly focus either on lateral (side-profile) sagittal views for craniovertebral angles, standing full-body sports activities, or wearable physical sensor arrays. These do not represent the upper-body field-of-view, viewing angle, or occlusions characteristic of modern computer monitor webcams.

To address this gap, this study collects an empirical primary dataset consisting of continuous, raw high-definition video recordings captured directly in a standardized desk environment using the Ansen 1080p Webcam operating at 30 frames per second (fps). In compliance with ISO 9241 ergonomic standards, the camera is positioned at eye level atop the primary workstation display at a standardized viewing distance of 60 cm from the seated participant.

To ensure the model adapts across diverse human anatomies without morphological bias, raw video data is collected across participants categorized by:
- Somatotypes: Ectomorph, Mesomorph, Endomorph
- BMI Categories: Underweight (<18.5), Normal (18.5–24.9), Overweight/Obese (≥25.0)

For each participant, standardized front-facing postural sequences are recorded across distinct ergonomic states:
1. Upright Ergonomic Sitting (Calibrated Natural Baseline)
2. Slouched Sitting / Thoracic Kyphosis (Spinal slump & cranial downward drift)
3. Asymmetrical Shoulder Drop / Lateral Lean (Left and Right unleveling)
4. Forward Head Jutting (Anterior cranial protrusion toward the monitor)
5. Natural Micro-Movements (Keyboard typing, subtle head turning, harmless fidgeting)
```

---

### Section 3.2.2 & 3.2.3: Revised Data Cleaning and Preprocessing (Pages 26–27)
```markdown
3.2.2 Video-to-Frame Extraction and Data Cleaning

The captured raw video recordings are processed through a systematic data extraction and cleaning pipeline:

1. Frame-by-Frame Skeletal Extraction:
Raw video files (.mp4 / .avi) are decoded into sequential RGB image frames at 30 fps. MediaPipe BlazePose extracts 33 spatial keypoint coordinates $(x_i, y_i, z_i)$ along with landmark visibility confidence scores for each frame.

2. Visibility Filtering & Missing Value Handling:
Frames where essential upper-body keypoints (nose, shoulders) have a visibility confidence score below 0.5 (e.g., severe occlusion or subject leaving the workstation) are flagged. For transient keypoint dropouts (less than 15 consecutive frames / 0.5 seconds), linear interpolation is applied between adjacent valid frames to maintain temporal continuity:
$$\hat{x}_t = x_{t_0} + (t - t_0) \frac{x_{t_1} - x_{t_0}}{t_1 - t_0}$$

3. Z-Score Outlier Detection and Jitter Elimination:
Sensor noise, sudden lighting shifts, or brief tracking glitches can produce unphysical coordinate spikes. Standard statistical thresholds ($Z$-score $> 3.0$) are applied across the sliding feature stream to detect and remove non-physiological anomalies:
$$Z_t = \frac{|f_t - \bar{f}|}{\sigma_f}$$
Values exceeding $Z > 3.0$ are replaced via local linear interpolation, preventing spurious data spikes from corrupting the baseline or triggering false alerts.

3.2.3 Data Preprocessing and Partitioning
The cleaned frame-level keypoints are transformed into scale-invariant geometric features ($R_{H2S}, \theta_{shoulder}, \theta_{head}, \Delta Z_{FHP}$) and structured into time-series sequences. The data is partitioned into:
- 70% Training Set: For baseline distribution modeling and feature boundary characterization.
- 15% Validation Set: For hyperparameter optimization (tuning the EMA smoothing factor $\alpha$ and deviation multipliers via grid search).
- 15% Test Set: For unbiased comparative evaluation against the static threshold baseline model.
```

---

### Revised Figure 1 & Figure 2 Descriptions (Pages 25–26)

* **Figure 1 (Primary Dataset Setup)**:  
  * *Description*: "Standardized front-facing upper-body webcam perspective showing detected MediaPipe skeletal landmarks: Eyes, Nose, Ears, Acromion (Shoulders), and calculated Mid-Shoulder centroid $M_S$ and biacromial span $W_{shoulder}$."
* **Figure 2 (Front-Facing Postural States)**:  
  * *Description*: "Comparison of front-facing postural states: (a) Upright Ergonomic Sitting showing normal Head-to-Shoulder ratio $R_{H2S}$ and neutral shoulder tilt $\theta_{shoulder} \approx 0^\circ$; (b) Slouched Posture showing marked vertical compression of $R_{H2S}$; (c) Asymmetrical Lateral Lean showing angular deviation of $\theta_{shoulder}$."

---

### Section 3.2.4: Revised Feature Engineering & Temporal Mapping (Page 27–28)
```markdown
3.2.4 Feature Engineering

Spatial Mapping (Frontal Vector Transformation)
The raw landmark stream $(x_i, y_i, z_i)$ for each frame $t$ is mapped into a 4-dimensional geometric feature vector:
$$\mathbf{F}_t = \big[ R_{H2S}(t),\ \theta_{shoulder}(t),\ \theta_{head}(t),\ \Delta Z_{FHP}(t) \big]^T$$

Online Personalized Calibration Module (Stage 2)
To eliminate morphological bias caused by natural anatomical differences (e.g., individual neck length, shoulder breadth, or baseline spinal curvature), an initial 3-second calibration phase (~90 frames at 30 fps) is executed while the user sits in their natural upright posture:
$$\boldsymbol{\mu}_{baseline} = \frac{1}{N_{calib}} \sum_{t=1}^{N_{calib}} \mathbf{F}_t, \quad \boldsymbol{\sigma}_{baseline} = \sqrt{\frac{1}{N_{calib}} \sum_{t=1}^{N_{calib}} (\mathbf{F}_t - \boldsymbol{\mu}_{baseline})^2}$$

Temporal Smoothing via Exponential Moving Average (EMA) (Stage 3)
During active monitoring, the raw incoming feature vector $\mathbf{F}_t$ is smoothed using the multi-channel recursive EMA formulation ($O(1)$ complexity):
$$\mathbf{EMA}_t = \alpha \cdot \mathbf{F}_t + (1 - \alpha) \cdot \mathbf{EMA}_{t-1}$$
where $\alpha \in (0, 1]$ is the smoothing factor (optimized via grid search, $\alpha \approx 0.15$). This dampens camera keypoint jitter and transient micro-movements (e.g., reaching for a cup, laughing, quick typing shifts) while preserving sustained postural deviations.

Adaptive Posture Classification & Deviation Vector:
The deviation from the user's personalized baseline is computed as:
$$\boldsymbol{\Delta}_t = \mathbf{EMA}_t - \boldsymbol{\mu}_{baseline}$$
A posture alert is triggered if any component of $\boldsymbol{\Delta}_t$ exceeds its adaptive threshold:
$$\text{Alert Condition} = \big( \Delta R_{H2S}(t) < -k_1 \cdot \text{Thresh}_{R} \big) \lor \big( |\Delta \theta_{shoulder}(t)| > k_2 \cdot \text{Thresh}_{S} \big) \lor \dots$$
```

---

### Section 3.3.2 Stage 4: Revised Explainable AI (XAI) Output Logic (Page 31–32)
```markdown
Stage 4: Explainable AI Integration (XAI Output Logic)

Unlike opaque black-box machine learning models, ErgoEMA generates human-interpretable ergonomic explanations by reporting the exact feature deviations relative to the user's calibrated baseline:

Example XAI Alert Outputs:
1. "Slouch Detected: Head-to-shoulder vertical compression is 24% below your upright baseline (Current: 0.38, Baseline: 0.50)."
2. "Asymmetric Lean: Left shoulder dropped 8.2° below your calibrated horizontal plane."
3. "Forward Head Posture: Head jutted anteriorly 12 cm closer to screen relative to torso."

This transparent feedback allows users to understand the exact root cause of an alert, preventing notification fatigue and directly promoting ergonomic self-correction.
```

---

## 3. SUMMARY OF COMPARATIVE ADVANTAGES

| Feature / Metric | Previous Generic Draft (Side CVA) | Revised ErgoEMA (Front-Facing Setup) |
| :--- | :--- | :--- |
| **Camera View** | Side profile (sagittal) | **Front-facing monitor webcam (anterior)** |
| **Primary Metric** | Craniovertebral Angle ($\theta_{CVA} \approx 55^\circ$) | **Normalized Head-to-Shoulder Ratio ($R_{H2S}$)** |
| **Lateral Metrics** | Not captured from side | **Shoulder Tilt ($\theta_{shoulder}$), Head Roll ($\theta_{head}$)** |
| **Depth Tracking** | Direct 2D side angle | **MediaPipe 3D $Z$-offset ($\Delta Z_{FHP}$)** |
| **Scale Invariance** | Sensitive to distance | **Normalized by biacromial shoulder span $W_{shoulder}$** |
| **Time Complexity** | $O(1)$ EMA | **$O(1)$ Multi-channel EMA** |
| **Explainability** | Generic angle threshold | **Quantified multi-feature relative deviation** |
