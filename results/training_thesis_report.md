# ErgoEMA Model Training & Empirical Validation Report

**Dataset**: 13,580 Real-World Front-Facing Video Frames
**Participants**: 9 Study Participants (Participant1, Participant2, Participant3, Participant4, Participant5, Participant6, Participant7, Participant8, Participant9)
**Methodology**: Online Adaptive Calibration + Exponential Moving Average (EMA)
**Validation Protocol**: Leave-One-Subject-Out (hyperparameters tuned on 8 participants, scored on the held-out participant)

---

## 1. Deployed Hyperparameters

Tuned on all 9 participants and saved to `optimized_parameters.json` for the live application:

| Hyperparameter | Symbol | Value | Description |
| :--- | :---: | :---: | :--- |
| **EMA Smoothing Factor** | $\alpha$ | **0.08** | Multi-channel noise suppression factor ($O(1)$ constant time) |
| **Slouch Ratio Drop Threshold** | $\delta_{slouch}$ | **8.0%** | Percentage compression in $R_{H2S}$ below calibrated baseline |
| **Forward Head Depth Threshold** | $\delta_{FHP}$ | **0.2** | Forward shift of the head depth $\Delta Z$ beyond the calibrated baseline |
| **Sustained Alert Window** | $W_{alert}$ | **1.0 s** | Temporal buffer before raising alert. Set in `config.py`, not tuned: frames are scored on the classified posture state, which does not depend on this window |
| **Side-On EMA Smoothing Factor** | $\alpha_{lat}$ | **0.25** | Used on frames detected as lateral; tuned on the lateral recordings (Section 6) |
| **CVA Drop Alert Threshold** | $\Delta_{CVA}$ | **5.0°** | Alert when the CVA is this far below the calibrated upright CVA; tuned on the lateral recordings |
| **Clinical FHP Criterion** | — | **50°** | CVA below this is named Forward Head Posture (expert criterion, not tuned) |

Scoring the front-view hyperparameters on the same 13,580 frames they were tuned on gives 81.25% accuracy, 77.03% F1-score and a 15.93% False Alarm Rate. This is an in-sample figure, not a validation result.

---

## 2. Leave-One-Subject-Out (LOSO) Cross-Validation Results

Each row is scored on a participant whose frames were excluded from hyperparameter tuning; the "Tuned" columns show the values selected from the other 8 participants. The personalized baseline is still calibrated on the held-out participant's own first upright frames, as in live use.

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned Slouch Drop (%)   | Tuned FHP Depth   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:------------------------|:------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Mesomorph    | Normal         |     1389 | 0.08          | 8.0                     | 0.2               |          99.42 |           99.24 |        99.54 |             99.32 |          99.39 |
| Participant2                 | Mesomorph    | Normal         |     1286 | 0.08          | 8.0                     | 0.2               |          73.02 |           70.51 |        81.52 |             64.06 |          75.61 |
| Participant3                 | Endomorph    | Normal         |     3812 | 0.08          | 8.0                     | 0.2               |          93.23 |           83.43 |        99.92 |             89.79 |          90.93 |
| Participant4                 | Endomorph    | Normal         |     3415 | 0.08          | 8.0                     | 0.2               |          63.54 |           32.31 |         9.29 |             90.37 |          14.43 |
| Participant5                 | Endomorph    | Normal         |     1146 | 0.12          | 10.0                    | 0.2               |          72.6  |           97.22 |        40.54 |             99.04 |          57.22 |
| Participant6                 | Mesomorph    | Normal         |      753 | 0.08          | 8.0                     | 0.3               |          69.06 |           67.58 |        79.95 |             56.78 |          73.25 |
| Participant7                 | Ectomorph    | Normal         |      317 | 0.08          | 8.0                     | 0.2               |          78.23 |           70    |       100    |             55.77 |          82.35 |
| Participant8                 | Mesomorph    | Normal         |      745 | 0.08          | 8.0                     | 0.2               |          77.32 |           67.93 |       100    |             56.33 |          80.9  |
| Participant9                 | Endomorph    | Overweight     |      717 | 0.08          | 8.0                     | 0.2               |          78.1  |           69.51 |       100    |             56.27 |          82.02 |
| Macro Average                | —            | —              |    13580 | —             | —                       | —                 |          78.28 |           73.08 |        78.97 |             74.19 |          72.9  |
| Pooled (All Held-Out Frames) | —            | —              |    13580 | —             | —                       | —                 |          79.38 |           76    |        72.19 |             84.33 |          74.05 |

---

## 3. Benchmark Architecture Comparison (Chapter 4 Results)

All models are scored under the same LOSO protocol: each participant is predicted by a model fitted on the other 8 participants. Latency is the median per-frame classification time over 1,000 frames, measured on the machine that generated this report; it excludes MediaPipe pose estimation, which every model shares.

| Model Architecture             |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series) |          79.38 |           76    |        72.19 |          74.05 |     15.67 |         0.0162 |
| Random Forest (100 Trees)      |          46.16 |           37.88 |        50.24 |          43.19 |     56.65 |         4.0785 |
| Support Vector Machine (RBF)   |          55.33 |           45.09 |        44.23 |          44.65 |     37.03 |         0.6419 |
| Logistic Regression            |          66.61 |           66.45 |        36.44 |          47.06 |     12.65 |         0.0715 |
| Static Rigid Threshold         |          52.12 |           42.12 |        46.81 |          44.34 |     44.23 |         0.0001 |

### Key Findings:
1. **Held-Out Performance**: On participants excluded from tuning, ErgoEMA reaches **79.38% accuracy** and **74.05% F1-score**. Per-participant F1 ranges from 14.43% (Participant4) to 99.39% (Participant1).
2. **False Alarm Rate**: The held-out False Alarm Rate (FAR) is **15.67%**.
3. **Baseline Comparison**: The highest held-out F1-score in the table is 74.05%, from ErgoEMA (Adaptive Time-Series).
4. **Measured Latency**: The ErgoEMA EMA update and classification step takes a median of **0.0162 ms** per frame.

---

## 4. Camera Viewpoint Sensitivity & Ablation Study (Chapter 4 Discussion)

The table below demonstrates the effect of camera placement geometry on biometric normalization. Every row runs the configuration the live application uses: the front-view hyperparameters from Section 1 for frontal and oblique frames, and the side-on settings for frames detected as lateral. The Front View row is an in-sample figure (those frames were used for tuning). The Lateral row uses the side-on settings tuned on those frames (Section 6), so it is also in-sample; the Oblique row uses the front-view settings, which were not tuned on its frames.

| Camera Perspective              |   Frames |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |
|:--------------------------------|---------:|---------------:|----------------:|-------------:|---------------:|----------:|
| Front View (0° - Primary Scope) |    13580 |          81.25 |           76.91 |        77.16 |          77.03 |     15.93 |
| Oblique Profile (45° View)      |    30827 |          53.53 |           82.36 |         9.42 |          16.9  |      2.03 |
| Lateral Profile (90° View)      |     3279 |          96.92 |           98.1  |        95.63 |          96.85 |      1.81 |
| All Combined Perspectives       |    47686 |          61.98 |           70.83 |        33.77 |          45.74 |     12.56 |

### Geometric Analysis:
* **Front View ($0^\circ$)**: The 2D biacromial shoulder span ($\|\text{Shoulder}_R - \text{Shoulder}_L\|$) provides an invariant denominator, yielding **81.25% Accuracy** and a **15.93% False Alarm Rate**.
* **Oblique View ($45^\circ$)**: Perspective foreshortening compresses the visible shoulder width, causing mild accuracy degradation compared to direct frontal placement.
* **Lateral View ($90^\circ$)**: The alert tracks the drop of the estimated craniovertebral angle (CVA) below the participant's calibrated upright CVA, with C7 located on the back-of-neck silhouette; Forward Head Posture is named only when the CVA is also below the clinical $50^\circ$ criterion. Thoracic kyphosis cannot be measured from video, so slouch is not assessed in this view.

---

## 5. Oblique Profile (45°) Benchmark

The protocol of Sections 2 and 3, run on the 30,827 oblique-profile frames (cameras $45^\circ$ to the left and right) of all 9 participants. ErgoEMA's hyperparameters are tuned on the other 8 participants' $45^\circ$ frames, and each participant's baseline is calibrated on their own $45^\circ$ upright frames, so this measures a system set up for a $45^\circ$ camera. Applying the front-view hyperparameters to these frames instead is the Oblique row of Section 4. The ML and static baselines use the same frontal features as in Section 3.

### 5.1 Leave-One-Subject-Out Cross-Validation

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned Slouch Drop (%)   | Tuned FHP Depth   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:------------------------|:------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Mesomorph    | Normal         |     2815 | 0.12          | 8.0                     | 0.04              |          51.19 |           57.14 |         0.87 |             99.37 |           1.72 |
| Participant2                 | Mesomorph    | Normal         |     2777 | 0.08          | 8.0                     | 0.04              |          44.26 |           14.03 |         2.23 |             86.31 |           3.85 |
| Participant3                 | Endomorph    | Normal         |     8808 | 0.2           | 8.0                     | 0.04              |          74.32 |           91.64 |        50.97 |             95.73 |          65.51 |
| Participant4                 | Endomorph    | Normal         |     8476 | 0.12          | 8.0                     | 0.04              |          42.46 |            0.54 |         0.04 |             90.67 |           0.08 |
| Participant5                 | Endomorph    | Normal         |     2725 | 0.12          | 8.0                     | 0.04              |          40.7  |           37.63 |        26.78 |             54.85 |          31.29 |
| Participant6                 | Mesomorph    | Normal         |     1580 | 0.12          | 8.0                     | 0.04              |          41.33 |           33.97 |        12.82 |             72.64 |          18.61 |
| Participant7                 | Ectomorph    | Normal         |      726 | 0.12          | 8.0                     | 0.04              |          83.61 |           85.49 |        78.78 |             87.96 |          82    |
| Participant8                 | Mesomorph    | Normal         |     1436 | 0.12          | 8.0                     | 0.04              |          54.32 |           66.24 |        14.71 |             92.73 |          24.07 |
| Participant9                 | Endomorph    | Overweight     |     1484 | 0.12          | 8.0                     | 0.04              |          87.6  |           79.89 |        99.86 |             75.79 |          88.77 |
| Macro Average                | —            | —              |    30827 | —             | —                       | —                 |          57.75 |           51.84 |        31.9  |             84.01 |          35.1  |
| Pooled (All Held-Out Frames) | —            | —              |    30827 | —             | —                       | —                 |          56    |           66.92 |        24.36 |             87.87 |          35.72 |

### 5.2 Benchmark Architecture Comparison

| Model Architecture             |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series) |          56    |           66.92 |        24.36 |          35.72 |     12.13 |         0.0163 |
| Random Forest (100 Trees)      |          55.91 |           56.89 |        50.11 |          53.29 |     38.25 |         4.1234 |
| Support Vector Machine (RBF)   |          58.59 |           61.87 |        45.53 |          52.46 |     28.27 |         1.8563 |
| Logistic Regression            |          50.18 |           50.45 |        40.65 |          45.02 |     40.21 |         0.0714 |
| Static Rigid Threshold         |          51.91 |           88.7  |         4.77 |           9.05 |      0.61 |         0.0001 |

On held-out participants, ErgoEMA reaches **56.00% accuracy**, **35.72% F1-score** and a **12.13% False Alarm Rate**; per-participant F1 ranges from 0.08% (Participant4) to 88.77% (Participant9). The highest held-out F1-score in the comparison is 53.29%, from Random Forest (100 Trees).

---

## 6. Lateral Profile (90°) Benchmark

The protocol of Sections 2 and 3, run on the 3,279 lateral-profile frames of the 8 participants recorded side-on (the 297 public benchmark images carry no participant ID and are excluded). In this view ErgoEMA raises an alert when the EMA-smoothed craniovertebral angle (CVA) falls more than a tuned number of degrees below the participant's calibrated upright CVA; the smoothing factor $\alpha$ and that drop are tuned on the other 7 participants. The state is named Forward Head Posture only when the CVA is also below the clinical $50^\circ$ criterion; a smaller CVA with a normal angle is reported as a head-forward shift from the participant's upright. The row *ErgoEMA, Clinical CVA Criterion Only* scores the clinical criterion (CVA below $50^\circ$) as the alert instead, with only $\alpha$ tuned. Frames labelled slouch or forward head both count as poor posture, as in the other views. The ML baselines are trained on the side-view measurements (CVA, ear-to-shoulder offset, nose-to-shoulder angle), and the static baseline applies the $50^\circ$ criterion to each frame without smoothing.

### 6.1 Leave-One-Subject-Out Cross-Validation

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned CVA Drop (°)   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:---------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Mesomorph    | Normal         |      214 | 0.25          | 6.0                  |          78.5  |          100    |        56.6  |            100    |          72.29 |
| Participant2                 | Mesomorph    | Normal         |      234 | 0.25          | 4.0                  |          91.45 |           85.93 |        99.15 |             83.76 |          92.06 |
| Participant3                 | Endomorph    | Normal         |      175 | 0.25          | 3.0                  |          89.71 |           75    |       100    |             85.12 |          85.71 |
| Participant4                 | Endomorph    | Normal         |      116 | 0.25          | 5.0                  |          79.31 |          100    |        57.89 |            100    |          73.33 |
| Participant5                 | Endomorph    | Normal         |      649 | 0.25          | 4.0                  |          99.23 |           98.51 |       100    |             98.43 |          99.25 |
| Participant6                 | Mesomorph    | Normal         |      634 | 0.25          | 4.0                  |          98.9  |           97.85 |       100    |             97.78 |          98.91 |
| Participant7                 | Ectomorph    | Normal         |      643 | 0.25          | 4.0                  |          99.22 |           98.55 |       100    |             98.35 |          99.27 |
| Participant8                 | Mesomorph    | Normal         |      614 | 0.25          | 6.0                  |          99.67 |          100    |        99.33 |            100    |          99.67 |
| Macro Average                | —            | —              |     3279 | —             | —                    |          92    |           94.48 |        89.12 |             95.43 |          90.06 |
| Pooled (All Held-Out Frames) | —            | —              |     3279 | —             | —                    |          96.13 |           96.64 |        95.5  |             96.74 |          96.07 |

### 6.2 Benchmark Architecture Comparison

| Model Architecture                       |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-----------------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series)           |          96.13 |           96.64 |        95.5  |          96.07 |      3.26 |         0.02   |
| ErgoEMA, Clinical CVA Criterion Only     |          83.14 |           76.11 |        96.12 |          84.95 |     29.61 |         0.0164 |
| Random Forest (100 Trees)                |          77.58 |           74.52 |        83.19 |          78.62 |     27.92 |         4.093  |
| Support Vector Machine (RBF)             |          86.52 |           86.04 |        86.88 |          86.46 |     13.84 |         0.1478 |
| Logistic Regression                      |          86.79 |           85.3  |        88.61 |          86.92 |     14.98 |         0.0716 |
| Static Rigid Threshold (CVA, unsmoothed) |          84.02 |           77.17 |        96.18 |          85.64 |     27.92 |         0.0001 |

On held-out participants, ErgoEMA reaches **96.13% accuracy**, **96.07% F1-score** and a **3.26% False Alarm Rate**; per-participant F1 ranges from 72.29% (Participant1) to 99.67% (Participant8). The highest held-out F1-score in the comparison is 96.07%, from ErgoEMA (Adaptive Time-Series).

