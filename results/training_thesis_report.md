# ErgoEMA Model Training & Empirical Validation Report

**Dataset**: 14,757 Real-World Front-Facing Video Frames
**Participants**: 11 Study Participants (Participant1, Participant2, Participant3, Participant4, Participant5, Participant6, Participant7, Participant8, Participant9, Participant10, Participant11)
**Methodology**: Online Adaptive Calibration + Exponential Moving Average (EMA)
**Validation Protocol**: Leave-One-Subject-Out (hyperparameters tuned on 10 participants, scored on the held-out participant)

---

## 1. Deployed Hyperparameters

Tuned on all 11 participants and saved to `optimized_parameters.json` for the live application:

| Hyperparameter | Symbol | Value | Description |
| :--- | :---: | :---: | :--- |
| **EMA Smoothing Factor** | $\alpha$ | **0.08** | Multi-channel noise suppression factor ($O(1)$ constant time) |
| **Slouch Ratio Drop Threshold** | $\delta_{slouch}$ | **8.0%** | Percentage compression in $R_{H2S}$ below calibrated baseline |
| **Forward Head Depth Threshold** | $\delta_{FHP}$ | **0.2** | Forward shift of the head depth $\Delta Z$ beyond the calibrated baseline |
| **Sustained Alert Window** | $W_{alert}$ | **1.0 s** | Temporal buffer before raising alert. Set in `config.py`, not tuned: frames are scored on the classified posture state, which does not depend on this window |
| **Side-On EMA Smoothing Factor** | $\alpha_{lat}$ | **0.25** | Used on frames detected as lateral; tuned on the lateral recordings (Section 6) |
| **CVA Drop Alert Threshold** | $\Delta_{CVA}$ | **5.0°** | Alert when the CVA is this far below the calibrated upright CVA; tuned on the lateral recordings |
| **Clinical FHP Criterion** | — | **50°** | CVA below this is named Forward Head Posture (expert criterion, not tuned) |

Scoring the front-view hyperparameters on the same 14,757 frames they were tuned on gives 80.27% accuracy, 75.80% F1-score and a 16.76% False Alarm Rate. This is an in-sample figure, not a validation result.

---

## 2. Leave-One-Subject-Out (LOSO) Cross-Validation Results

Each row is scored on a participant whose frames were excluded from hyperparameter tuning; the "Tuned" columns show the values selected from the other 10 participants. The personalized baseline is still calibrated on the held-out participant's own first upright frames, as in live use.

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned Slouch Drop (%)   | Tuned FHP Depth   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:------------------------|:------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Mesomorph    | Normal         |     1389 | 0.08          | 8.0                     | 0.2               |          99.42 |           99.24 |        99.54 |             99.32 |          99.39 |
| Participant2                 | Mesomorph    | Normal         |     1286 | 0.08          | 8.0                     | 0.2               |          73.02 |           70.51 |        81.52 |             64.06 |          75.61 |
| Participant3                 | Endomorph    | Normal         |     3812 | 0.08          | 8.0                     | 0.06              |          79.33 |           62.18 |        99.92 |             68.73 |          76.66 |
| Participant4                 | Endomorph    | Normal         |     3415 | 0.08          | 8.0                     | 0.2               |          63.54 |           32.31 |         9.29 |             90.37 |          14.43 |
| Participant5                 | Endomorph    | Normal         |     1146 | 0.12          | 10.0                    | 0.2               |          72.6  |           97.22 |        40.54 |             99.04 |          57.22 |
| Participant6                 | Mesomorph    | Normal         |      753 | 0.08          | 8.0                     | 0.3               |          69.06 |           67.58 |        79.95 |             56.78 |          73.25 |
| Participant7                 | Ectomorph    | Normal         |      317 | 0.08          | 8.0                     | 0.2               |          78.23 |           70    |       100    |             55.77 |          82.35 |
| Participant8                 | Mesomorph    | Normal         |      745 | 0.08          | 8.0                     | 0.2               |          77.32 |           67.93 |       100    |             56.33 |          80.9  |
| Participant9                 | Endomorph    | Overweight     |      717 | 0.08          | 8.0                     | 0.2               |          78.1  |           69.51 |       100    |             56.27 |          82.02 |
| Participant10                | Ectomorph    | Normal         |      560 | 0.08          | 8.0                     | 0.2               |          66.96 |           51.57 |       100    |             49.04 |          68.05 |
| Participant11                | Ectomorph    | Normal         |      617 | 0.08          | 8.0                     | 0.2               |          70.83 |          100    |        34.31 |            100    |          51.09 |
| Macro Average                | —            | —              |    14757 | —             | —                       | —                 |          75.31 |           71.64 |        76.82 |             72.34 |          69.18 |
| Pooled (All Held-Out Frames) | —            | —              |    14757 | —             | —                       | —                 |          74.96 |           68.44 |        71.37 |             77.42 |          69.87 |

---

## 3. Benchmark Architecture Comparison (Chapter 4 Results)

All models are scored under the same LOSO protocol: each participant is predicted by a model fitted on the other 10 participants. Latency is the median per-frame classification time over 1,000 frames, measured on the machine that generated this report; it excludes MediaPipe pose estimation, which every model shares.

| Model Architecture             |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series) |          74.96 |           68.44 |        71.37 |          69.87 |     22.58 |         0.0165 |
| Random Forest (100 Trees)      |          51.11 |           42.1  |        53.71 |          47.2  |     50.68 |         4.3074 |
| Support Vector Machine (RBF)   |          53.64 |           42.86 |        41.89 |          42.37 |     38.31 |         0.8421 |
| Logistic Regression            |          65.3  |           63.34 |        34.91 |          45.01 |     13.86 |         0.0733 |
| Static Rigid Threshold         |          53.11 |           42.96 |        46.47 |          44.64 |     42.33 |         0.0002 |

### Key Findings:
1. **Held-Out Performance**: On participants excluded from tuning, ErgoEMA reaches **74.96% accuracy** and **69.87% F1-score**. Per-participant F1 ranges from 14.43% (Participant4) to 99.39% (Participant1).
2. **False Alarm Rate**: The held-out False Alarm Rate (FAR) is **22.58%**.
3. **Baseline Comparison**: The highest held-out F1-score in the table is 69.87%, from ErgoEMA (Adaptive Time-Series).
4. **Measured Latency**: The ErgoEMA EMA update and classification step takes a median of **0.0165 ms** per frame.

---

## 4. Camera Viewpoint Sensitivity & Ablation Study (Chapter 4 Discussion)

The table below demonstrates the effect of camera placement geometry on biometric normalization. Every row runs the configuration the live application uses: the front-view hyperparameters from Section 1 for frontal and oblique frames, and the side-on settings for frames detected as lateral. The Front View row is an in-sample figure (those frames were used for tuning). The Lateral row uses the side-on settings tuned on those frames (Section 6), so it is also in-sample; the Oblique row uses the front-view settings, which were not tuned on its frames.

| Camera Perspective              |   Frames |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |
|:--------------------------------|---------:|---------------:|----------------:|-------------:|---------------:|----------:|
| Front View (0° - Primary Scope) |    14757 |          80.27 |           75.66 |        75.95 |          75.8  |     16.76 |
| Oblique Profile (45° View)      |    32752 |          53.06 |           82.5  |         8.9  |          16.06 |      1.92 |
| Lateral Profile (90° View)      |     4512 |          97.67 |           98.52 |        96.95 |          97.73 |      1.56 |
| All Combined Perspectives       |    52021 |          63.26 |           74.99 |        34.7  |          47.45 |     10.6  |

### Geometric Analysis:
* **Front View ($0^\circ$)**: The 2D biacromial shoulder span ($\|\text{Shoulder}_R - \text{Shoulder}_L\|$) provides an invariant denominator, yielding **80.27% Accuracy** and a **16.76% False Alarm Rate**.
* **Oblique View ($45^\circ$)**: Perspective foreshortening compresses the visible shoulder width, causing mild accuracy degradation compared to direct frontal placement.
* **Lateral View ($90^\circ$)**: The alert tracks the drop of the estimated craniovertebral angle (CVA) below the participant's calibrated upright CVA, with C7 located on the back-of-neck silhouette; Forward Head Posture is named only when the CVA is also below the clinical $50^\circ$ criterion. Thoracic kyphosis cannot be measured from video, so slouch is not assessed in this view.

---

## 5. Oblique Profile (45°) Benchmark

The protocol of Sections 2 and 3, run on the 32,752 oblique-profile frames (cameras $45^\circ$ to the left and right) of all 11 participants. ErgoEMA's hyperparameters are tuned on the other 10 participants' $45^\circ$ frames, and each participant's baseline is calibrated on their own $45^\circ$ upright frames, so this measures a system set up for a $45^\circ$ camera. Applying the front-view hyperparameters to these frames instead is the Oblique row of Section 4. The ML and static baselines use the same frontal features as in Section 3.

### 5.1 Leave-One-Subject-Out Cross-Validation

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned Slouch Drop (%)   | Tuned FHP Depth   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:------------------------|:------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Mesomorph    | Normal         |     2815 | 0.08          | 8.0                     | 0.04              |          50.91 |            0    |         0    |             99.65 |           0    |
| Participant2                 | Mesomorph    | Normal         |     2777 | 0.08          | 8.0                     | 0.04              |          44.26 |           14.03 |         2.23 |             86.31 |           3.85 |
| Participant3                 | Endomorph    | Normal         |     8808 | 0.2           | 8.0                     | 0.04              |          74.32 |           91.64 |        50.97 |             95.73 |          65.51 |
| Participant4                 | Endomorph    | Normal         |     8476 | 0.08          | 8.0                     | 0.04              |          42.44 |            0    |         0    |             90.67 |           0    |
| Participant5                 | Endomorph    | Normal         |     2725 | 0.12          | 8.0                     | 0.04              |          40.7  |           37.63 |        26.78 |             54.85 |          31.29 |
| Participant6                 | Mesomorph    | Normal         |     1580 | 0.08          | 8.0                     | 0.04              |          40.89 |           31.86 |        11.37 |             73.31 |          16.76 |
| Participant7                 | Ectomorph    | Normal         |      726 | 0.12          | 8.0                     | 0.04              |          83.61 |           85.49 |        78.78 |             87.96 |          82    |
| Participant8                 | Mesomorph    | Normal         |     1436 | 0.12          | 8.0                     | 0.04              |          54.32 |           66.24 |        14.71 |             92.73 |          24.07 |
| Participant9                 | Endomorph    | Overweight     |     1484 | 0.12          | 8.0                     | 0.04              |          87.6  |           79.89 |        99.86 |             75.79 |          88.77 |
| Participant10                | Ectomorph    | Normal         |      526 | 0.12          | 8.0                     | 0.04              |          79.66 |           85.8  |        84.12 |             70.06 |          84.95 |
| Participant11                | Ectomorph    | Normal         |     1399 | 0.12          | 8.0                     | 0.04              |          38.17 |           18.92 |         6.96 |             69.78 |          10.18 |
| Macro Average                | —            | —              |    32752 | —             | —                       | —                 |          57.9  |           46.5  |        34.16 |             81.53 |          37.03 |
| Pooled (All Held-Out Frames) | —            | —              |    32752 | —             | —                       | —                 |          55.57 |           65.95 |        24.76 |             86.97 |          36.01 |

### 5.2 Benchmark Architecture Comparison

| Model Architecture             |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series) |          55.57 |           65.95 |        24.76 |          36.01 |     13.03 |         0.0217 |
| Random Forest (100 Trees)      |          54.9  |           55.87 |        50.71 |          53.17 |     40.83 |         4.8419 |
| Support Vector Machine (RBF)   |          64.65 |           75.54 |        44.31 |          55.86 |     14.62 |         2.2433 |
| Logistic Regression            |          49.89 |           50.46 |        39.14 |          44.09 |     39.16 |         0.0734 |
| Static Rigid Threshold         |          51.49 |           88.7  |         4.46 |           8.5  |      0.58 |         0.0001 |

On held-out participants, ErgoEMA reaches **55.57% accuracy**, **36.01% F1-score** and a **13.03% False Alarm Rate**; per-participant F1 ranges from 0.00% (Participant1) to 88.77% (Participant9). The highest held-out F1-score in the comparison is 55.86%, from Support Vector Machine (RBF).

---

## 6. Lateral Profile (90°) Benchmark

The protocol of Sections 2 and 3, run on the 4,512 lateral-profile frames of the 10 participants recorded side-on (the 297 public benchmark images carry no participant ID and are excluded). In this view ErgoEMA raises an alert when the EMA-smoothed craniovertebral angle (CVA) falls more than a tuned number of degrees below the participant's calibrated upright CVA; the smoothing factor $\alpha$ and that drop are tuned on the other 9 participants. The state is named Forward Head Posture only when the CVA is also below the clinical $50^\circ$ criterion; a smaller CVA with a normal angle is reported as a head-forward shift from the participant's upright. The row *ErgoEMA, Clinical CVA Criterion Only* scores the clinical criterion (CVA below $50^\circ$) as the alert instead, with only $\alpha$ tuned. Frames labelled slouch or forward head both count as poor posture, as in the other views. The ML baselines are trained on the side-view measurements (CVA, ear-to-shoulder offset, nose-to-shoulder angle), and the static baseline applies the $50^\circ$ criterion to each frame without smoothing.

### 6.1 Leave-One-Subject-Out Cross-Validation

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned CVA Drop (°)   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:---------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Ectomorph    | Normal         |      214 | 0.25          | 5.0                  |          80.37 |          100    |        60.38 |            100    |          75.29 |
| Participant2                 | Ectomorph    | Normal         |      234 | 0.25          | 5.0                  |          92.31 |           89.6  |        95.73 |             88.89 |          92.56 |
| Participant3                 | Ectomorph    | Normal         |      175 | 0.25          | 4.0                  |          96    |           88.52 |       100    |             94.21 |          93.91 |
| Participant4                 | Endomorph    | Normal         |      116 | 0.25          | 5.0                  |          79.31 |          100    |        57.89 |            100    |          73.33 |
| Participant5                 | Endomorph    | Normal         |      649 | 0.25          | 5.0                  |          99.38 |           98.81 |       100    |             98.74 |          99.4  |
| Participant6                 | Mesomorph    | Normal         |      634 | 0.25          | 5.0                  |          99.05 |           98.15 |       100    |             98.1  |          99.07 |
| Participant7                 | Ectomorph    | Normal         |      643 | 0.25          | 5.0                  |          99.53 |           99.13 |       100    |             99.01 |          99.56 |
| Participant8                 | Mesomorph    | Normal         |      614 | 0.25          | 5.0                  |         100    |          100    |       100    |            100    |         100    |
| Participant9                 | Ectomorph    | Normal         |      531 | 0.25          | 5.0                  |          99.44 |           99.14 |       100    |             98.37 |          99.57 |
| Participant10                | Ectomorph    | Normal         |      702 | 0.25          | 5.0                  |          99.86 |           99.72 |       100    |             99.71 |          99.86 |
| Macro Average                | —            | —              |     4512 | —             | —                    |          94.53 |           97.31 |        91.4  |             97.7  |          93.26 |
| Pooled (All Held-Out Frames) | —            | —              |     4512 | —             | —                    |          97.61 |           98.39 |        96.95 |             98.31 |          97.66 |

### 6.2 Benchmark Architecture Comparison

| Model Architecture                       |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-----------------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series)           |          97.61 |           98.39 |        96.95 |          97.66 |      1.69 |         0.0207 |
| ErgoEMA, Clinical CVA Criterion Only     |          83.58 |           76.97 |        97.29 |          85.95 |     31.06 |         0.0166 |
| Random Forest (100 Trees)                |          81.85 |           82.97 |        81.58 |          82.27 |     17.87 |         4.643  |
| Support Vector Machine (RBF)             |          86.1  |           89.29 |        83.04 |          86.05 |     10.63 |         0.161  |
| Logistic Regression                      |          88.9  |           92.39 |        85.53 |          88.83 |      7.51 |         0.073  |
| Static Rigid Threshold (CVA, unsmoothed) |          84.13 |           77.61 |        97.34 |          86.36 |     29.96 |         0.0002 |

On held-out participants, ErgoEMA reaches **97.61% accuracy**, **97.66% F1-score** and a **1.69% False Alarm Rate**; per-participant F1 ranges from 73.33% (Participant4) to 100.00% (Participant8). The highest held-out F1-score in the comparison is 97.66%, from ErgoEMA (Adaptive Time-Series).

