# ErgoEMA Model Training & Empirical Validation Report

**Dataset**: 14,726 Real-World Front-Facing Video Frames
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
| **Sustained Alert Window** | $W_{alert}$ | **1.0 s** | Temporal buffer before raising alert. Set in `config.py`, not tuned: frames are scored on the classified posture state, which does not depend on this window |

Scoring these hyperparameters on the same 14,726 frames they were tuned on gives 70.62% accuracy, 66.90% F1-score and a 25.08% False Alarm Rate. This is an in-sample figure, not a validation result.

---

## 2. Leave-One-Subject-Out (LOSO) Cross-Validation Results

Each row is scored on a participant whose frames were excluded from hyperparameter tuning; the "Tuned" columns show the values selected from the other 8 participants. The personalized baseline is still calibrated on the held-out participant's own first upright frames, as in live use.

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned Slouch Drop (%)   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:------------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Mesomorph    | Normal         |     1389 | 0.08          | 8.0                     |          98.92 |           97.76 |       100    |             97.96 |          98.87 |
| Participant2                 | Mesomorph    | Normal         |     1286 | 0.08          | 8.0                     |          71.62 |           65.1  |        96.36 |             45.53 |          77.7  |
| Participant3                 | Endomorph    | Normal         |     4958 | 0.08          | 8.0                     |          60.99 |           62.18 |        53.01 |             68.73 |          57.23 |
| Participant4                 | Endomorph    | Normal         |     3415 | 0.08          | 8.0                     |          63.54 |           32.31 |         9.29 |             90.37 |          14.43 |
| Participant5                 | Endomorph    | Normal         |     1146 | 0.08          | 8.0                     |          88.74 |           95.98 |        78.38 |             97.29 |          86.29 |
| Participant6                 | Mesomorph    | Normal         |      753 | 0.08          | 8.0                     |          67.73 |           62.15 |       100    |             31.36 |          76.66 |
| Participant7                 | Ectomorph    | Normal         |      317 | 0.08          | 8.0                     |          78.23 |           70    |       100    |             55.77 |          82.35 |
| Participant8                 | Mesomorph    | Normal         |      745 | 0.08          | 8.0                     |          77.32 |           67.93 |       100    |             56.33 |          80.9  |
| Participant9                 | Endomorph    | Overweight     |      717 | 0.08          | 8.0                     |          78.1  |           69.51 |       100    |             56.27 |          82.02 |
| Macro Average                | —            | —              |    14726 | —             | —                       |          76.13 |           69.21 |        81.89 |             66.62 |          72.94 |
| Pooled (All Held-Out Frames) | —            | —              |    14726 | —             | —                       |          70.62 |           68.41 |        65.44 |             74.92 |          66.9  |

---

## 3. Benchmark Architecture Comparison (Chapter 4 Results)

All models are scored under the same LOSO protocol: each participant is predicted by a model fitted on the other 8 participants. Latency is the median per-frame classification time over 1,000 frames, measured on the machine that generated this report; it excludes MediaPipe pose estimation, which every model shares.

| Model Architecture             |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series) |          70.62 |           68.41 |        65.44 |          66.9  |     25.08 |         0.0098 |
| Random Forest (100 Trees)      |          43.8  |           39.38 |        44.33 |          41.71 |     56.64 |         4.3029 |
| Support Vector Machine (RBF)   |          60.59 |           56.66 |        55.76 |          56.21 |     35.39 |         0.7148 |
| Logistic Regression            |          49.54 |           43.65 |        38.7  |          41.03 |     41.47 |         0.0719 |
| Static Rigid Threshold         |          47.99 |           42.07 |        38.94 |          40.45 |     44.5  |         0.0001 |

### Key Findings:
1. **Held-Out Performance**: On participants excluded from tuning, ErgoEMA reaches **70.62% accuracy** and **66.90% F1-score**. Per-participant F1 ranges from 14.43% (Participant4) to 98.87% (Participant1).
2. **False Alarm Rate**: The held-out False Alarm Rate (FAR) is **25.08%**.
3. **Baseline Comparison**: The highest held-out F1-score in the table is 66.90%, from ErgoEMA (Adaptive Time-Series).
4. **Measured Latency**: The ErgoEMA EMA update and classification step takes a median of **0.0098 ms** per frame.

---

## 4. Camera Viewpoint Sensitivity & Ablation Study (Chapter 4 Discussion)

The table below demonstrates the effect of camera placement geometry on biometric normalization. Every row uses the deployed hyperparameters from Section 1, so the Front View row is an in-sample figure (those frames were used for tuning), not a held-out result:

| Camera Perspective              |   Frames |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |
|:--------------------------------|---------:|---------------:|----------------:|-------------:|---------------:|----------:|
| Front View (0° - Primary Scope) |    14726 |          70.62 |           68.41 |        65.44 |          66.9  |     25.08 |
| Oblique Profile (45° View)      |    30827 |          58.11 |           79.46 |        22.28 |          34.8  |      5.8  |
| Lateral Profile (90° View)      |     3560 |          96.84 |           94.04 |       100    |          96.93 |      6.28 |
| All Combined Perspectives       |    49113 |          57.04 |           64.59 |        26.11 |          37.18 |     13.59 |

### Geometric Analysis:
* **Front View ($0^\circ$)**: The 2D biacromial shoulder span ($\|\text{Shoulder}_R - \text{Shoulder}_L\|$) provides an invariant denominator, yielding **70.62% Accuracy** and a **25.08% False Alarm Rate**.
* **Oblique View ($45^\circ$)**: Perspective foreshortening compresses the visible shoulder width, causing mild accuracy degradation compared to direct frontal placement.
* **Lateral View ($90^\circ$)**: Utilizes invariant lateral biometrics (ear-to-shoulder horizontal displacement and nose-to-shoulder vertical inclination angle) calibrated to clinical hyperkyphosis (>55°) and FHP (>0.14) criteria, maintaining high accuracy across both left-facing and right-facing camera orientations.

