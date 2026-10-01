# ErgoEMA Model Training & Empirical Validation Report

**Dataset**: 3,168 Real-World Front-Facing Video Frames
**Participants**: 4 Study Participants (Participant1, Participant7, Participant8, Participant9)
**Methodology**: Online Adaptive Calibration + Exponential Moving Average (EMA)
**Validation Protocol**: Leave-One-Subject-Out (hyperparameters tuned on 3 participants, scored on the held-out participant)

---

## 1. Deployed Hyperparameters

Tuned on all 4 participants and saved to `optimized_parameters.json` for the live application:

| Hyperparameter | Symbol | Value | Description |
| :--- | :---: | :---: | :--- |
| **EMA Smoothing Factor** | $\alpha$ | **0.15** | Multi-channel noise suppression factor ($O(1)$ constant time) |
| **Slouch Ratio Drop Threshold** | $\delta_{slouch}$ | **10.0%** | Percentage compression in $R_{H2S}$ below calibrated baseline |
| **Forward Head Depth Threshold** | $\delta_{FHP}$ | **0.15** | Forward shift of the head depth $\Delta Z$ beyond the calibrated baseline |
| **Sustained Alert Window** | $W_{alert}$ | **1.0 s** | Temporal buffer before raising alert. Set in `config.py`, not tuned: frames are scored on the classified posture state, which does not depend on this window |

Scoring the front-view hyperparameters on the same 3,168 frames they were tuned on gives 96.34% accuracy, 96.09% F1-score and a 0.61% False Alarm Rate. This is an in-sample figure, not a validation result.

---

## 2. Leave-One-Subject-Out (LOSO) Cross-Validation Results

Each row is scored on a participant whose frames were excluded from hyperparameter tuning; the "Tuned" columns show the values selected from the other 3 participants. The personalized baseline is still calibrated on the held-out participant's own first upright frames, as in live use.

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned Slouch Drop (%)   | Tuned FHP Depth   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:------------------------|:------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Mesomorph    | Normal         |     1389 | 0.25          | 12.0                    | 0.1               |          95.75 |            97.3 |        93.58 |             97.69 |          95.4  |
| Participant7                 | Mesomorph    | Normal         |      317 | 0.12          | 10.0                    | 0.15              |          71.92 |           100   |        44.72 |            100    |          61.8  |
| Participant8                 | Mesomorph    | Normal         |      745 | 0.25          | 10.0                    | 0.15              |          96.38 |            93.9 |        98.88 |             94.06 |          96.33 |
| Participant9                 | Endomorph    | Overweight     |      717 | 0.12          | 10.0                    | 0.15              |          99.02 |           100   |        98.04 |            100    |          99.01 |
| Macro Average                | —            | —              |     3168 | —             | —                       | —                 |          90.77 |            97.8 |        83.81 |             97.94 |          88.14 |
| Pooled (All Held-Out Frames) | —            | —              |     3168 | —             | —                       | —                 |          94.26 |            97.2 |        90.73 |             97.56 |          93.85 |

---

## 3. Benchmark Architecture Comparison (Chapter 4 Results)

All models are scored under the same LOSO protocol: each participant is predicted by a model fitted on the other 3 participants. Latency is the median per-frame classification time over 1,000 frames, measured on the machine that generated this report; it excludes MediaPipe pose estimation, which every model shares.

| Model Architecture             |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series) |          94.26 |           97.2  |        90.73 |          93.85 |      2.44 |         0.0127 |
| Random Forest (100 Trees)      |          63.23 |           63.28 |        56.96 |          59.95 |     30.91 |         6.4664 |
| Support Vector Machine (RBF)   |          31.85 |           22.11 |        16.26 |          18.74 |     53.57 |         0.3033 |
| Logistic Regression            |          62.09 |           59.73 |        66.17 |          62.78 |     41.72 |         0.1266 |
| Static Rigid Threshold         |          72.73 |           78.53 |        59.96 |          68    |     15.33 |         0.0001 |

### Key Findings:
1. **Held-Out Performance**: On participants excluded from tuning, ErgoEMA reaches **94.26% accuracy** and **93.85% F1-score**. Per-participant F1 ranges from 61.80% (Participant7) to 99.01% (Participant9).
2. **False Alarm Rate**: The held-out False Alarm Rate (FAR) is **2.44%**.
3. **Baseline Comparison**: The highest held-out F1-score in the table is 93.85%, from ErgoEMA (Adaptive Time-Series).
4. **Measured Latency**: The ErgoEMA EMA update and classification step takes a median of **0.0127 ms** per frame.

