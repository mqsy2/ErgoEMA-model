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
| **EMA Smoothing Factor** | $\alpha$ | **0.12** | Multi-channel noise suppression factor ($O(1)$ constant time) |
| **Slouch Ratio Drop Threshold** | $\delta_{slouch}$ | **10.0%** | Percentage compression in $R_{H2S}$ below calibrated baseline |
| **Sustained Alert Window** | $W_{alert}$ | **1.0 s** | Temporal buffer before raising alert. Set in `config.py`, not tuned: frames are scored on the classified posture state, which does not depend on this window |

Scoring these hyperparameters on the same 3,168 frames they were tuned on gives 94.79% accuracy, 94.52% F1-score and a 3.42% False Alarm Rate. This is an in-sample figure, not a validation result.

---

## 2. Leave-One-Subject-Out (LOSO) Cross-Validation Results

Each row is scored on a participant whose frames were excluded from hyperparameter tuning; the "Tuned" columns show the values selected from the other 3 participants. The personalized baseline is still calibrated on the held-out participant's own first upright frames, as in live use.

| Participant                  | Somatotype   | BMI Category   |   Frames | Tuned alpha   | Tuned Slouch Drop (%)   |   Accuracy (%) |   Precision (%) |   Recall (%) |   Specificity (%) |   F1-Score (%) |
|:-----------------------------|:-------------|:---------------|---------:|:--------------|:------------------------|---------------:|----------------:|-------------:|------------------:|---------------:|
| Participant1                 | Mesomorph    | Normal         |     1389 | 0.25          | 12.0                    |          92.66 |           90.95 |        93.73 |             91.7  |          92.32 |
| Participant7                 | Mesomorph    | Normal         |      317 | 0.08          | 10.0                    |          70.98 |          100    |        42.86 |            100    |          60    |
| Participant8                 | Mesomorph    | Normal         |      745 | 0.12          | 10.0                    |          97.99 |           98.04 |        97.77 |             98.19 |          97.9  |
| Participant9                 | Endomorph    | Overweight     |      717 | 0.08          | 10.0                    |          98.47 |          100    |        96.93 |            100    |          98.44 |
| Macro Average                | —            | —              |     3168 | —             | —                       |          90.02 |           97.25 |        82.82 |             97.47 |          87.17 |
| Pooled (All Held-Out Frames) | —            | —              |     3168 | —             | —                       |          93.06 |           95.3  |        90.07 |             95.85 |          92.61 |

---

## 3. Benchmark Architecture Comparison (Chapter 4 Results)

All models are scored under the same LOSO protocol: each participant is predicted by a model fitted on the other 3 participants. Latency is the median per-frame classification time over 1,000 frames, measured on the machine that generated this report; it excludes MediaPipe pose estimation, which every model shares.

| Model Architecture             |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1-Score (%) |   FAR (%) |   Latency (ms) |
|:-------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|---------------:|
| ErgoEMA (Adaptive Time-Series) |          93.06 |           95.3  |        90.07 |          92.61 |      4.15 |         0.0076 |
| Random Forest (100 Trees)      |          63.23 |           63.28 |        56.96 |          59.95 |     30.91 |         6.4424 |
| Support Vector Machine (RBF)   |          31.85 |           22.11 |        16.26 |          18.74 |     53.57 |         0.3074 |
| Logistic Regression            |          62.09 |           59.73 |        66.17 |          62.78 |     41.72 |         0.1276 |
| Static Rigid Threshold         |          72.73 |           78.53 |        59.96 |          68    |     15.33 |         0.0001 |

### Key Findings:
1. **Held-Out Performance**: On participants excluded from tuning, ErgoEMA reaches **93.06% accuracy** and **92.61% F1-score**. Per-participant F1 ranges from 60.00% (Participant7) to 98.44% (Participant9).
2. **False Alarm Rate**: The held-out False Alarm Rate (FAR) is **4.15%**.
3. **Baseline Comparison**: The highest held-out F1-score in the table is 92.61%, from ErgoEMA (Adaptive Time-Series).
4. **Measured Latency**: The ErgoEMA EMA update and classification step takes a median of **0.0076 ms** per frame.

