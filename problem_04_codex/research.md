# Problem 04 research

- Train: 120 labeled sequences, 15 steps each, six force/torque channels.
- Test: 41 sequences of the same shape. No missing values or duplicate sample/step keys.
- Class counts (0–4): 33, 19, 33, 16, 19.
- Train/test IDs are disjoint; labels and submission IDs match their respective series.
- Class 0 has relatively low variation; class 3 has large negative force_z and variation.
- Sampling interval, experimental grouping, and sensor units are unknown.
- Metric: macro-F1 over labels [0,1,2,3,4], zero_division=0.
- Validation must split entire sequences. Test data must not select models.

## Approved refinement results
Four candidates scored 0.73374, 0.75131, 0.74563, and 0.72015 on identical folds. None exceeded baseline 0.80024. Model and submission retained byte-for-byte; results saved to refinement_results.csv. No additional search performed.

## Temporal feature comparison
Same five folds and ExtraTrees settings: autocorrelation 0.72389, mean absolute second difference 0.74492, direction reversals 0.76752, all 0.73003. Baseline 0.80024 retained. Constant, linear, and alternating synthetic signals passed feature definition checks. Model/submission hashes unchanged.
