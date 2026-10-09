# Approved implementation plan

1. Build 42 features: per-channel mean, sample standard deviation, min, max, last-minus-first, mean/max absolute first difference.
2. Compare ExtraTrees and standardized RBF SVM using identical shuffled stratified 5-fold splits (seed 42). Evaluate limited hyperparameters and class weights.
3. Compare standardized flattened 90-value sequences as a secondary representation.
4. If the initial best model confuses classes 1 and 2, test one feature extension: channel means in three consecutive five-step windows.
5. Select by pooled out-of-fold macro-F1; retain fold scores, class scores, and confusion matrix. Small differences remain uncertain; this is model-selection validation, not an independent final evaluation.
6. Fit selected model on all 120 sequences, persist model and feature metadata, predict 41 test sequences in sample_submission order, and validate submission.csv.

Files: train.py, research.md, plan.md, cv_results.csv, oof_predictions.csv, metrics.json, model.joblib, submission.csv.
Time constraint: prioritize final artifact and checks; no broad search or deep learning.

## User-approved refinement
The user explicitly approved retaining the validation/features/model/time scope and updating submission only if validation improves. Preserve the current artifacts under baseline/.
- Error analysis: median force_z mean for misclassified class 1 is 8.17, close to correctly classified class 2 (8.70); the opposite overlap also occurs.
- Compare exactly four ExtraTrees candidates: original stats with max_features=0.5/1.0, and stats plus force/torque relationships with max_features=sqrt/0.5.
- Relationships: nine within-sequence force/torque cross-correlations and three torque-axis RMS shares. These are hypotheses, not verified physical contact locations.
- Keep identical folds and score definition. Replace final model/submission only for strictly higher pooled OOF macro-F1. Do not repeat the unsuccessful window experiment.

## Approved temporal comparison
Compare baseline plus six channel-wise lag-one Pearson correlations, six mean absolute second differences, six strict direction reversal fractions, and all 18 combined. Constant lag segments map to zero correlation; zero differences are not reversals, denominator is 13. Keep ExtraTrees settings and the original five folds. Update artifacts only on higher OOF Macro-F1. Run with python3 compare_temporal.py.
