"""Run the four approved refinement candidates; retain baseline unless OOF improves."""
import json
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from train import ROOT, CHANNELS, CLASSES, features, score


def main(candidates=None, results_file='refinement_results.csv'):
    started = time.monotonic()
    baseline = ROOT / 'baseline'
    baseline.mkdir(exist_ok=True)
    for name in ['submission.csv', 'model.joblib', 'metrics.json', 'oof_predictions.csv', 'cv_results.csv']:
        if not (baseline / name).exists():
            shutil.copy2(ROOT / name, baseline / name)
    prior = json.loads((ROOT / 'metrics.json').read_text())
    labels = pd.read_csv(ROOT / 'train_labels.csv')
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    train = features(pd.read_csv(ROOT / 'train_series.csv'), labels.sample_id)
    y = labels.label.to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train['stats'], y))
    old_oof = pd.read_csv(ROOT / 'oof_predictions.csv')
    assert old_oof.sample_id.equals(labels.sample_id)
    assert np.isclose(score(y, old_oof.prediction), prior['selected']['macro_f1'])
    for i, (_, valid) in enumerate(folds):
        assert (old_oof.iloc[valid]['fold'] == i + 1).all()
    best_score = prior['selected']['macro_f1']
    winner = None
    rows = []
    if candidates is None:
        candidates = [('stats', 0.5), ('stats', 1.0),
                      ('relationships', 'sqrt'), ('relationships', 0.5)]
    for representation, max_features in candidates:
        model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                    class_weight='balanced', max_features=max_features, random_state=42, n_jobs=2)
        x = train[representation]
        oof = np.empty(len(y), dtype=int)
        scores = []
        for fit, valid in folds:
            fitted = clone(model).fit(x[fit], y[fit])
            oof[valid] = fitted.predict(x[valid])
            scores.append(score(y[valid], oof[valid]))
        row = {'name': f'extra_{representation}_features{max_features}', 'representation': representation,
               'macro_f1': score(y, oof), 'fold_mean': float(np.mean(scores)),
               'fold_std': float(np.std(scores)), **{f'fold_{i+1}': v for i, v in enumerate(scores)}}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row['macro_f1'] > best_score:
            best_score = row['macro_f1']
            winner = (row, model, oof.copy())
    pd.DataFrame(rows).to_csv(ROOT / results_file, index=False)
    if winner is None:
        print('No improvement; model and submission preserved.', flush=True)
        return
    row, model, oof = winner
    representation = row['representation']
    model.fit(train[representation], y)
    test = features(pd.read_csv(ROOT / 'test_series.csv'), sample.sample_id)
    submission = sample.copy()
    submission['label'] = model.predict(test[representation]).astype(int)
    assert submission.columns.tolist() == ['sample_id', 'label']
    assert len(submission) == 41 and submission.sample_id.is_unique
    assert submission.sample_id.equals(sample.sample_id)
    assert not submission.isna().any().any() and submission.label.isin(CLASSES).all()
    artifact = {'model': model, 'representation': representation, 'channels': CHANNELS, 'classes': CLASSES}
    joblib.dump(artifact, ROOT / 'model.joblib')
    loaded = joblib.load(ROOT / 'model.joblib')
    assert np.array_equal(loaded['model'].predict(test[representation]), submission.label)
    submission.to_csv(ROOT / 'submission.csv', index=False)
    pd.testing.assert_frame_equal(pd.read_csv(ROOT / 'submission.csv'), submission)
    output = labels.copy()
    output['prediction'] = oof
    output['fold'] = old_oof['fold']
    output.to_csv(ROOT / 'oof_predictions.csv', index=False)
    results = pd.concat([pd.read_csv(ROOT / 'cv_results.csv'), pd.DataFrame(rows)], ignore_index=True)
    results.drop_duplicates('name', keep='last').sort_values('macro_f1', ascending=False).to_csv(ROOT / 'cv_results.csv', index=False)
    old_submission = pd.read_csv(baseline / 'submission.csv')
    report = {'selected': row, 'class_f1': f1_score(y, oof, labels=CLASSES, average=None, zero_division=0).tolist(),
              'confusion_matrix': confusion_matrix(y, oof, labels=CLASSES).tolist(),
              'previous_macro_f1': prior['selected']['macro_f1'],
              'test_label_counts': submission.label.value_counts().sort_index().to_dict(),
              'changed_test_predictions_from_baseline': int((old_submission.label != submission.label).sum()),
              'elapsed_seconds': time.monotonic() - started,
              'validation_note': prior['validation_note']}
    (ROOT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
