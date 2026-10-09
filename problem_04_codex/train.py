from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
CHANNELS = ['force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z']
CLASSES = [0, 1, 2, 3, 4]


def features(frame, ids):
    assert not frame.duplicated(['sample_id', 'step']).any()
    assert set(frame.sample_id) == set(ids)
    groups = frame.groupby('sample_id')
    arrays = []
    for sid in ids:
        group = groups.get_group(sid).sort_values('step')
        assert group.step.tolist() == list(range(15))
        arrays.append(group[CHANNELS].to_numpy(dtype=float))
    x = np.stack(arrays)
    delta = np.abs(np.diff(x, axis=1))
    stats = np.concatenate([x.mean(1), x.std(1, ddof=1), x.min(1), x.max(1),
                            x[:, -1] - x[:, 0], delta.mean(1), delta.max(1)], axis=1)
    windows = np.concatenate([x[:, i:i+5].mean(1) for i in [0, 5, 10]], axis=1)
    result = {'stats': stats, 'raw': x.reshape(len(x), -1),
              'windows': np.concatenate([stats, windows], axis=1)}
    centered = x - x.mean(axis=1, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=1))
    cross = np.einsum('nti,ntj->nij', centered[:, :, :3], centered[:, :, 3:])
    denominator = norms[:, :3, None] * norms[:, None, 3:]
    correlations = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
    torque_rms = np.sqrt((x[:, :, 3:] ** 2).mean(axis=1))
    total = torque_rms.sum(axis=1, keepdims=True)
    shares = np.divide(torque_rms, total, out=np.zeros_like(torque_rms), where=total > 0)
    result['relationships'] = np.concatenate([stats, correlations.reshape(len(x), -1), shares], axis=1)
    # Lag-one Pearson correlation; constant segments receive zero.
    left, right = x[:, :-1], x[:, 1:]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denom = np.sqrt((left ** 2).sum(1) * (right ** 2).sum(1))
    autocorr = np.divide((left * right).sum(1), denom,
                         out=np.zeros_like(denom), where=denom > 0)
    second_diff = np.abs(np.diff(x, n=2, axis=1)).mean(axis=1)
    signed_diff = np.diff(x, axis=1)
    # Count strict sign reversals among 13 adjacent difference pairs; zeros do not reverse.
    turning = (signed_diff[:, :-1] * signed_diff[:, 1:] < 0).mean(axis=1)
    for name, values in [('autocorr', autocorr), ('second_diff', second_diff), ('turning', turning)]:
        result[name] = np.concatenate([stats, values], axis=1)
    result['temporal_all'] = np.concatenate([stats, autocorr, second_diff, turning], axis=1)
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def score(y, pred):
    return f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0)


def main():
    started = time.monotonic()
    labels = pd.read_csv(ROOT / 'train_labels.csv')
    submission = pd.read_csv(ROOT / 'sample_submission.csv')
    assert labels.sample_id.is_unique and submission.sample_id.is_unique
    assert labels.label.isin(CLASSES).all()
    assert set(labels.sample_id).isdisjoint(submission.sample_id)
    train = features(pd.read_csv(ROOT / 'train_series.csv'), labels.sample_id)
    test = features(pd.read_csv(ROOT / 'test_series.csv'), submission.sample_id)
    y = labels.label.to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train['stats'], y))
    results, predictions, candidates = [], {}, {}

    def evaluate(name, representation, model):
        x = train[representation]
        oof = np.empty(len(y), dtype=int)
        fold_scores = []
        for fit_idx, val_idx in folds:
            fitted = clone(model).fit(x[fit_idx], y[fit_idx])
            oof[val_idx] = fitted.predict(x[val_idx])
            fold_scores.append(score(y[val_idx], oof[val_idx]))
        row = {'name': name, 'representation': representation, 'macro_f1': score(y, oof),
               'fold_mean': float(np.mean(fold_scores)), 'fold_std': float(np.std(fold_scores)),
               **{f'fold_{i+1}': v for i, v in enumerate(fold_scores)}}
        results.append(row)
        predictions[name] = oof
        candidates[name] = (representation, model)
        print(f"{name}: OOF={row['macro_f1']:.5f}, folds={np.round(fold_scores, 3)}", flush=True)

    for leaf in [1, 2]:
        for weight in [None, 'balanced']:
            evaluate(f'extra_leaf{leaf}_{weight}', 'stats', ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=leaf, class_weight=weight,
                random_state=42, n_jobs=2))
    for representation in ['stats', 'raw']:
        for c in [1, 10, 100]:
            for weight in [None, 'balanced']:
                evaluate(f'svm_{representation}_C{c}_{weight}', representation,
                         make_pipeline(StandardScaler(), SVC(C=c, class_weight=weight)))

    best = max(results, key=lambda row: row['macro_f1'])
    matrix = confusion_matrix(y, predictions[best['name']], labels=CLASSES)
    if matrix[1, 2] + matrix[2, 1] > 0:
        for prefix in ['extra', 'svm_stats']:
            base = max((r for r in results if r['name'].startswith(prefix)), key=lambda r: r['macro_f1'])
            evaluate(base['name'] + '_windows', 'windows', candidates[base['name']][1])
    best = max(results, key=lambda row: row['macro_f1'])
    representation, model = candidates[best['name']]
    model.fit(train[representation], y)
    submission['label'] = model.predict(test[representation]).astype(int)
    assert submission.columns.tolist() == ['sample_id', 'label']
    assert len(submission) == 41 and not submission.isna().any().any()
    assert submission.label.isin(CLASSES).all()
    assert submission.sample_id.equals(pd.read_csv(ROOT / 'sample_submission.csv').sample_id)
    submission.to_csv(ROOT / 'submission.csv', index=False)
    saved = pd.read_csv(ROOT / 'submission.csv')
    pd.testing.assert_frame_equal(saved, submission)
    joblib.dump({'model': model, 'representation': representation, 'channels': CHANNELS,
                 'classes': CLASSES}, ROOT / 'model.joblib')
    loaded = joblib.load(ROOT / 'model.joblib')
    assert np.array_equal(loaded['model'].predict(test[representation]), saved.label)
    pd.DataFrame(results).sort_values('macro_f1', ascending=False).to_csv(ROOT / 'cv_results.csv', index=False)
    oof = labels.copy()
    oof['prediction'] = predictions[best['name']]
    oof['fold'] = -1
    for i, (_, val_idx) in enumerate(folds):
        oof.loc[val_idx, 'fold'] = i + 1
    oof.to_csv(ROOT / 'oof_predictions.csv', index=False)
    report = {'selected': best, 'class_f1': f1_score(y, oof.prediction, labels=CLASSES,
              average=None, zero_division=0).tolist(),
              'confusion_matrix': confusion_matrix(y, oof.prediction, labels=CLASSES).tolist(),
              'test_label_counts': saved.label.value_counts().sort_index().to_dict(),
              'elapsed_seconds': time.monotonic() - started,
              'validation_note': 'Model selection CV; no independent holdout. Experimental groups unknown.'}
    (ROOT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
