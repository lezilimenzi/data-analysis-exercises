"""Diagnose baseline OOF errors without changing model or submission."""
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from train import ROOT, features


def main():
    protected = ['model.joblib', 'submission.csv', 'metrics.json', 'oof_predictions.csv']
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in protected}
    labels = pd.read_csv(ROOT / 'train_labels.csv')
    old = pd.read_csv(ROOT / 'oof_predictions.csv')
    artifact = joblib.load(ROOT / 'model.joblib')
    x = features(pd.read_csv(ROOT / 'train_series.csv'), labels.sample_id)[artifact['representation']]
    y = labels.label.to_numpy()
    probabilities = np.zeros((len(y), 5))
    neighbors = []
    for fold, (fit, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y), 1):
        model = clone(artifact['model']).fit(x[fit], y[fit])
        probabilities[valid] = model.predict_proba(x[valid])
        assert np.array_equal(model.classes_, np.arange(5))
        assert (old.iloc[valid]['fold'] == fold).all()
        scaler = StandardScaler().fit(x[fit])
        a, b = scaler.transform(x[valid]), scaler.transform(x[fit])
        distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
        for row, idx in enumerate(valid):
            for rank, j in enumerate(np.argsort(distances[row])[:3], 1):
                neighbor = fit[j]
                neighbors.append({'sample_id': labels.sample_id.iloc[idx], 'label': int(y[idx]),
                    'prediction': int(probabilities[idx].argmax()), 'fold': fold,
                    'rank': rank, 'neighbor_id': labels.sample_id.iloc[neighbor],
                    'neighbor_label': int(y[neighbor]), 'distance': float(distances[row, j])})
    pred = probabilities.argmax(axis=1)
    assert np.array_equal(pred, old.prediction)
    assert np.allclose(probabilities.sum(axis=1), 1)
    out = old.copy()
    for c in range(5):
        out[f'p{c}'] = probabilities[:, c]
    ordered = np.sort(probabilities, axis=1)
    out['confidence'] = ordered[:, -1]
    out['top2_margin'] = ordered[:, -1] - ordered[:, -2]
    out['true_probability'] = probabilities[np.arange(len(y)), y]
    out['correct'] = pred == y
    out.to_csv(ROOT / 'error_probabilities.csv', index=False)
    near = pd.DataFrame(neighbors)
    near.to_csv(ROOT / 'error_neighbors.csv', index=False)
    target = out[out.label.isin([1, 2])]
    print('GROUP SUMMARY\n', target.groupby(['label', 'prediction']).agg(n=('sample_id', 'size'),
        confidence_median=('confidence', 'median'), margin_median=('top2_margin', 'median')).round(4).to_string())
    errors = target[~target.correct].sort_values('confidence', ascending=False)
    print('ERRORS\n', errors[['sample_id','label','prediction','p0','p1','p2','confidence','top2_margin']].round(4).to_string(index=False))
    pair = errors[errors.prediction.isin([1,2])]
    print('PAIR ERRORS', len(pair), 'margin <= .1', int((pair.top2_margin <= .1).sum()),
          'confidence >= .7', int((pair.confidence >= .7).sum()))
    nn = near[near.sample_id.isin(pair.sample_id)]
    print('PAIR NEIGHBORS\n', nn.round(3).to_string(index=False))
    for description, subset in [('pair_errors', pair), ('class12_correct', target[target.correct])]:
        first = near[(near['rank']==1)&near.sample_id.isin(subset.sample_id)]
        print(description, 'nearest matches truth', int((first.label==first.neighbor_label).sum()), '/',len(first))
    for f in protected:
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == hashes[f]
    print('Protected artifacts unchanged; OOF labels exactly reproduced.')


if __name__ == '__main__':
    main()
