"""Recompute historical OOF evidence and implementation diagnostics; no fitting new candidates."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
result = {}

# Historical P2 OOF ordering is classes.json then sorted paths, as load_images.py.
classes = json.loads((ROOT/'problem_02/classes.json').read_text())
y = np.concatenate([np.full(len(list((ROOT/'problem_02'/c['folder']).glob('*.png'))), c['label']) for c in classes])
probs = [np.load(ROOT/f'problem_02/results/oof_proba_cnn_{name}.npy') for name in ['C0','C0_s1','C0_s2']]
preds = np.stack([p.argmax(1) for p in probs])
ensemble = np.mean(probs, axis=0).argmax(1)
err = ensemble != y
result['P2_saved_oof'] = dict(ensemble_macro_f1=float(f1_score(y,ensemble,labels=range(7),average='macro')),
    ensemble_errors=int(err.sum()), all_seeds_wrong=int(np.all(preds != y,axis=0).sum()),
    ensemble_errors_all_seeds_wrong=int((err & np.all(preds != y,axis=0)).sum()),
    coat_pullover_errors=int((((y==2)&(ensemble==6))|((y==6)&(ensemble==2))).sum()))

# Recalculate full precision scores from saved probabilities rather than rounded JSON.
d = np.load(ROOT/'problem_05/cache/posthoc_oof.npz')
y = d['y']
assert np.array_equal(y, pd.read_csv(ROOT/'problem_05/data/train.csv').label)
names = ['b1', 'ft', 'ensemble', 'ft128', 'ensemble128']
pr = [d['P_lr'], d['ft64_lr3e-4'], (d['P_lr']+d['ft64_lr3e-4'])/2,
      d['ft128_lr3e-4'], (d['P_lr']+d['ft128_lr3e-4'])/2]
scores = {name: [] for name in names}
for _, b in StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)), y):
    for name, p in zip(names, pr):
        scores[name].append(float(f1_score(y[b],p[b].argmax(1),labels=range(5),average='macro',zero_division=0)))
comparisons = {}
for cand, base in [('ensemble','b1'), ('ensemble','ft'), ('ft','b1'), ('ft128','ft'), ('ensemble128','ensemble')]:
    diff = np.array(scores[cand])-scores[base]
    comparisons[f'{cand}_vs_{base}'] = dict(mean_d=float(diff.mean()),sd_d=float(diff.std()),
        wins=int((diff>1e-9).sum()),losses=int((diff<-1e-9).sum()),ties=int((np.abs(diff)<=1e-9).sum()))
result['P5_saved_oof'] = dict(scores=scores,comparisons=comparisons)

# P3 check target alignment independently at two historical forecast origins.
spec = importlib.util.spec_from_file_location('forecast_review',ROOT/'problem_03/forecast.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
df = pd.read_csv(ROOT/'problem_03/train.csv',parse_dates=['timestamp'])
checks = []
for origin in ['2014-01-01 23:00:00','2014-12-25 23:00:00']:
    hist = df[df.timestamp <= pd.Timestamp(origin)].copy()
    X,R,x0,b0 = module.residual_data(hist)
    days = module.day_matrix(hist, 'temperature_c')
    base = module.m1_all(days, **module.M1_BEST)
    # Last training origin must be three days before the current origin.
    np.testing.assert_allclose(R[-1]+base[-4],hist.temperature_c.iloc[-72:].to_numpy(),atol=1e-12)
    pred = module.m2_ridge(hist, **module.M2_BEST)
    assert pred.shape == (72,) and np.isfinite(pred).all()
    checks.append(dict(origin=origin, training_rows=len(X),feature_count=X.shape[1],
                       last_target_equals_history_tail=True,finite_72_predictions=True))
result['P3_target_alignment'] = checks

# A class-absent group needs a fixed labels list to use the prescribed metric.
tr = pd.read_csv(ROOT/'problem_07/data/train.csv')
one = tr.groupby('session_id').label.nunique()
result['P7_single_class_sessions'] = {}
for sess in one[one == 1].index:
    labels = tr.loc[tr.session_id == sess,'label']
    result['P7_single_class_sessions'][sess] = dict(rows=len(labels), label=int(labels.iloc[0]),
        perfect_prediction_default_macro=float(f1_score(labels,labels,average='macro',zero_division=0)),
        perfect_prediction_fixed_macro=float(f1_score(labels,labels,labels=[0,1],average='macro',zero_division=0)))

# Keep initial selection and follow-up sensitivity separate, plus their combined summary.
pairfiles = [OUT/'p1_pairs.csv', OUT/'p1_confirm_pairs.csv']
if all(p.exists() for p in pairfiles):
    pairs = pd.concat([pd.read_csv(p) for p in pairfiles], ignore_index=True)
    pairs = pairs[pairs.model.isin(['baseline','impute_then_derive'])]
    wide = pairs.pivot(index='repeat',columns='model',values='score')
    diff = wide.impute_then_derive-wide.baseline
    result['P1_combined'] = dict(repeats=len(wide), baseline=float(wide.baseline.mean()),
        candidate=float(wide.impute_then_derive.mean()), mean_d=float(diff.mean()),sd_d=float(diff.std(ddof=0)),
        wins=int((diff>1e-9).sum()),losses=int((diff<-1e-9).sum()),ties=int((diff.abs()<=1e-9).sum()))
    pairs.to_csv(OUT/'p1_combined_pairs.csv',index=False)
    train = pd.read_csv(ROOT/'problem_01/train.csv')
    missing = train.drop(columns=['record_id','inspection_required']).isna().any(axis=1).to_numpy()
    changes = {'missing':{'fix':0,'break':0}, 'complete':{'fix':0,'break':0}}
    for rep in range(10):
        prefix = 'p1' if rep < 3 else 'p1_confirm'
        z = np.load(OUT/f'{prefix}_oof_{rep}.npz')
        assert np.array_equal(z['y'],train.inspection_required)
        for name, mask in [('missing',missing),('complete',~missing)]:
            changes[name]['fix'] += int((mask & (z['baseline']!=z['y']) & (z['impute_then_derive']==z['y'])).sum())
            changes[name]['break'] += int((mask & (z['baseline']==z['y']) & (z['impute_then_derive']!=z['y'])).sum())
    result['P1_combined']['changed_predictions_over_10_repeats'] = changes

(OUT/'recheck.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
