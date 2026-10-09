"""Predeclared P1 and P5 comparisons. See PROTOCOL.md; never read test labels."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def summarize(rows, problem):
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'{problem}_pairs.csv', index=False)
    base = df[df.model == 'baseline'].set_index(['repeat', 'fold']).score
    result = {}
    for name, group in df.groupby('model', sort=False):
        values = group.set_index(['repeat', 'fold']).score
        d = (values-base).to_numpy(dtype=np.float64)
        win, loss = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        result[name] = dict(score=float(values.mean()), mean_d=float(d.mean()), sd_d=float(d.std()),
                            wins=win, losses=loss, ties=int(len(d)-win-loss), pairs=len(d),
                            passes=bool(d.mean()>0 and win>loss))
    (OUT / f'{problem}_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def p1(confirm=False):
    df = pd.read_csv(ROOT / 'problem_01/train.csv')
    y = df.inspection_required.to_numpy()
    raw = df[['ambient_temp_c','process_temp_c','rotation_rpm','torque_nm','tool_usage_min']].copy()
    raw['grade'] = df.equipment_grade.map({'A':0,'B':1,'C':2})
    X = raw.to_numpy(float)

    def derive(x):
        return np.column_stack([x, x[:,3]*x[:,2]*(2*np.pi/60), x[:,4]*x[:,3], x[:,1]-x[:,0]])

    names = ['baseline', 'impute_then_derive'] if confirm else ['baseline', 'impute_then_derive', 'extra_trees', 'rf_all_features']
    repeats = range(3, 10) if confirm else range(3)
    tag = 'p1_confirm' if confirm else 'p1'
    rows = []
    for r in repeats:
        oof = {name: np.zeros(len(y), int) for name in names}
        for fold, (a,b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42+r).split(X,y)):
            im = SimpleImputer(strategy='median').fit(derive(X[a]))
            fa, fb = im.transform(derive(X[a])), im.transform(derive(X[b]))
            imraw = SimpleImputer(strategy='median').fit(X[a])
            ra, rb = derive(imraw.transform(X[a])), derive(imraw.transform(X[b]))
            for name in names:
                cls = ExtraTreesClassifier if name == 'extra_trees' else RandomForestClassifier
                kw = {'max_features':1.0} if name == 'rf_all_features' else {}
                model = cls(n_estimators=300, random_state=42+r, n_jobs=2, **kw)
                aa, bb = (ra,rb) if name == 'impute_then_derive' else (fa,fb)
                pred = (model.fit(aa,y[a]).predict_proba(bb)[:, 1] >= 0.5).astype(int)
                oof[name][b] = pred
            print(f'p1 repeat={r} fold={fold}', flush=True)
        for name in names:
            rows.append(dict(repeat=r, fold=-1, model=name, score=f1_score(y,oof[name],zero_division=0)))
        np.savez_compressed(OUT / f'{tag}_oof_{r}.npz', y=y, **oof)
    summarize(rows, tag)
    # Verify the original reference before interpreting the new comparisons.
    old = pd.read_csv(ROOT / 'problem_01/posthoc_model_family_reps.csv').iloc[list(repeats)]['rf_der__f1_05'].to_numpy()
    now = np.array([x['score'] for x in rows if x['model']=='baseline'])
    assert np.allclose(now, old, atol=1e-12, rtol=0), (now, old)
    print('P1 original baseline reproduced.', flush=True)


def p5():
    folder = ROOT / 'problem_05_codex'
    df = pd.read_csv(folder / 'data/train.csv')
    # The helper hashes ordered image contents, model contents and preprocessing identifier.
    h = hashlib.sha256(b'p05-rgb-bilinear224-imagenet-v1')
    h.update((folder / 'models/image_encoder.onnx').read_bytes())
    for name in df.image_path:
        h.update(hashlib.sha256((folder/name).read_bytes()).digest())
    with np.load(folder / 'train_features.npz', allow_pickle=False) as cache:
        assert cache['signature'].item() == h.hexdigest(), 'Cache signature mismatch'
        X = cache['features'].astype(np.float64)
    assert X.shape == (len(df), 1280) and np.isfinite(X).all()
    y = df.label.to_numpy()
    scor = make_scorer(f1_score, labels=list(range(5)), average='macro', zero_division=0)
    variants = {'baseline': [('scale',StandardScaler()),('norm',Normalizer())],
                'l2_only':[('norm',Normalizer())], 'scale_only':[('scale',StandardScaler())], 'raw':[]}
    rows, choices = [], []
    for r in range(3):
        oof = {name:np.zeros(len(y),int) for name in variants}
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261009+r).split(X,y)):
            for name, steps in variants.items():
                model = GridSearchCV(Pipeline(steps+[('lr',LogisticRegression(max_iter=2000))]),
                    {'lr__C':[0.01,0.1,1,10,100]},
                    cv=StratifiedKFold(3,shuffle=True,random_state=7),scoring=scor,n_jobs=1)
                pred = model.fit(X[a],y[a]).predict(X[b])
                oof[name][b] = pred
                rows.append(dict(repeat=r,fold=fold,model=name,score=f1_score(y[b],pred,labels=range(5),average='macro',zero_division=0)))
                choices.append(dict(repeat=r,fold=fold,model=name,C=model.best_params_['lr__C']))
            print(f'p5 repeat={r} fold={fold}',flush=True)
        np.savez_compressed(OUT / f'p5_oof_{r}.npz',y=y,**oof)
    pd.DataFrame(choices).to_csv(OUT/'p5_inner_choices.csv',index=False)
    summarize(rows, 'p5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('problem',choices=['p1','p1_confirm','p5'])
    args = parser.parse_args()
    start = time.perf_counter()
    {'p1':p1,'p1_confirm':lambda: p1(confirm=True),'p5':p5}[args.problem]()
    print(f'elapsed_seconds={time.perf_counter()-start:.2f}',flush=True)
