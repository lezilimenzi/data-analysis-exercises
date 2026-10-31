"""Recompute reported scores from saved OOF without training or test data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from tabular import ROOT, OUT, mf


def close(a,b):
    assert np.allclose(a,b,atol=1e-12,rtol=0), (a,b)


def score_rows(tag):
    df=pd.read_csv(OUT/f'{tag}_pairs.csv')
    if tag=='p2':df=df.rename(columns={'seed_base':'repeat'})
    base_name='gap' if tag=='p2' else 'baseline'
    base=df[df.model==base_name].set_index(['repeat','fold']).score
    stored=json.loads((OUT/f'{tag}_summary.json').read_text())
    if tag=='p2':stored=stored['results']
    for name,g in df.groupby('model'):
        d=(g.set_index(['repeat','fold']).score-base).to_numpy() * (-1 if tag=='p3' else 1)
        z=stored[name]
        close(g.score.mean(),z['mean_score' if tag=='p2' else 'score'])
        close(d.mean(),z['mean_d']);close(d.std(),z['sd_d'])
        assert int((d>1e-9).sum())==z['wins'] and int((d<-1e-9).sum())==z['losses']
        if tag=='p2':
            passes=d.mean()>=.003 and (d>1e-9).sum()>=6 and z['ensemble_score']>=stored['gap']['ensemble_score']
        else:
            passes=d.mean()>0 and (d>1e-9).sum()>(d<-1e-9).sum()
        assert bool(passes)==z['passes']
    return df


def classification(tag,repeats,folds,seed0,classes,groups=None):
    df=score_rows(tag)
    for r in range(repeats):
        with np.load(OUT/f'{tag}_oof_{r}.npz',allow_pickle=False) as z:
            y=z['y']
            if groups is None:
                split=list(StratifiedKFold(folds,shuffle=True,random_state=seed0+r).split(np.zeros(len(y)),y))
            else:
                split=list(StratifiedGroupKFold(folds,shuffle=True,random_state=seed0+r).split(np.zeros(len(y)),y,groups))
                for a,b in split:assert not set(groups[a])&set(groups[b])
            for name in z.files:
                if name=='y':continue
                p=z[name]
                assert len(p)==len(y) and np.isfinite(p).all()
                rows=df[(df.repeat==r)&(df.model==name)].sort_values('fold')
                if tag=='p1':close(float(f1_score(y,p>=.5,zero_division=0)),rows.score.iloc[0])
                else:
                    assert np.isin(p,range(classes)).all()
                    close([mf(y[b],p[b],classes) for a,b in split],rows.score.to_numpy())


def main():
    classification('p1',10,5,42,2)
    classification('p4',10,5,0,5)
    grouped=pd.read_csv(OUT/'p4_group_scores.csv')
    for r in range(10):
        with np.load(OUT/f'p4_group_oof_{r}.npz',allow_pickle=False) as z:
            split=StratifiedGroupKFold(5,shuffle=True,random_state=r).split(np.zeros(len(z['y'])),z['y'],z['groups'])
            for k,(a,b) in enumerate(split):
                assert not set(z['groups'][a])&set(z['groups'][b])
                close(mf(z['y'][b],z['pred'][b],5),grouped[(grouped.repeat==r)&(grouped.fold==k)].score.iloc[0])
    classification('p5',3,5,20261009,5)
    import re
    messages=pd.read_csv(ROOT/'problem_06/data/train.csv').message
    groups=messages.map(lambda s:re.sub(r'\s+',' ',re.sub(r'\d+','<NUM>',s.lower())).strip()).to_numpy()
    classification('p6',3,5,20261009,2,groups)
    df=score_rows('p7')
    with np.load(OUT/'p7_oof.npz',allow_pickle=False) as z:
        for row in df.itertuples():
            b=z['sessions']==row.session
            close(mf(z['y'][b],z[row.model][b],2),row.score)
    df=score_rows('p3')
    with np.load(OUT/'p3_predictions.npz',allow_pickle=False) as z:
        assert z['pred'].shape==(359,2,72) and z['y'].shape==(359,72)
        assert np.isfinite(z['pred']).all()
        for i,name in enumerate(['baseline','smooth']):
            close(np.abs(z['pred'][:,i]-z['y']).mean(1),df[df.model==name].score.to_numpy())
    df=score_rows('p2')
    p2=json.loads((OUT/'p2_summary.json').read_text())
    with np.load(OUT/'p2_oof.npz',allow_pickle=False) as z:
        y=z['y'];split=list(StratifiedKFold(3,shuffle=True,random_state=0).split(np.zeros(len(y)),y))
        base_pred=np.mean([z[f'gap_s{r}'] for r in range(3)],axis=0).argmax(1)
        for row in df.itertuples():
            p=z[f'{row.model}_s{row.repeat}']
            assert np.isfinite(p).all() and p.shape==(4900,7)
            assert np.allclose(p.sum(1),1,atol=2e-6)
            b=split[row.fold][1]
            close(mf(y[b],p[b].argmax(1),7),row.score)
        for name,result in p2['results'].items():
            pred=np.mean([z[f'{name}_s{r}'] for r in range(3)],axis=0).argmax(1)
            close(mf(y,pred,7),result['ensemble_score'])
            assert int(((base_pred!=y)&(pred==y)).sum())==result['fixes']
            assert int(((base_pred==y)&(pred!=y)).sum())==result['breaks']
            assert int((pred!=y).sum())==int((base_pred!=y).sum())-result['fixes']+result['breaks']
    assert len(list(OUT.glob('p2_*_s?_f?.pt')))==18
    # Report both conditional TTA effects; all candidates were fixed before results.
    factorial={}
    table=df.pivot(index=['repeat','fold'],columns='model',values='score')
    for a,b in [('gap','gap_tta'),('grid','grid_tta'),('gap','grid'),('gap_tta','grid_tta')]:
        d=(table[b]-table[a]).to_numpy()
        factorial[f'{b} minus {a}']={'mean_d':float(d.mean()),'sd_d':float(d.std()),
            'wins':int((d>1e-9).sum()),'losses':int((d<-1e-9).sum()),'ties':int((np.abs(d)<=1e-9).sum())}
    result={'all_saved_scores_recomputed':True,'p2_conditional_effects':factorial,
            'final_script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.py'))}}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
