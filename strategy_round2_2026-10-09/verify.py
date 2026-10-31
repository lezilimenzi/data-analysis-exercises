"""Recompute all reported results and check split/input provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PREV=ROOT/'strategy_validation_2026-10-09'


def close(a,b):
    assert np.allclose(a,b,atol=1e-12,rtol=0), (a,b)


def mf(y,p,n):
    return float(f1_score(y,p,labels=range(n),average='macro',zero_division=0))


def main():
    checked=[];changes={}
    for tag,n,reps,foldn,seed in [('p1',2,10,5,42),('p4',5,10,5,0),('p5',5,3,5,20261009),('p6',2,3,5,20261009),('p7',2,1,14,None)]:
        df=pd.read_csv(OUT/f'{tag}_pairs.csv');changes[tag]=[]
        for r in range(reps):
            z=np.load(OUT/f'{tag}_oof_{r}.npz');y=z['y'];folds=z['folds']
            if tag=='p7':
                split=[(np.flatnonzero(z['groups']!=g),np.flatnonzero(z['groups']==g)) for g in pd.unique(z['groups'])]
                old=np.load(PREV/'p7_oof.npz')
            else:
                if tag=='p6':split=list(StratifiedGroupKFold(foldn,shuffle=True,random_state=seed+r).split(np.zeros(len(y)),y,z['groups']))
                else:split=list(StratifiedKFold(foldn,shuffle=True,random_state=seed+r).split(np.zeros(len(y)),y))
                old=np.load(PREV/f'{tag}_oof_{r}.npz')
            assert np.array_equal(y,old['y'])
            close(z['baseline'],old['impute_first' if tag=='p1' else 'baseline'])
            for k,(a,b) in enumerate(split):
                assert np.all(folds[b]==k)
                if tag in ['p6','p7']:assert not set(z['groups'][a])&set(z['groups'][b])
                if tag!='p1':
                    for name in ['baseline','candidate']:
                        close(mf(y[b],z[name][b],n),df[(df.repeat==r)&(df.fold==k)&(df.model==name)].score.item())
            for name in ['baseline','candidate']:
                assert len(z[name])==len(y) and np.isfinite(z[name]).all()
                if tag=='p1':close(f1_score(y,z[name]>=.5,zero_division=0),df[(df.repeat==r)&(df.model==name)].score.item())
                else:assert np.isin(z[name],range(n)).all()
            bp=z['baseline']>=.5 if tag=='p1' else z['baseline'];cp=z['candidate']>=.5 if tag=='p1' else z['candidate']
            change=dict(repeat=r,fixes=int(((bp!=y)&(cp==y)).sum()),breaks=int(((bp==y)&(cp!=y)).sum()),
                baseline_class_f1=f1_score(y,bp,labels=range(n),average=None,zero_division=0).tolist(),
                candidate_class_f1=f1_score(y,cp,labels=range(n),average=None,zero_division=0).tolist())
            if n==2:
                change.update(baseline_fn=int(((bp==0)&(y==1)).sum()),candidate_fn=int(((cp==0)&(y==1)).sum()),
                    baseline_fp=int(((bp==1)&(y==0)).sum()),candidate_fp=int(((cp==1)&(y==0)).sum()))
            changes[tag].append(change)
        checked.append(tag+' scores, baseline identity, folds')
    df=pd.read_csv(OUT/'p2_pairs.csv');z=np.load(OUT/'p2_oof.npz');old=np.load(PREV/'p2_oof.npz');y=z['y']
    assert np.array_equal(y,old['y'])
    for k,(_,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=0).split(np.zeros(len(y)),y)):
        assert np.all(z['folds'][b]==k)
    for r in range(3):
        close(z[f'baseline_{r}'],old[f'gap_s{r}'])
        close(z[f'candidate_{r}'],(old[f'gap_s{r}'].astype(float)+old[f'grid_s{r}'].astype(float))/2)
        for name in ['baseline','candidate']:
            p=z[f'{name}_{r}'];assert np.allclose(p.sum(1),1,atol=2e-6)
            for k in range(3):
                b=z['folds']==k
                close(mf(y[b],p[b].argmax(1),7),df[(df.repeat==r)&(df.fold==k)&(df.model==name)].score.item())
    s=json.loads((OUT/'p2_summary.json').read_text())
    bp=np.mean([z[f'baseline_{r}'] for r in range(3)],axis=0).argmax(1);cp=np.mean([z[f'candidate_{r}'] for r in range(3)],axis=0).argmax(1)
    close(mf(y,bp,7),s['ensemble_baseline']);close(mf(y,cp,7),s['ensemble_candidate'])
    assert s['fixes']==int(((bp!=y)&(cp==y)).sum()) and s['breaks']==int(((bp==y)&(cp!=y)).sum())
    checked.append('p2 probabilities, folds and ensemble')
    df=pd.read_csv(OUT/'p3_pairs.csv');z=np.load(OUT/'p3_predictions.npz');old=np.load(PREV/'p3_predictions.npz')
    close(z['y'],old['y']);close(z['pred'][:,0],old['pred'][:,1])
    assert np.isfinite(z['pred']).all() and z['pred'].shape==(359,2,72)
    for j,name in enumerate(['baseline','candidate']):close(abs(z['pred'][:,j]-z['y']).mean(1),df[df.model==name].score.to_numpy())
    for entry in json.loads((OUT/'p3_time_boundaries.json').read_text()):assert pd.Timestamp(entry['last_training_target'])<=pd.Timestamp(entry['origin'])
    checked.append('p3 predictions, baseline and training target time limits')
    for tag in [f'p{i}' for i in range(1,8)]:
        df=pd.read_csv(OUT/f'{tag}_pairs.csv');s=json.loads((OUT/f'{tag}_summary.json').read_text())
        table=df.pivot(index=['repeat','fold'],columns='model',values='score');d=(table.candidate-table.baseline).to_numpy()*(-1 if tag=='p3' else 1)
        close(table.baseline.mean(),s['baseline']);close(table.candidate.mean(),s['candidate']);close(d.mean(),s['mean_d']);close(d.std(),s['sd_d'])
        assert s['wins']==int((d>1e-9).sum()) and s['losses']==int((d<-1e-9).sum()) and s['ties']==int((abs(d)<=1e-9).sum()) and s['pairs']==len(d)
        passed=d.mean()>0 and (d>1e-9).sum()>(d<-1e-9).sum()
        if tag=='p2':passed=d.mean()>=.003 and (d>1e-9).sum()>=6 and s['ensemble_candidate']>=s['ensemble_baseline']
        assert s['passes']==bool(passed)
        if tag=='p1':assert s['strict_8_of_10']==bool(d.mean()>0 and (d>1e-9).sum()>=8)
    z=np.load(OUT/'p8_predictions.npz');df=pd.read_csv(OUT/'p8_stress.csv');s=json.loads((OUT/'p8_summary.json').read_text())
    for row in df.itertuples():
        p=z[f'{row.model}_{row.duration_samples}']
        close(mf(z['y'],p,4),row.score);assert int((p!=z['y']).sum())==row.errors
    assert len(z['injected_indices'])==132 and len(np.unique(z['injected_indices']))==132
    assert z['injected_starts'].min()>=0 and z['injected_starts'].max()<20
    close(df.score.to_numpy(),[r['score'] for r in s['results']])
    train=pd.read_csv(ROOT/'problem_08/data/train.csv');sensors=np.load(ROOT/'problem_08/data/sensor_data.npz')
    mapping={v:i for i,v in enumerate(sensors['cycle_id'])}
    pressure=sensors['pressure'][train.cycle_id.map(mapping).to_numpy(),900:1200].astype(float)
    fit=train.split.eq('fit').to_numpy();labels=train.label.to_numpy()
    assert np.array_equal(labels[~fit],z['y'])
    for duration in range(4):
        noisy=pressure[~fit].copy()
        for idx,start in zip(z['injected_indices'],z['injected_starts']):
            noisy[idx,start:start+duration]=np.maximum(noisy[idx,start:start+duration],25.)
        for name,length in [('baseline',1),('candidate',3)]:
            crossing=lambda a:np.lib.stride_tricks.sliding_window_view(a>20,length,axis=1).all(-1).argmax(1)+900
            tr=crossing(pressure[fit]);rr=[(tr[labels[fit]==k].min(),tr[labels[fit]==k].max()) for k in range(4)]
            bounds=[(rr[k][1]+rr[k+1][0])/2 for k in range(3)]
            close(bounds,s['bounds'][name])
            assert np.array_equal(np.digitize(crossing(noisy),bounds),z[f'{name}_{duration}'])
    checked.append('p8 clean/stress scores and injection coverage')
    hashes=json.loads((OUT/'manifest.json').read_text())
    for name,h in hashes.items():
        if name in ['STRATEGY_full_2026-10-05.md','strategy_round2_2026-10-09/run.py']:continue
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    for name,h in json.loads((OUT/'environment.json').read_text())['supporting_files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    original_code=hashes['strategy_round2_2026-10-09/run.py']
    resumed_code=json.loads((OUT/'resume_2026-10-10.json').read_text())['code_sha256']
    assert hashlib.sha256((OUT/'run.py').read_bytes()).hexdigest()==resumed_code
    for i in range(1,9):
        ex=json.loads((OUT/f'p{i}_execution.json').read_text())
        assert ex['code_sha256']==(original_code if i==2 else resumed_code)
        assert ex['protocol_sha256']==hashes['strategy_round2_2026-10-09/PROTOCOL.md']
    result=dict(checks=checked+['all summary statistics/adoption flags','input/cache/protocol hashes unchanged','execution code provenance'],
        repeated_prediction_changes=changes,
        final_script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified':True,'checks':result['checks']},indent=2))


if __name__=='__main__':main()
