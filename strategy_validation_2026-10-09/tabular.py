"""P1/P3/P4/P7 predeclared follow-ups. Train inputs only; see PROTOCOL.md."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def mf(y, p, n):
    return float(f1_score(y, p, labels=range(n), average='macro', zero_division=0))


def summary(rows, tag, sign=1):
    df = pd.DataFrame(rows)
    df.to_csv(OUT/f'{tag}_pairs.csv', index=False)
    base = df[df.model=='baseline'].set_index(['repeat', 'fold']).score
    result = {}
    for name, g in df.groupby('model', sort=False):
        v = g.set_index(['repeat', 'fold']).score
        d = sign*(v-base).to_numpy()
        result[name] = dict(score=float(v.mean()), mean_d=float(d.mean()), sd_d=float(d.std()),
            wins=int((d>1e-9).sum()), losses=int((d<-1e-9).sum()), ties=int((np.abs(d)<=1e-9).sum()),
            pairs=len(d), passes=bool(d.mean()>0 and (d>1e-9).sum()>(d<-1e-9).sum()))
    save(f'{tag}_summary', result)
    return result


def save(name, obj):
    (OUT/f'{name}.json').write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    print(name, json.dumps(obj, ensure_ascii=False), flush=True)


def fixes(y, base, new):
    return dict(fixes=int(((base!=y)&(new==y)).sum()), breaks=int(((base==y)&(new!=y)).sum()))


def p1():
    df = pd.read_csv(ROOT/'problem_01/train.csv')
    raw = df[['ambient_temp_c','process_temp_c','rotation_rpm','torque_nm','tool_usage_min']].copy()
    raw['grade'] = df.equipment_grade.map({'A':0,'B':1,'C':2})
    x, y = raw.to_numpy(float), df.inspection_required.to_numpy()
    derive = lambda a: np.column_stack([a, a[:,3]*a[:,2]*(2*np.pi/60), a[:,4]*a[:,3], a[:,1]-a[:,0]])
    rows, changes = [], []
    for r in range(10):
        oo = {n: np.zeros(len(y)) for n in ['baseline', 'impute_first']}
        for k, (a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42+r).split(x,y)):
            im = SimpleImputer(strategy='median').fit(derive(x[a]))
            ir = SimpleImputer(strategy='median').fit(x[a])
            inputs = {'baseline':(im.transform(derive(x[a])), im.transform(derive(x[b]))),
                      'impute_first':(derive(ir.transform(x[a])), derive(ir.transform(x[b])))}
            for name, (xa, xb) in inputs.items():
                models = [RandomForestClassifier(n_estimators=300,random_state=42+r,n_jobs=1),
                          GradientBoostingClassifier(random_state=42+r)]
                oo[name][b] = np.mean([m.fit(xa,y[a]).predict_proba(xb)[:,1] for m in models], axis=0)
        for name, p in oo.items():
            rows.append(dict(repeat=r,fold=-1,model=name,score=float(f1_score(y,p>=.5,zero_division=0))))
        changes.append(dict(repeat=r, **fixes(y,oo['baseline']>=.5,oo['impute_first']>=.5)))
        np.savez_compressed(OUT/f'p1_oof_{r}.npz', y=y, **oo)
        pd.DataFrame(rows).to_csv(OUT/'p1_pairs.csv',index=False)
        print('P1 done repeat',r,flush=True)
    old = pd.read_csv(ROOT/'problem_01/posthoc_model_family_reps.csv').ens_der__f1_05.to_numpy()
    now = pd.DataFrame(rows).query("model=='baseline'").score.to_numpy()
    assert np.allclose(now,old,atol=1e-12,rtol=0), (now,old)
    summary(rows,'p1')
    save('p1_diagnostics',dict(baseline_max_abs_diff=float(np.max(np.abs(now-old))),repeated_prediction_changes=changes))


def p3():
    sys.path.insert(0,str(ROOT/'problem_03'))
    import forecast as F
    from exp_strategy import diagnostics
    train = pd.read_csv(ROOT/'problem_03/train.csv',parse_dates=['timestamp'])
    target = train.set_index('timestamp').temperature_c
    rows, preds, truths = [], [], []
    origins = pd.date_range('2014-01-01 23:00','2014-12-25 23:00',freq='D')
    for k,o in enumerate(origins):
        hist = train[train.timestamp<=o]
        y = target[o+pd.Timedelta(hours=1):o+pd.Timedelta(hours=72)].to_numpy()
        assert len(y)==72 and hist.timestamp.iloc[-1]==o
        base = F.m2_ridge(hist, **F.M2_BEST)
        b0 = F.m1(hist,**F.M1_BEST)
        smooth = b0+np.convolve(np.pad(base-b0,(1,1),mode='edge'),[.25,.5,.25],mode='valid')
        for name,p in [('baseline',base),('smooth',smooth)]:
            err = np.abs(p-y)
            dtr, event = diagnostics(p,y)
            rows.append(dict(repeat=0,fold=k,origin=str(o),month=o.month,model=name,score=float(err.mean()),
                h1_24=float(err[:24].mean()),h25_48=float(err[24:48].mean()),h49_72=float(err[48:].mean()),
                dtr_err=float(dtr), event_mae=float(event)))
        preds.append([base,smooth]); truths.append(y)
        if k%60==0: print('P3 origin',k,flush=True)
    old = pd.read_csv(ROOT/'problem_03/logs/strategy_pairs.csv').query("candidate=='base'").mae.to_numpy()
    now = pd.DataFrame(rows).query("model=='baseline'").score.to_numpy()
    assert np.allclose(now,old,atol=1e-10,rtol=0), np.max(np.abs(now-old))
    np.savez_compressed(OUT/'p3_predictions.npz',origins=origins.astype(str).to_numpy(dtype=str),y=truths,pred=preds)
    summary(rows,'p3',sign=-1)
    df = pd.DataFrame(rows)
    detail = {'baseline_max_abs_diff':float(np.max(np.abs(now-old))),
              'diagnostics':df.groupby('model')[['h1_24','h25_48','h49_72','dtr_err','event_mae']].mean().to_dict('index'),
              'monthly_mae':df.groupby(['model','month']).score.mean().reset_index().to_dict('records')}
    save('p3_diagnostics',detail)


def p4_inputs():
    df = pd.read_csv(ROOT/'problem_04/train_series.csv')
    labels = pd.read_csv(ROOT/'problem_04/train_labels.csv')
    cols = ['force_x','force_y','force_z','torque_x','torque_y','torque_z']
    seq = {k:g.sort_values('step')[cols].to_numpy(float) for k,g in df.groupby('sample_id')}
    s = np.stack([seq[k] for k in labels.sample_id]); y = labels.label.to_numpy()
    stats = np.hstack([s.mean(1),s.std(1),s.min(1),s.max(1),s[:,-1]-s[:,0],np.abs(np.diff(s,axis=1)).mean(1)])
    A = lambda m,n:s[:,m-1:n].mean(1)
    D = lambda m,n:(s[:,n-1]-s[:,m-1])/(n-m)
    trend = np.where(A(1,5)<A(11,15),1.,-1.)
    M = ((s[:,:-1]*trend[:,None,:])<(s[:,1:]*trend[:,None,:])).sum(1)
    x = np.hstack([stats]+[A(a,b) for a,b in [(1,3),(4,6),(7,9),(10,12),(13,15),(1,5),(6,10),(11,15),(1,15)]]+[D(1,8),D(8,15),M])
    assert x.shape==(120,108)
    return x, y, labels.sample_id.to_numpy(), s


def p4():
    x, y, _, _ = p4_inputs()
    model = lambda:ExtraTreesClassifier(n_estimators=500,class_weight='balanced',random_state=42,n_jobs=1)
    apply = lambda p,w:(p*np.array([1,w,1,1,1])).argmax(1)
    rows, choices, changes = [], [], []
    for r in range(10):
        oo = {n:np.zeros(len(y),int) for n in ['baseline','unweighted','inner_weight']}
        for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
            inner = np.zeros((len(a),5))
            for ia,ib in StratifiedKFold(3,shuffle=True,random_state=7).split(x[a],y[a]):
                inner[ib] = model().fit(x[a[ia]],y[a[ia]]).predict_proba(x[a[ib]])
            ws = [1.,1.2,1.4,1.6]
            scores = [mf(y[a],apply(inner,w),5) for w in ws]
            w = ws[int(np.argmax(scores))]
            p = model().fit(x[a],y[a]).predict_proba(x[b])
            choices.append(dict(repeat=r,fold=k,weight=w,inner_scores=scores))
            for name,wt in [('baseline',1.4),('unweighted',1.),('inner_weight',w)]:
                oo[name][b] = apply(p,wt)
                rows.append(dict(repeat=r,fold=k,model=name,score=mf(y[b],oo[name][b],5)))
        changes.append(dict(repeat=r,pooled={n:mf(y,p,5) for n,p in oo.items()},
                            changes={n:fixes(y,oo['baseline'],p) for n,p in oo.items()}))
        np.savez_compressed(OUT/f'p4_oof_{r}.npz',y=y,**oo)
        print('P4 done repeat',r,flush=True)
    summary(rows,'p4')
    save('p4_diagnostics',dict(inner_choices=choices,repeats=changes))


def p7():
    train = pd.read_csv(ROOT/'problem_07/data/train.csv')
    log = pd.read_csv(ROOT/'problem_07/data/link_log.csv')
    log = log[log.sample_id.isin(train.sample_id)].copy()
    log['snr_db'] = pd.to_numeric(log.snr_db,errors='coerce')
    features = {}
    for sid,g in log.groupby('sample_id'):
        g = g.sort_values('relative_second')
        r,dl = g.relative_second.to_numpy(),g.dl_kbps.to_numpy()
        assert len(np.unique(r))==len(r) and r[-1]==0 and r[0]>=-44
        d,low = np.log1p(dl),(dl<20000).astype(int)
        m5,m15 = r>=-4,r>=-14
        run = 0
        for i in range(len(r)-1,-1,-1):
            if not low[i] or (i<len(r)-1 and r[i+1]-r[i]!=1):break
            run+=1
        hi = r[low==0]
        snr_valid = np.isfinite(g.snr_db.to_numpy())
        snr_times = r[snr_valid]
        features[sid] = [d[m5].mean(),d[m15].mean(),d.mean(),low.mean(),d[m15].min(),d[m15].std(),
            np.polyfit(r[m15],d[m15],1)[0],run,-hi.max() if len(hi) else 45,low[m5].mean(),
            m5.sum()/5,m15.sum()/15,len(r)/45,np.max(np.diff(np.r_[-45,r,1])-1),
            (snr_valid&m15).sum()/15,-snr_times[-1] if len(snr_times) else 45]
    x = np.array([features[k] for k in train.sample_id],float)
    y,g = train.label.to_numpy(),train.session_id.to_numpy()
    assert np.isfinite(x).all() and x.shape==(len(train),16)
    oo = {n:np.zeros(len(y),int) for n in ['baseline','coverage']}
    rows = []
    for k,sid in enumerate(pd.unique(g)):
        a,b = g!=sid,g==sid
        for name,z in [('baseline',x[:,:10]),('coverage',x)]:
            m = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=2000))
            oo[name][b] = m.fit(z[a],y[a]).predict_proba(z[b])[:,1]>.5
            rows.append(dict(repeat=0,fold=k,session=sid,model=name,score=mf(y[b],oo[name][b],2),
                fn=int(((oo[name][b]==0)&(y[b]==1)).sum()),fp=int(((oo[name][b]==1)&(y[b]==0)).sum()),n=int(b.sum())))
    assert round(mf(y,oo['baseline'],2),4)==.8538
    np.savez_compressed(OUT/'p7_oof.npz',y=y,sessions=g.astype(str),**oo)
    pd.DataFrame(x,index=train.sample_id).to_csv(OUT/'p7_features.csv')
    summary(rows,'p7')
    save('p7_diagnostics',dict(pooled={n:mf(y,p,2) for n,p in oo.items()},
        changes=fixes(y,oo['baseline'],oo['coverage']),feature_std=x.std(0).tolist()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('problem',choices=['p1','p3','p4','p7'])
    args=parser.parse_args();start=time.perf_counter()
    globals()[args.problem]()
    print(f'elapsed_seconds={time.perf_counter()-start:.2f}',flush=True)
