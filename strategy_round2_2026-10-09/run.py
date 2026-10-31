"""Fixed follow-up experiments. Only student train inputs; see PROTOCOL.md."""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PREV = ROOT/'strategy_validation_2026-10-09'
sys.path.insert(0, str(PREV))
from tabular import p4_inputs


def mf(y, p, n):
    return float(f1_score(y, p, labels=range(n), average='macro', zero_division=0))


def save(name, value):
    (OUT/f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(name, json.dumps(value, ensure_ascii=False), flush=True)


def finish(tag, rows, detail=None):
    df = pd.DataFrame(rows)
    df.to_csv(OUT/f'{tag}_pairs.csv', index=False)
    t = df.pivot(index=['repeat','fold'], columns='model', values='score')
    d = (t.candidate-t.baseline).to_numpy(dtype=np.float64)*(-1 if tag=='p3' else 1)
    result = dict(baseline=float(t.baseline.mean()), candidate=float(t.candidate.mean()),
                  mean_d=float(d.mean()), sd_d=float(d.std()), wins=int((d>1e-9).sum()),
                  losses=int((d<-1e-9).sum()), ties=int((abs(d)<=1e-9).sum()), pairs=len(d),
                  passes=bool(d.mean()>0 and (d>1e-9).sum()>(d<-1e-9).sum()))
    if tag=='p1':result['strict_8_of_10']=bool(d.mean()>0 and (d>1e-9).sum()>=8)
    if detail:result.update(detail)
    if tag=='p2':result['passes']=bool(d.mean()>=.003 and (d>1e-9).sum()>=6 and result['ensemble_candidate']>=result['ensemble_baseline'])
    save(f'{tag}_summary', result)


def old_oof(tag, r, name, y):
    with np.load(PREV/f'{tag}_oof_{r}.npz') as z:
        assert np.array_equal(z['y'], y)
        return z[name].copy()


def p1():
    df = pd.read_csv(ROOT/'problem_01/train.csv')
    raw = df[['ambient_temp_c','process_temp_c','rotation_rpm','torque_nm','tool_usage_min']].copy()
    raw['grade'] = df.equipment_grade.map({'A':0,'B':1,'C':2})
    x, y = raw.to_numpy(float), df.inspection_required.to_numpy()
    derive = lambda a:np.column_stack([a,a[:,3]*a[:,2]*(2*np.pi/60),a[:,4]*a[:,3],a[:,1]-a[:,0]])
    rows=[]
    for r in range(10):
        base=old_oof('p1',r,'impute_first',y);pred=np.zeros(len(y));folds=np.full(len(y),-1)
        for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42+r).split(x,y)):
            scaler=StandardScaler().fit(x[a])
            imp=IterativeImputer(max_iter=10,random_state=7).fit(scaler.transform(x[a]))
            xa,xb=x[a].copy(),x[b].copy()
            for z in [xa,xb]:
                missing=np.isnan(z)
                filled=scaler.inverse_transform(imp.transform(scaler.transform(z)))
                z[missing]=filled[missing]
                assert np.isfinite(z).all()
            models=[RandomForestClassifier(n_estimators=300,random_state=42+r,n_jobs=1),GradientBoostingClassifier(random_state=42+r)]
            pred[b]=np.mean([m.fit(derive(xa),y[a]).predict_proba(derive(xb))[:,1] for m in models],axis=0)
            folds[b]=k
        for name,p in [('baseline',base),('candidate',pred)]:
            rows.append(dict(repeat=r,fold=-1,model=name,score=float(f1_score(y,p>=.5,zero_division=0))))
        np.savez_compressed(OUT/f'p1_oof_{r}.npz',y=y,folds=folds,baseline=base,candidate=pred)
        print('P1 repeat',r,rows[-2:],flush=True)
    finish('p1',rows)


def p2():
    z=np.load(PREV/'p2_oof.npz');y=z['y'];rows=[];out={'y':y}
    split=list(StratifiedKFold(3,shuffle=True,random_state=0).split(np.zeros(len(y)),y))
    folds=np.zeros(len(y),int)
    for k,(_,b) in enumerate(split):folds[b]=k
    for r in range(3):
        base=z[f'gap_s{r}'].astype(float);p=(base+z[f'grid_s{r}'].astype(float))/2
        for name,prob in [('baseline',base),('candidate',p)]:
            out[f'{name}_{r}']=prob
            for k,(_,b) in enumerate(split):rows.append(dict(repeat=r,fold=k,model=name,score=mf(y[b],prob[b].argmax(1),7)))
    bp=np.mean([out[f'baseline_{r}'] for r in range(3)],axis=0).argmax(1)
    cp=np.mean([out[f'candidate_{r}'] for r in range(3)],axis=0).argmax(1)
    np.savez_compressed(OUT/'p2_oof.npz',folds=folds,**out)
    finish('p2',rows,dict(ensemble_baseline=mf(y,bp,7),ensemble_candidate=mf(y,cp,7),
        fixes=int(((bp!=y)&(cp==y)).sum()),breaks=int(((bp==y)&(cp!=y)).sum()),
        baseline_errors=int((bp!=y).sum()),candidate_errors=int((cp!=y).sum())))


def p3():
    sys.path.insert(0,str(ROOT/'problem_03'))
    import forecast as F
    from exp_strategy import diagnostics
    train=pd.read_csv(ROOT/'problem_03/train.csv',parse_dates=['timestamp'])
    old=np.load(PREV/'p3_predictions.npz');rows=[];pred=[];bounds=[]
    for k,ot in enumerate(old['origins']):
        o=pd.Timestamp(str(ot));hist=train[train.timestamp<=o]
        X,R,x0,b0=F.residual_data(hist)
        # All eligible origin days are consecutive: no missing historical rows.
        N=len(hist)//24
        ok=np.isfinite(F.origin_features(hist,F.M1_BEST['k'])[:N-3]).all(1)&np.isfinite(F.m1_all(F.day_matrix(hist,'temperature_c'),**F.M1_BEST)[:N-3]).all(1)
        days=np.flatnonzero(ok)
        assert len(days)==len(X) and np.all(np.diff(days)==1) and days.max()+3<=N-1
        w=2.**(-(N-1-days)/180.);w/=w.mean()
        scaler=StandardScaler().fit(X,sample_weight=w)
        residual=Ridge(alpha=1.).fit(scaler.transform(X),R,sample_weight=w).predict(scaler.transform(x0[None]))[0]
        smooth=lambda r:np.convolve(np.pad(r,(1,1),mode='edge'),[.25,.5,.25],mode='valid')
        base=b0+smooth(F.m2_ridge(hist,**F.M2_BEST)-b0)
        assert np.allclose(base,old['pred'][k,1],atol=1e-10,rtol=0)
        cp=b0+smooth(residual);y=old['y'][k]
        actual=train.set_index('timestamp').temperature_c.loc[o+pd.Timedelta(hours=1):o+pd.Timedelta(hours=72)].to_numpy()
        assert np.array_equal(y,actual)
        for name,p in [('baseline',base),('candidate',cp)]:
            dtr,event=diagnostics(p,y);err=abs(p-y)
            rows.append(dict(repeat=0,fold=k,model=name,score=float(err.mean()),origin=str(o),month=o.month,
                h1_24=float(err[:24].mean()),h25_48=float(err[24:48].mean()),h49_72=float(err[48:].mean()),dtr_err=float(dtr),event_mae=float(event)))
        pred.append([base,cp]);bounds.append(dict(origin=str(o),last_training_target=str(hist.timestamp.iloc[(days.max()+4)*24-1])))
        if k%60==0:print('P3 origin',k,flush=True)
    np.savez_compressed(OUT/'p3_predictions.npz',y=old['y'],origins=old['origins'],pred=pred)
    save('p3_time_boundaries',bounds)
    df=pd.DataFrame(rows)
    finish('p3',rows,dict(monthly_mae=df.groupby(['model','month']).score.mean().unstack(0).to_dict(),
        diagnostics=df.groupby('model')[['h1_24','h25_48','h49_72','dtr_err','event_mae']].mean().to_dict('index')))


def p4():
    x,y,ids,s=p4_inputs();v=s-s.mean(1,keepdims=True)
    cov=np.einsum('ntc,ntd->ncd',v,v);den=np.sqrt(np.diagonal(cov,axis1=1,axis2=2))
    denom=den[:,:,None]*den[:,None,:]
    corr=np.divide(cov,denom,out=np.zeros_like(cov),where=denom>0)
    i,j=np.triu_indices(6,k=1);xx=np.hstack([x,corr[:,i,j]])
    assert xx.shape==(120,123) and np.isfinite(xx).all()
    rows=[]
    for r in range(10):
        base=old_oof('p4',r,'baseline',y);cp=np.zeros(len(y),int);folds=np.zeros(len(y),int)
        for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=r).split(x,y)):
            m=ExtraTreesClassifier(n_estimators=500,class_weight='balanced',random_state=42,n_jobs=1)
            cp[b]=(m.fit(xx[a],y[a]).predict_proba(xx[b])*[1,1.4,1,1,1]).argmax(1);folds[b]=k
            for name,p in [('baseline',base),('candidate',cp)]:rows.append(dict(repeat=r,fold=k,model=name,score=mf(y[b],p[b],5)))
        np.savez_compressed(OUT/f'p4_oof_{r}.npz',y=y,folds=folds,baseline=base,candidate=cp)
        print('P4 repeat',r,flush=True)
    finish('p4',rows)


def p5():
    folder=ROOT/'problem_05_codex';sys.path.insert(0,str(folder))
    from feature_helper import extract_features
    df=pd.read_csv(folder/'data/train.csv');y=df.label.to_numpy()
    x=extract_features([folder/p for p in df.image_path],model_path=folder/'models/image_encoder.onnx',cache_path=folder/'train_features.npz').astype(float)
    scorer=make_scorer(f1_score,labels=range(5),average='macro',zero_division=0)
    rows=[];choices=[]
    for r in range(3):
        base=old_oof('p5',r,'baseline',y);cp=np.zeros(len(y),int);folds=np.zeros(len(y),int)
        for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261009+r).split(x,y)):
            m=GridSearchCV(make_pipeline(StandardScaler(),PCA(n_components=128,svd_solver='randomized',random_state=7),Normalizer(),LogisticRegression(max_iter=2000)),
                {'logisticregression__C':[.01,.1,1,10,100]},cv=StratifiedKFold(3,shuffle=True,random_state=7),scoring=scorer,n_jobs=1)
            cp[b]=m.fit(x[a],y[a]).predict(x[b]);folds[b]=k
            choices.append(dict(repeat=r,fold=k,C=m.best_params_['logisticregression__C'],explained_variance=float(m.best_estimator_.named_steps.pca.explained_variance_ratio_.sum())))
            for name,p in [('baseline',base),('candidate',cp)]:rows.append(dict(repeat=r,fold=k,model=name,score=mf(y[b],p[b],5)))
            print('P5 fold',r,k,flush=True)
        np.savez_compressed(OUT/f'p5_oof_{r}.npz',y=y,folds=folds,baseline=base,candidate=cp)
    finish('p5',rows,dict(inner_choices=choices))


def p6():
    df=pd.read_csv(ROOT/'problem_06/data/train.csv');x=df.message.to_numpy();y=df.label.to_numpy()
    groups=np.array([re.sub(r'\s+',' ',re.sub(r'\d+','<NUM>',s.lower())).strip() for s in x])
    scorer=make_scorer(f1_score,labels=[0,1],average='macro',zero_division=0);rows=[];choices=[]
    for r in range(3):
        base=old_oof('p6',r,'baseline',y);cp=np.zeros(len(y),int);folds=np.zeros(len(y),int)
        for k,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=20261009+r).split(x,y,groups)):
            assert not set(groups[a])&set(groups[b])
            union=FeatureUnion([('wb',TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,sublinear_tf=True)),
                ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,sublinear_tf=True))])
            inner=list(StratifiedKFold(3,shuffle=True,random_state=7).split(x[a],y[a]))
            m=TunedThresholdClassifierCV(make_pipeline(union,Normalizer(),LinearSVC(C=1,class_weight='balanced',random_state=7)),
                scoring=scorer,thresholds=41,cv=inner,n_jobs=1)
            cp[b]=m.fit(x[a],y[a]).predict(x[b]);folds[b]=k
            choices.append(dict(repeat=r,fold=k,threshold=float(m.best_threshold_),inner_group_overlap=[len(set(groups[a[ia]])&set(groups[a[ib]])) for ia,ib in inner]))
            for name,p in [('baseline',base),('candidate',cp)]:rows.append(dict(repeat=r,fold=k,model=name,score=mf(y[b],p[b],2)))
            print('P6 fold',r,k,flush=True)
        np.savez_compressed(OUT/f'p6_oof_{r}.npz',y=y,folds=folds,groups=groups,baseline=base,candidate=cp)
    finish('p6',rows,dict(inner_choices=choices))


def p7():
    df=pd.read_csv(ROOT/'problem_07/data/train.csv');features=pd.read_csv(PREV/'p7_features.csv',index_col=0)
    assert np.array_equal(features.index.to_numpy(),df.sample_id.to_numpy())
    x=features.to_numpy()[:,:10];y=df.label.to_numpy();g=df.session_id.to_numpy();rows=[]
    old=np.load(PREV/'p7_oof.npz');assert np.array_equal(y,old['y']) and np.array_equal(g,old['sessions'])
    bp=np.zeros(len(y),int);cp=bp.copy();folds=bp.copy();weights=[]
    for k,s in enumerate(pd.unique(g)):
        a,b=g!=s,g==s
        _,counts=np.unique(g[a],return_counts=True)
        sizes=pd.Series(g[a]).value_counts();w=np.array([a.sum()/(len(sizes)*sizes[t]) for t in g[a]])
        assert abs(w.mean()-1)<1e-12
        weights.append(dict(session=s,min=float(w.min()),max=float(w.max())))
        for name,wt in [('baseline',np.ones(a.sum())),('candidate',w)]:
            sc=StandardScaler().fit(x[a],sample_weight=wt)
            m=LogisticRegression(C=1,max_iter=2000).fit(sc.transform(x[a]),y[a],sample_weight=wt)
            p=(m.predict_proba(sc.transform(x[b]))[:,1]>.5).astype(int)
            (bp if name=='baseline' else cp)[b]=p
            rows.append(dict(repeat=0,fold=k,session=s,model=name,score=mf(y[b],p,2),n=int(b.sum()),fn=int(((p==0)&(y[b]==1)).sum()),fp=int(((p==1)&(y[b]==0)).sum())))
        folds[b]=k
    assert np.array_equal(bp,old['baseline'])
    np.savez_compressed(OUT/'p7_oof_0.npz',y=y,folds=folds,groups=g.astype(str),baseline=bp,candidate=cp)
    finish('p7',rows,dict(pooled_baseline=mf(y,bp,2),pooled_candidate=mf(y,cp,2),training_weights=weights))


def p8():
    df=pd.read_csv(ROOT/'problem_08/data/train.csv');data=np.load(ROOT/'problem_08/data/sensor_data.npz')
    pos={v:i for i,v in enumerate(data['cycle_id'])};p=data['pressure'][df.cycle_id.map(pos).to_numpy(),900:1200].astype(float)
    y=df.label.to_numpy();fit=df.split.eq('fit').to_numpy();v=p[~fit];yv=y[~fit]
    def rise(a,run):
        mask=a>20
        if run==3:mask=mask[:,:-2]&mask[:,1:-1]&mask[:,2:]
        assert mask.any(1).all()
        return mask.argmax(1)+900
    bounds={}
    for name,run in [('baseline',1),('candidate',3)]:
        r=rise(p[fit],run);ranges=[(r[y[fit]==k].min(),r[y[fit]==k].max()) for k in range(4)]
        bounds[name]=[(float(ranges[k][1])+float(ranges[k+1][0]))/2 for k in range(3)]
    rng=np.random.default_rng(20261009);idx=rng.choice(len(v),size=int(.3*len(v)),replace=False);starts=rng.integers(0,20,size=len(idx))
    rows=[];out={'y':yv,'injected_indices':idx,'injected_starts':starts}
    for duration in [0,1,2,3]:
        noisy=v.copy()
        for j in range(duration):noisy[idx,starts+j]=np.maximum(noisy[idx,starts+j],25.)
        for name,run in [('baseline',1),('candidate',3)]:
            pred=np.digitize(rise(noisy,run),bounds[name]);out[f'{name}_{duration}']=pred
            rows.append(dict(duration_samples=duration,model=name,score=mf(yv,pred,4),errors=int((pred!=yv).sum())))
    np.savez_compressed(OUT/'p8_predictions.npz',**out)
    pd.DataFrame(rows).to_csv(OUT/'p8_stress.csv',index=False)
    save('p8_summary',dict(fit_n=int(fit.sum()),valid_n=len(yv),injected_n=len(idx),bounds=bounds,results=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('problem',choices=[f'p{i}' for i in range(1,9)]);args=parser.parse_args()
    start=time.perf_counter();globals()[args.problem]()
    save(args.problem+'_execution',dict(elapsed_seconds=time.perf_counter()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest()))
