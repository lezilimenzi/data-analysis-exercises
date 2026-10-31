"""P5 frozen center crop and P6 grouped threshold validation; see PROTOCOL.md."""
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from tabular import OUT, ROOT, fixes, mf, save, summary


def center_features(paths, model):
    import onnxruntime as ort
    from PIL import Image
    h = hashlib.sha256(b'p05-center75-floor-left-top-floor-size-rgb-bilinear224-imagenet-v1')
    h.update(model.read_bytes())
    for p in paths:
        h.update(hashlib.sha256(p.read_bytes()).digest())
    signature = h.hexdigest()
    path = OUT/'p5_crop_features.npz'
    if path.exists():
        with np.load(path) as z:
            assert z['signature'].item()==signature
            return z['features'].copy()
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    sess = ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    mean=np.array([.485,.456,.406],np.float32)[:,None,None]
    std=np.array([.229,.224,.225],np.float32)[:,None,None]
    result=[]
    for i,p in enumerate(paths):
        with Image.open(p) as im:
            im=im.convert('RGB'); w,h=im.size
            cw,ch=int(.75*w),int(.75*h); left,top=(w-cw)//2,(h-ch)//2
            crop=im.crop((left,top,left+cw,top+ch)).resize((224,224),Image.Resampling.BILINEAR)
            x=np.asarray(crop,dtype=np.float32).transpose(2,0,1)/255.
        v=sess.run(None,{sess.get_inputs()[0].name:np.ascontiguousarray(((x-mean)/std)[None])})[0].reshape(-1)
        assert v.shape==(1280,) and np.isfinite(v).all()
        result.append(v)
        if (i+1)%100==0:print('P5 crop',i+1,flush=True)
    x=np.stack(result).astype(np.float32)
    np.savez_compressed(path,signature=signature,features=x)
    return x


def p5():
    folder=ROOT/'problem_05_codex'
    sys.path.insert(0,str(folder))
    from feature_helper import extract_features
    df=pd.read_csv(folder/'data/train.csv');y=df.label.to_numpy()
    paths=[folder/p for p in df.image_path]
    model=folder/'models/image_encoder.onnx'
    # Helper verifies content signatures; existing verified cache is never mutated.
    full=extract_features(paths,model_path=model,cache_path=folder/'train_features.npz').astype(float)
    crop=center_features(paths,model).astype(float)
    variants={'baseline':full,'center_crop':np.hstack([full,crop])}
    scorer=make_scorer(f1_score,labels=range(5),average='macro',zero_division=0)
    rows,choices,changes=[],[],[]
    for r in range(3):
        oo={n:np.zeros(len(y),int) for n in variants}
        for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261009+r).split(full,y)):
            for name,x in variants.items():
                m=GridSearchCV(make_pipeline(StandardScaler(),Normalizer(),LogisticRegression(max_iter=2000)),
                    {'logisticregression__C':[.01,.1,1,10,100]},
                    cv=StratifiedKFold(3,shuffle=True,random_state=7),scoring=scorer,n_jobs=1)
                oo[name][b]=m.fit(x[a],y[a]).predict(x[b])
                rows.append(dict(repeat=r,fold=k,model=name,score=mf(y[b],oo[name][b],5)))
                choices.append(dict(repeat=r,fold=k,model=name,C=m.best_params_['logisticregression__C']))
            print('P5 done',r,k,flush=True)
        changes.append(dict(repeat=r,**fixes(y,oo['baseline'],oo['center_crop'])))
        np.savez_compressed(OUT/f'p5_oof_{r}.npz',y=y,**oo)
    old=pd.read_csv(ROOT/'strategy_review_2026-10-09/p5_pairs.csv').query("model=='baseline'").score.to_numpy()
    now=pd.DataFrame(rows).query("model=='baseline'").score.to_numpy()
    assert np.allclose(now,old,atol=1e-12,rtol=0)
    summary(rows,'p5')
    save('p5_diagnostics',dict(inner_choices=choices,changes=changes,baseline_max_abs_diff=float(np.max(np.abs(now-old)))))


def p6():
    df=pd.read_csv(ROOT/'problem_06/data/train.csv')
    x,y=df.message.to_numpy(),df.label.to_numpy()
    norm=lambda s:re.sub(r'\s+',' ',re.sub(r'\d+','<NUM>',s.lower())).strip()
    groups=np.array([norm(s) for s in x])
    scorer=make_scorer(f1_score,labels=[0,1],average='macro',zero_division=0)
    rows,choices,changes=[],[],[]
    for r in range(3):
        oo={n:np.zeros(len(y),int) for n in ['baseline','group_inner']}
        outer=StratifiedGroupKFold(5,shuffle=True,random_state=20261009+r)
        for k,(a,b) in enumerate(outer.split(x,y,groups)):
            assert not set(groups[a])&set(groups[b])
            inner={'baseline':list(StratifiedKFold(3,shuffle=True,random_state=7).split(x[a],y[a])),
                   'group_inner':list(StratifiedGroupKFold(3,shuffle=True,random_state=7).split(x[a],y[a],groups[a]))}
            for name,cv in inner.items():
                overlaps=[len(set(groups[a[ia]])&set(groups[a[ib]])) for ia,ib in cv]
                if name=='group_inner': assert sum(overlaps)==0
                pipe=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),sublinear_tf=True,min_df=2),
                                   LinearSVC(C=1,class_weight='balanced',random_state=7))
                m=TunedThresholdClassifierCV(pipe,scoring=scorer,thresholds=41,cv=cv,n_jobs=1)
                oo[name][b]=m.fit(x[a],y[a]).predict(x[b])
                rows.append(dict(repeat=r,fold=k,model=name,score=mf(y[b],oo[name][b],2)))
                choices.append(dict(repeat=r,fold=k,model=name,threshold=float(m.best_threshold_),
                                    inner_overlapping_groups=overlaps))
            print('P6 done',r,k,flush=True)
        changes.append(dict(repeat=r,**fixes(y,oo['baseline'],oo['group_inner']),
                            pooled={n:mf(y,p,2) for n,p in oo.items()}))
        np.savez_compressed(OUT/f'p6_oof_{r}.npz',y=y,**oo)
    summary(rows,'p6')
    save('p6_diagnostics',dict(groups=len(set(groups)),inner_choices=choices,changes=changes))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('problem',choices=['p5','p6'])
    args=parser.parse_args();start=time.perf_counter()
    globals()[args.problem]()
    print(f'elapsed_seconds={time.perf_counter()-start:.2f}',flush=True)
