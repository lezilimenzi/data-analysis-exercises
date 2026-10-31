"""Related-sequence grouping sensitivity only, not a model improvement comparison."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from tabular import OUT, mf, p4_inputs, save


def main():
    x,y,ids,s=p4_inputs()
    keys=[tuple(sorted(tuple(row) for row in seq)) for seq in s]
    codes={key:i for i,key in enumerate(dict.fromkeys(keys))}
    groups=np.array([codes[key] for key in keys])
    assert len(set(groups))==118
    pairs=[ids[groups==g].tolist() for g in np.unique(groups) if (groups==g).sum()>1]
    rows, repeats, leakage=[],[],[]
    for r in range(10):
        pred=np.zeros(len(y),int)
        original=StratifiedKFold(5,shuffle=True,random_state=r)
        leakage.append(dict(repeat=r,overlapping_groups_per_fold=[len(set(groups[a])&set(groups[b])) for a,b in original.split(x,y)]))
        for k,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=r).split(x,y,groups)):
            assert not set(groups[a])&set(groups[b])
            model=ExtraTreesClassifier(n_estimators=500,class_weight='balanced',random_state=42,n_jobs=1)
            p=model.fit(x[a],y[a]).predict_proba(x[b])
            pred[b]=(p*np.array([1,1.4,1,1,1])).argmax(1)
            rows.append(dict(repeat=r,fold=k,score=mf(y[b],pred[b],5),n=len(b)))
        repeats.append(dict(repeat=r,pooled=mf(y,pred,5)))
        np.savez_compressed(OUT/f'p4_group_oof_{r}.npz',y=y,pred=pred,groups=groups,ids=ids.astype(str))
        print('P4 group repeat',r,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'p4_group_scores.csv',index=False)
    original=pd.read_csv(OUT/'p4_pairs.csv').query("model=='baseline'")
    save('p4_group_diagnostics',dict(groups=len(set(groups)),related_pairs=pairs,
        original_fold_mean=float(original.score.mean()),grouped_fold_mean=float(df.score.mean()),
        grouped_repeat_pooled=repeats,original_group_overlap=leakage,
        interpretation='Different outer partitions: diagnostic sensitivity, not paired model improvement; no model selected.'))


if __name__=='__main__':
    start=time.perf_counter();main();print(f'elapsed_seconds={time.perf_counter()-start:.2f}',flush=True)
