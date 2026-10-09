"""Read-only signal analysis; write diagnostic artifacts only."""
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
protected = ['model.joblib', 'submission.csv']
before = {p: hashlib.sha256((ROOT/p).read_bytes()).digest() for p in protected}
s = pd.read_csv(ROOT/'train_series.csv')
y = pd.read_csv(ROOT/'train_labels.csv').set_index('sample_id').label
cols = list(s.columns[2:])
seq = {sid: g.sort_values('step')[cols].to_numpy() for sid,g in s.groupby('sample_id')}
groups = defaultdict(list)
for sid,x in seq.items():
    groups[tuple(sorted(map(tuple,x)))].append(sid)
rows = []
for ids in groups.values():
    if len(ids)>1:
        for sid in ids:
            rows.append({'group':ids[0], 'sample_id':sid, 'label':int(y[sid]), 'members':len(ids),
                         'different_labels':len(set(y[i] for i in ids))>1})
pd.DataFrame(rows, columns=['group','sample_id','label','members','different_labels']).to_csv(ROOT/'permutation_groups.csv',index=False)
print('ROW-MULTISET GROUPS',rows)
pairs=[('SEG-1061','SEG-1100'),('SEG-1057','SEG-1014'),('SEG-1080','SEG-1044')]
fig, axes=plt.subplots(6,3,figsize=(15,15),sharex=True)
metrics=[]
for j,(a,b) in enumerate(pairs):
    x,z=seq[a],seq[b]
    print('PAIR',a,b,'same row multiset', sorted(map(tuple,x))==sorted(map(tuple,z)))
    if sorted(map(tuple,x))==sorted(map(tuple,z)):
        remaining=list(range(15)); mapping=[]
        for row in z:
            k=next(k for k in remaining if np.array_equal(x[k],row)); mapping.append(k); remaining.remove(k)
        print('B row indices in A:',mapping)
    for i,c in enumerate(cols):
        ax=axes[i,j]; ax.plot(range(15),x[:,i],'-o',ms=3,label=f'{a} (class {y[a]})')
        ax.plot(range(15),z[:,i],'--x',ms=4,label=f'{b} (class {y[b]})')
        ax.set_ylabel(c); ax.grid(alpha=.25)
        if i==0: ax.legend(fontsize=8)
        if i==5: ax.set_xlabel('step')
        metrics.append({'pair':f'{a}/{b}','channel':c,
            'mae':float(abs(x[:,i]-z[:,i]).mean()),
            'mean_difference':float(x[:,i].mean()-z[:,i].mean()),
            'median_difference':float(np.median(x[:,i])-np.median(z[:,i])),
            'iqr_difference':float(np.diff(np.percentile(x[:,i],[25,75]))[0]-np.diff(np.percentile(z[:,i],[25,75]))[0])})
fig.tight_layout(); fig.savefig(ROOT/'waveform_pairs.png',dpi=130); plt.close(fig)
pd.DataFrame(metrics).to_csv(ROOT/'waveform_pair_metrics.csv',index=False)
for p in protected: assert hashlib.sha256((ROOT/p).read_bytes()).digest()==before[p]
print('Model and submission unchanged.')
