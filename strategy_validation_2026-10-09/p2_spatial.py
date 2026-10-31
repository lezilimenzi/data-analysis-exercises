"""Fixed P2 factorial comparison; train only. See PROTOCOL.md."""
import faulthandler
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / 'problem_02'))
from common import data, folds
from exp_cnn import make_model

torch.set_num_threads(8)


def score(y, p):
    return float(f1_score(y, p, labels=range(7), average='macro', zero_division=0))


def infer(model, x):
    with torch.no_grad():
        return torch.cat([model(b).softmax(1) for b in x.split(128)]).numpy()


def train(x, y, kind, seed):
    faulthandler.dump_traceback_later(180, repeat=True)
    torch.manual_seed(seed)
    model = make_model()
    if kind == 'grid':
        model[3] = nn.Identity()
        model[6] = nn.Linear(128 * 3 * 3, 7)
    xt, yt = torch.tensor(x).unsqueeze(1), torch.tensor(y)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.002, total_steps=20*((len(x)+127)//128))
    for ep in range(20):
        model.train()
        perm = torch.randperm(len(x))
        for ids in perm.split(128):
            opt.zero_grad()
            F.cross_entropy(model(xt[ids]), yt[ids]).backward()
            opt.step()
            sched.step()
        print(f'{kind} seed={seed} epoch={ep+1}', flush=True)
    faulthandler.cancel_dump_traceback_later()
    return model.eval()


def run():
    x, y = data()
    fl = folds(y)
    rows, all_probs, timing = [], {}, []
    t0 = time.perf_counter()
    for s in range(3):
        for kind in ['gap', 'grid']:
            probs = np.zeros((len(y), 7), dtype=np.float32)
            tta = probs.copy()
            for k, (a, b) in enumerate(fl):
                stem = OUT / f'p2_{kind}_s{s}_f{k}'
                result = stem.with_suffix('.npz')
                if result.exists():
                    with np.load(result) as z:
                        assert np.array_equal(z['valid'], b) and np.array_equal(z['y'], y[b])
                        probs[b], tta[b] = z['plain'], z['tta']
                    timing.append(json.loads(stem.with_suffix('.json').read_text()))
                    continue
                start = time.perf_counter()
                model = train(x[a], y[a], kind, k+10*s)
                train_seconds = time.perf_counter()-start
                torch.save(model.state_dict(), stem.with_suffix('.pt'))
                xt = torch.tensor(x[b]).unsqueeze(1)
                start = time.perf_counter()
                probs[b] = infer(model, xt)
                plain_seconds = time.perf_counter()-start
                start = time.perf_counter()
                padded = F.pad(xt, (1, 1, 1, 1))
                views = [probs[b]]
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    views.append(infer(model, padded[:, :, 1+dy:29+dy, 1+dx:29+dx]))
                tta[b] = np.mean(views, axis=0)
                meta = dict(seed_base=s, fold=k, model=kind, train_seconds=train_seconds,
                            plain_seconds=plain_seconds, tta_seconds=plain_seconds+time.perf_counter()-start,
                            parameters=sum(p.numel() for p in model.parameters()))
                np.savez_compressed(result, valid=b, y=y[b], plain=probs[b], tta=tta[b])
                stem.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
                timing.append(meta)
                print(f'DONE {kind} s{s} f{k}: plain={score(y[b], probs[b].argmax(1)):.6f} '
                      f'tta={score(y[b], tta[b].argmax(1)):.6f} train_s={train_seconds:.1f}', flush=True)
            for name, p in [(kind, probs), (kind+'_tta', tta)]:
                all_probs[f'{name}_s{s}'] = p
                for k, (_, b) in enumerate(fl):
                    rows.append(dict(seed_base=s, fold=k, model=name, score=score(y[b], p[b].argmax(1))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'p2_pairs.csv', index=False)
    np.savez_compressed(OUT/'p2_oof.npz', y=y, **all_probs)
    base = df[df.model=='gap'].set_index(['seed_base', 'fold']).score
    bp = np.mean([all_probs[f'gap_s{s}'] for s in range(3)], axis=0).argmax(1)
    summary = {}
    for name in ['gap', 'gap_tta', 'grid', 'grid_tta']:
        v = df[df.model==name].set_index(['seed_base', 'fold']).score
        d = (v-base).to_numpy()
        p = np.mean([all_probs[f'{name}_s{s}'] for s in range(3)], axis=0).argmax(1)
        cm = confusion_matrix(y, p, labels=range(7))
        summary[name] = dict(mean_score=float(v.mean()), mean_d=float(d.mean()), sd_d=float(d.std()),
            wins=int((d>1e-9).sum()), losses=int((d<-1e-9).sum()), ties=int((np.abs(d)<=1e-9).sum()),
            ensemble_score=score(y,p), ensemble_d=score(y,p)-score(y,bp),
            fixes=int(((bp!=y)&(p==y)).sum()), breaks=int(((bp==y)&(p!=y)).sum()),
            errors=int((p!=y).sum()), coat_pullover_errors=int(cm[2,6]+cm[6,2]),
            class_f1=f1_score(y,p,labels=range(7),average=None,zero_division=0).tolist(),
            confusion=cm.tolist(), passes=bool(d.mean()>=.003 and (d>1e-9).sum()>=6 and score(y,p)>=score(y,bp)))
    reference = {}
    for s, tag in enumerate(['C0', 'C0_s1', 'C0_s2']):
        old = np.load(ROOT/'problem_02/results'/f'oof_proba_cnn_{tag}.npy')
        new = all_probs[f'gap_s{s}']
        reference[tag] = dict(max_abs_probability_diff=float(np.max(np.abs(old-new))),
                             label_differences=int((old.argmax(1)!=new.argmax(1)).sum()),
                             score_diff=score(y,new.argmax(1))-score(y,old.argmax(1)))
    output = dict(results=summary, baseline_reproduction=reference, timing=timing,
                  torch_version=torch.__version__, threads=torch.get_num_threads(), elapsed_seconds=time.perf_counter()-t0)
    (OUT/'p2_summary.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    faulthandler.enable()
    run()
