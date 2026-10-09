"""사후 검증 (2026-10-05): fine-tuning 학습률 x0.5/x2, 입력 128, 5-fold 앙상블 재평가.
test 점수를 본 뒤의 실험이므로 제출 모델 변경에는 쓰지 않는다. 학습 코드는 problem.ipynb 3단계와 동일."""
import time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from feature_helper import extract_features

T0 = time.time(); SEED = 42; LABELS = [0, 1, 2, 3, 4]
torch.set_num_threads(8)
tr = pd.read_csv('data/train.csv', encoding='utf-8-sig'); y = tr.label.values
def load(p):
    with Image.open(p) as im: return np.asarray(im.convert('RGB'))
Xtr_img = np.stack([load(p) for p in tr.image_path])
F_tr = extract_features(tr.image_path, cache_path='cache/feat_orig_train.npz', verbose=False)
mf1 = lambda a, b: f1_score(a, b, labels=LABELS, average='macro', zero_division=0)
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
Xt_all = (torch.tensor(Xtr_img).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
EP = 12
def log(*a): print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

def aug(x):
    k = np.random.randint(4); x = torch.rot90(x, k, (2, 3))
    return torch.flip(x, (3,)) if np.random.rand() < .5 else x
def tta(m, X, res):
    m.eval(); out = 0
    with torch.no_grad():
        for k in range(4):
            for f in (False, True):
                x = torch.rot90(X, k, (2, 3)); x = torch.flip(x, (3,)) if f else x
                if res != 64: x = nn.functional.interpolate(x, size=res, mode='bilinear')
                out = out + torch.softmax(m(x), 1)
    return (out / 8).numpy()
def train_ft(idx, val, lr, res):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = torchvision.models.resnet18(weights='IMAGENET1K_V1'); m.fc = nn.Linear(512, 5)
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    X = Xt_all[idx]; Y = torch.tensor(y[idx]); hist = []
    for ep in range(EP):
        m.train(); perm = torch.randperm(len(idx))
        for s in range(0, len(idx), 32):
            b = perm[s:s+32]; x = aug(X[b])
            if res != 64: x = nn.functional.interpolate(x, size=res, mode='bilinear')
            opt.zero_grad(); nn.functional.cross_entropy(m(x), Y[b]).backward(); opt.step()
        sch.step()
        if ep in (3, 7, EP-1): hist.append((ep+1, round(mf1(y[val], tta(m, Xt_all[val], res).argmax(1)), 4)))
    return tta(m, Xt_all[val], res), hist

# 0. 재현 확인: 노트북 3-fold의 fold 1 (기존 값 0.976)
cv3 = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
a, b = cv3[0]; P, h = train_ft(a, b, 3e-4, 64)
log('reproduce 3-fold fold1:', round(mf1(y[b], P.argmax(1)), 4), '(notebook 0.976)', h)

# 1. 5-fold
cv5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
CFG = {'ft64_lr3e-4': (3e-4, 64), 'ft64_lr1.5e-4': (1.5e-4, 64), 'ft64_lr6e-4': (6e-4, 64), 'ft128_lr3e-4': (3e-4, 128)}
P_lr = np.zeros((len(y), 5)); folds = {'b1': []}
for i, (a, b) in enumerate(cv5):
    m = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                     {'logisticregression__C': [0.01, 0.1, 1]}, cv=3, scoring='f1_macro', n_jobs=8).fit(F_tr[a], y[a])
    P_lr[b] = m.predict_proba(F_tr[b]); folds['b1'].append(mf1(y[b], P_lr[b].argmax(1)))
log('b1 folds', np.round(folds['b1'], 4))
P_ft = {k: np.zeros((len(y), 5)) for k in CFG}; hists = {k: [] for k in CFG}
for k, (lr, res) in CFG.items():
    folds[k] = []; folds['ens_' + k] = []
    for i, (a, b) in enumerate(cv5):
        P, h = train_ft(a, b, lr, res); P_ft[k][b] = P; hists[k].append(h)
        folds[k].append(mf1(y[b], P.argmax(1))); folds['ens_' + k].append(mf1(y[b], ((P + P_lr[b]) / 2).argmax(1)))
        log(k, 'fold', i, round(folds[k][-1], 4), 'ens', round(folds['ens_' + k][-1], 4), h)
    np.savez('cache/posthoc_oof.npz', y=y, P_lr=P_lr, **{kk: v for kk, v in P_ft.items()})

pooled = {'b1': mf1(y, P_lr.argmax(1))}
for k in CFG:
    pooled[k] = mf1(y, P_ft[k].argmax(1)); pooled['ens_' + k] = mf1(y, ((P_ft[k] + P_lr) / 2).argmax(1))
perclass = {'b1': f1_score(y, P_lr.argmax(1), labels=LABELS, average=None).round(3).tolist()}
for k in CFG:
    perclass[k] = f1_score(y, P_ft[k].argmax(1), labels=LABELS, average=None).round(3).tolist()
    perclass['ens_' + k] = f1_score(y, ((P_ft[k] + P_lr) / 2).argmax(1), labels=LABELS, average=None).round(3).tolist()
def cmp(x, ref):
    d = np.array(folds[x]) - np.array(folds[ref])
    return dict(cand=x, ref=ref, mean_delta=round(d.mean(), 4), wins=f'{(d > 0).sum()}/5', ties=int((d == 0).sum()),
                consistent=bool(d.mean() >= 0.005 and (d > 0).sum() >= 4))
comps = [cmp('ens_ft64_lr3e-4', 'b1'), cmp('ens_ft64_lr3e-4', 'ft64_lr3e-4'), cmp('ft64_lr3e-4', 'b1'),
         cmp('ft64_lr1.5e-4', 'ft64_lr3e-4'), cmp('ft64_lr6e-4', 'ft64_lr3e-4'), cmp('ft128_lr3e-4', 'ft64_lr3e-4'),
         cmp('ens_ft64_lr1.5e-4', 'ens_ft64_lr3e-4'), cmp('ens_ft64_lr6e-4', 'ens_ft64_lr3e-4'), cmp('ens_ft128_lr3e-4', 'ens_ft64_lr3e-4')]
out = dict(folds={k: np.round(v, 4).tolist() for k, v in folds.items()}, pooled={k: round(v, 4) for k, v in pooled.items()},
           perclass=perclass, comparisons=comps, hists=hists, elapsed_s=round(time.time() - T0))
json.dump(out, open('exp_posthoc_results.json', 'w'), ensure_ascii=False, indent=1)
log('pooled', out['pooled']); [log(c) for c in comps]
