"""사후 검증 (2026-10-05): fine-tuning seed 변동 측정. 5-fold의 fold 0에서 FT64 lr 3e-4를 seed 0,1,2로 학습.
seed 42 결과는 exp_posthoc의 OOF 확률을 재사용한다. 학습 코드는 problem.ipynb 3단계와 동일(seed만 인자화)."""
import time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

T0 = time.time(); LABELS = [0, 1, 2, 3, 4]; EP = 12; LR = 3e-4
torch.set_num_threads(8)
tr = pd.read_csv('data/train.csv', encoding='utf-8-sig'); y = tr.label.values
def load(p):
    with Image.open(p) as im: return np.asarray(im.convert('RGB'))
Xtr_img = np.stack([load(p) for p in tr.image_path])
mf1 = lambda a, b: f1_score(a, b, labels=LABELS, average='macro', zero_division=0)
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
Xt_all = (torch.tensor(Xtr_img).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
def log(*a): print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

def aug(x):
    k = np.random.randint(4); x = torch.rot90(x, k, (2, 3))
    return torch.flip(x, (3,)) if np.random.rand() < .5 else x
def tta(m, X):
    m.eval(); out = 0
    with torch.no_grad():
        for k in range(4):
            for f in (False, True):
                x = torch.rot90(X, k, (2, 3)); x = torch.flip(x, (3,)) if f else x
                out = out + torch.softmax(m(x), 1)
    return (out / 8).numpy()
def train_ft(idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = torchvision.models.resnet18(weights='IMAGENET1K_V1'); m.fc = nn.Linear(512, 5)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    X = Xt_all[idx]; Y = torch.tensor(y[idx])
    for ep in range(EP):
        m.train(); perm = torch.randperm(len(idx))
        for s in range(0, len(idx), 32):
            b = perm[s:s+32]; x = aug(X[b])
            opt.zero_grad(); nn.functional.cross_entropy(m(x), Y[b]).backward(); opt.step()
        sch.step()
    return m

a, b = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))[0]
oof = np.load('cache/posthoc_oof.npz'); P_lr = oof['P_lr'][b]
P = {42: oof['ft64_lr3e-4'][b]}
for s in (0, 1, 2):
    P[s] = tta(train_ft(a, s), Xt_all[b]); log('seed', s, 'done')
res = {}
for s, p in P.items():
    res[f'seed{s}'] = dict(ft=round(mf1(y[b], p.argmax(1)), 4), ens=round(mf1(y[b], ((p + P_lr) / 2).argmax(1)), 4),
                           river=round(f1_score(y[b], p.argmax(1), labels=LABELS, average=None)[4], 3))
ft = np.array([v['ft'] for v in res.values()]); en = np.array([v['ens'] for v in res.values()])
P_se = np.mean([P[s] for s in (0, 1, 2)], 0)  # seed 42 제외: 42는 이미 본 결과
summary = dict(per_seed=res, ft_mean=round(ft.mean(), 4), ft_sd=round(ft.std(ddof=1), 4), ft_range=round(ft.max() - ft.min(), 4),
               ens_mean=round(en.mean(), 4), ens_sd=round(en.std(ddof=1), 4), ens_range=round(en.max() - en.min(), 4),
               seed_ens3_ft=round(mf1(y[b], P_se.argmax(1)), 4), seed_ens3_plus_b1=round(mf1(y[b], ((P_se + P_lr) / 2).argmax(1)), 4),
               b1=round(mf1(y[b], P_lr.argmax(1)), 4),
               pred_disagree_vs_42={s: int((P[s].argmax(1) != P[42].argmax(1)).sum()) for s in (0, 1, 2)}, n_val=len(b),
               elapsed_s=round(time.time() - T0))
json.dump(summary, open('exp_seed_results.json', 'w'), indent=1)
log(json.dumps(summary, indent=1))
