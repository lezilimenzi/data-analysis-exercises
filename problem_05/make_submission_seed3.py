"""seed 3개(0,1,2) ResNet18 fine-tuning(입력 64) 확률 평균 + B1(특징 + LR) 확률을 1:1 평균해 test 예측.
학습 코드는 problem.ipynb와 동일. 기존 submission.csv는 건드리지 않고 submission_seed3.csv로 저장."""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from feature_helper import extract_features

T0 = time.time(); EP = 12; LR = 3e-4; SEEDS = (0, 1, 2)
torch.set_num_threads(8)
tr = pd.read_csv('data/train.csv', encoding='utf-8-sig'); te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
ss = pd.read_csv('data/sample_submission.csv', encoding='utf-8-sig'); y = tr.label.values
def load(p):
    with Image.open(p) as im: return np.asarray(im.convert('RGB'))
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
to_t = lambda X: (torch.tensor(X).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
Xtr = to_t(np.stack([load(p) for p in tr.image_path])); Xte = to_t(np.stack([load(p) for p in te.image_path]))
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
def train_ft(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = torchvision.models.resnet18(weights='IMAGENET1K_V1'); m.fc = nn.Linear(512, 5)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    Y = torch.tensor(y)
    for ep in range(EP):
        m.train(); perm = torch.randperm(len(y))
        for s in range(0, len(y), 32):
            b = perm[s:s+32]
            opt.zero_grad(); nn.functional.cross_entropy(m(aug(Xtr[b])), Y[b]).backward(); opt.step()
        sch.step()
    return m

F_tr = extract_features(tr.image_path, cache_path='cache/feat_orig_train.npz', verbose=False)
F_te = extract_features(te.image_path, cache_path='cache/feat_orig_test.npz', verbose=False)
b1 = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                  {'logisticregression__C': [0.01, 0.1, 1]}, cv=3, scoring='f1_macro', n_jobs=8).fit(F_tr, y)
log('B1 best C', b1.best_params_)
P_ft = []
for s in SEEDS:
    P_ft.append(tta(train_ft(s), Xte)); log('seed', s, 'done')
P_ft = np.mean(P_ft, 0)
pred = ((P_ft + b1.predict_proba(F_te)) / 2).argmax(1)
sub = pd.DataFrame({'image_id': te.image_id, 'label': pred.astype(int)})
sub.to_csv('submission_seed3.csv', index=False, encoding='utf-8-sig')

s = pd.read_csv('submission_seed3.csv', encoding='utf-8-sig')
assert list(s.columns) == ['image_id', 'label'] and len(s) == 250 and (s.image_id.values == ss.image_id.values).all()
assert s.label.dtype.kind == 'i' and s.label.between(0, 4).all()
old = pd.read_csv('submission.csv', encoding='utf-8-sig')
log('format OK. dist', s.label.value_counts().sort_index().to_dict())
log('rows differing from submission.csv:', int((old.label.values != s.label.values).sum()),
    old[old.label.values != s.label.values].assign(new=s.label[old.label.values != s.label.values].values).to_dict('records'))
