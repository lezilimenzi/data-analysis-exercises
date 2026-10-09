import sys, numpy as np, pandas as pd, sklearn
sys.path.insert(0, "/home/akswosla/maicon/problem_01")
from train import add_derived, RAW_NUM, DERIVED, CAT_COL, TARGET
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
print("sklearn", sklearn.__version__)

df = add_derived(pd.read_csv("/home/akswosla/maicon/problem_01/train.csv"))
df["grade_code"] = df[CAT_COL].cat.codes
y = df[TARGET].values
FS = {"raw": RAW_NUM + ["grade_code"], "derived": RAW_NUM + ["grade_code"] + DERIVED}

def rf(s): return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1))
def gb(s): return make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=s))
def hgb(s): return HistGradientBoostingClassifier(random_state=s)

def oof(build, X, rep):
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
        p[va] = build(rep).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return p

def best_f1(p):
    ths = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, p >= t) for t in ths]
    i = int(np.argmax(f)); return f[i], ths[i]

REPS = range(5)
res = {}
for name, fs, builders in [
    ("RF raw", "raw", [rf]), ("RF derived", "derived", [rf]),
    ("GB derived", "derived", [gb]), ("RF+GB derived", "derived", [rf, gb]),
    ("HGB derived (기존 계열)", "derived", [hgb])]:
    X = df[FS[fs]]
    rows = []
    for r in REPS:
        p = np.mean([oof(b, X, r) for b in builders], axis=0)
        fb, t = best_f1(p)
        rows.append((f1_score(y, p >= 0.5), fb, t))
    a = np.array(rows); res[name] = a
    print(f"{name:24s} F1@0.5 {a[:,0].mean():.4f}±{a[:,0].std():.4f} | 선택용 bestF1 {a[:,1].mean():.4f}±{a[:,1].std():.4f} thr {a[:,2].mean():.2f}")

# paired diffs on best F1
for a_, b_ in [("RF raw","RF derived"),("RF derived","RF+GB derived"),("GB derived","RF+GB derived"),("HGB derived (기존 계열)","RF+GB derived")]:
    d = res[b_][:,1]-res[a_][:,1]
    print(f"paired {a_} -> {b_}: mean {d.mean():+.4f}, wins {int((d>0).sum())}/{len(d)}")

# single 75/25 holdout variability: same model, different split seeds
X = df[FS["derived"]]
for name, builders in [("RF derived", [rf]), ("RF+GB derived", [rf, gb])]:
    s = []
    for seed in range(20):
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        p = np.mean([b(0).fit(Xtr, ytr).predict_proba(Xva)[:, 1] for b in builders], axis=0)
        s.append(f1_score(yva, p >= 0.5))
    s = np.array(s)
    print(f"holdout25% {name}: F1@0.5 mean {s.mean():.4f} sd {s.std():.4f} min {s.min():.4f} max {s.max():.4f} (val 양성 {int(round(yva.sum()))})")
