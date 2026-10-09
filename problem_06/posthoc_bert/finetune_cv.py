# 사후 실험: 사전학습 언어모델 fine-tuning 5-fold × 1회 (seed 2026 첫 분할). 제출 파일은 바꾸지 않는다.
# 사용: python3 posthoc_bert/finetune_cv.py distilbert-base-uncased
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.metrics import f1_score, make_scorer

NAME = sys.argv[1]
EPOCHS, LR, BS, MAXLEN, SEED = 3, 5e-5, 32, 64, 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), NAME.replace("/", "_"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

tr = pd.read_csv("data/train.csv", encoding="utf-8-sig")
txt, y = tr["message"].values, tr["label"].values
splits = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(txt, y))

# 같은 5 fold의 Cthr 점수 (비교 기준)
cthr_path = os.path.join(os.path.dirname(OUT), "cthr_folds.json")
if not os.path.exists(cthr_path):
    mf = make_scorer(f1_score, average="macro")
    cthr = TunedThresholdClassifierCV(
        make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
                      LinearSVC(C=1, class_weight="balanced")),
        scoring=mf, thresholds=41, cv=StratifiedKFold(3, shuffle=True, random_state=7))
    s = [f1_score(y[b], cthr.fit(txt[a], y[a]).predict(txt[b]), average="macro") for a, b in splits]
    json.dump(s, open(cthr_path, "w"))

tok = AutoTokenizer.from_pretrained(NAME)
for k, (a, b) in enumerate(splits):
    path = os.path.join(OUT, f"fold{k}.json")
    if os.path.exists(path):
        print(f"fold{k} 이미 있음, 건너뜀", flush=True); continue
    t0 = time.time()
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = EPOCHS * int(np.ceil(len(a) / BS))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * n_steps), n_steps)
    model.train()
    for ep in range(EPOCHS):
        idx = rng.permutation(a)
        for i in range(0, len(idx), BS):
            j = idx[i:i + BS]
            batch = tok(list(txt[j]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
            model(**batch, labels=torch.tensor(y[j])).loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
        print(f"fold{k} epoch{ep + 1} {time.time() - t0:.0f}s", flush=True)
    model.eval(); prob = []
    with torch.no_grad():
        for i in range(0, len(b), 64):
            batch = tok(list(txt[b[i:i + 64]]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
            prob.append(torch.softmax(model(**batch).logits, -1)[:, 1].numpy())
    prob = np.concatenate(prob); pred = (prob >= 0.5).astype(int)
    res = {"fold": k, "macro_f1": f1_score(y[b], pred, average="macro"),
           "fn": int(((y[b] == 1) & (pred == 0)).sum()), "fp": int(((y[b] == 0) & (pred == 1)).sum()),
           "sec": time.time() - t0, "val_idx": b.tolist(), "prob": prob.tolist()}
    json.dump(res, open(path, "w"))
    print(f"fold{k} macro_f1 {res['macro_f1']:.4f} FN {res['fn']} FP {res['fp']} {res['sec']:.0f}s", flush=True)

cthr = np.array(json.load(open(cthr_path)))
f = np.array([json.load(open(os.path.join(OUT, f"fold{k}.json")))["macro_f1"] for k in range(5)])
d = f - cthr
print(f"{NAME}: fold 평균 {f.mean():.4f} | Cthr {cthr.mean():.4f} | 차이 {d.mean():+.4f} | 개선 {(d > 0).sum()} / 악화 {(d < 0).sum()}")
print("fold별 차이", np.round(d, 4).tolist())
