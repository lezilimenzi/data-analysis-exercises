"""Step 4: final training, submission, and the 80:20 report score.
Usage: python3 final.py holdout|submit [--config C0] [--seeds 3]
"""
import argparse, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
from common import data, mf1, log, save_json, ROOT, RESULTS
from exp_cnn import CONFIGS, train_predict
from load_images import load_test, save_submission

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["holdout", "submit"])
ap.add_argument("--config", default="C0")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
aug = CONFIGS[a.config]
seeds = [1000 + i for i in range(a.seeds)]
X, y = data()


def ensemble(Xtr, ytr, Xev):
    ps = []
    for s in seeds:
        t = time.time()
        ps.append(train_predict(Xtr, ytr, Xev, aug, seed=s))
        log(f"  seed {s} done {time.time() - t:.0f}s")
    return np.mean(ps, axis=0), ps


if a.stage == "holdout":
    # same split as the EDA baseline (stratified 80:20, random_state=0)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
    avg, ps = ensemble(X[tr], y[tr], X[va])
    out = {"config": a.config, "seeds": seeds,
           "single_seed_f1": [mf1(y[va], p.argmax(1)) for p in ps], "ensemble_f1": mf1(y[va], avg.argmax(1)),
           "ensemble_acc": float((avg.argmax(1) == y[va]).mean())}
    save_json("holdout.json", out)
    log(f"holdout {out}")
else:
    Xt, ids = load_test()
    avg, _ = ensemble(X, y, Xt)
    np.save(RESULTS / "test_proba.npy", avg)
    pred = avg.argmax(1)
    save_submission(ids, pred, ROOT / "submission.json")
    # checks
    sub = json.loads((ROOT / "submission.json").read_text(encoding="utf-8"))
    sample = json.loads((ROOT / "sample_submission.json").read_text(encoding="utf-8"))
    assert len(sub) == 1400 and len({r["image_id"] for r in sub}) == 1400
    assert {r["image_id"] for r in sub} == {r["image_id"] for r in sample}
    assert all(set(r) == {"image_id", "label"} and type(r["label"]) is int and 0 <= r["label"] <= 6 for r in sub)
    log(f"submission OK. class counts: {np.bincount(pred, minlength=7).tolist()}")
