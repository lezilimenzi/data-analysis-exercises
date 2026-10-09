"""Shared helpers: data, folds, metric, results IO."""
import json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from load_images import load_train

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def data():
    return load_train()  # X: (4900,28,28) float32 0~1, y: (4900,)


def folds(y):  # every experiment uses the same 3-fold split
    return list(StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def mf1(y, p):
    return f1_score(y, p, labels=list(range(7)), average="macro", zero_division=0)


def save_json(name, obj):
    (RESULTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
