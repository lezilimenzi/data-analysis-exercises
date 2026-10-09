"""Post-hoc analysis: one submission per candidate model, trained on all of train.
Written after the final submission was scored; does not change submission.csv."""
from functools import partial
from pathlib import Path

import forecast as F
from starter import load_data, save_submission

MODELS = {
    'persist': F.persist,
    'hm1': partial(F.hour_mean, k=1),
    'hm3': partial(F.hour_mean, k=3),
    'hm5': partial(F.hour_mean, k=5),
    'hm7': partial(F.hour_mean, k=7),
    'M1': partial(F.m1, **F.M1_BEST),
    'M1b': partial(F.m1, k=14, k_level=3, m=1, alpha=0.97),
    'M2-ridge': partial(F.m2_ridge, **F.M2_BEST),
    'M2-gbm': partial(F.m2_gbm, loss='squared_error', max_iter=200, max_leaf_nodes=31),
    'M4-mlp128': partial(F.m4, kind='mlp', hidden=128),
    'M4-lstm64': partial(F.m4, kind='lstm', hidden=64),
    'M3-w0.7': partial(F.m3, w=0.7),
}

train, test = load_data()
out = Path('posthoc')
out.mkdir(exist_ok=True)
for name, fn in MODELS.items():
    save_submission(test, fn(train), out / f'submission_{name}.csv')
    print('saved', out / f'submission_{name}.csv', flush=True)
