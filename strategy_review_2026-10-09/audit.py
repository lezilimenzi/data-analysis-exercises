"""Read contestant inputs only; write reproducible diagnostics beside this script."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
report = {}


def read(name):
    return pd.read_csv(ROOT / name)


def counts(values):
    return {str(k): int(v) for k, v in pd.Series(values).value_counts().sort_index().items()}


def images(paths):
    hashes, modes, sizes = [], Counter(), Counter()
    for path in paths:
        with Image.open(path) as im:
            modes[im.mode] += 1
            sizes[str(im.size)] += 1
            hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
    return hashes, dict(modes), dict(sizes)


tr, te = read('problem_01/train.csv'), read('problem_01/test.csv')
raw = tr.drop(columns=['record_id', 'inspection_required'])
derived = pd.DataFrame({'power': tr.torque_nm * tr.rotation_rpm,
                        'wear': tr.tool_usage_min * tr.torque_nm,
                        'delta_temp': tr.process_temp_c - tr.ambient_temp_c})
report['P1'] = dict(train=len(tr), test=len(te), labels=counts(tr.inspection_required),
                    missing=tr.isna().sum().to_dict(),
                    missing_rows=int(raw.isna().any(axis=1).sum()),
                    missing_positive=int(tr.loc[raw.isna().any(axis=1), 'inspection_required'].sum()),
                    derived_missing=derived.isna().sum().to_dict(),
                    duplicate_features=int(raw.duplicated().sum()),
                    id_overlap=len(set(tr.record_id) & set(te.record_id)))

p = ROOT / 'problem_02'
train_paths, test_paths = sorted(p.glob('train/*/*.png')), sorted(p.glob('test/*.png'))
a, modes, sizes = images(train_paths)
b, tmodes, tsizes = images(test_paths)
report['P2'] = dict(train=len(a), test=len(b), labels=dict(Counter(x.parent.name for x in train_paths)),
                    modes=modes, sizes=sizes, test_modes=tmodes, test_sizes=tsizes,
                    train_duplicate_pixels=len(a)-len(set(a)), test_duplicate_pixels=len(b)-len(set(b)),
                    overlap_pixels=len(set(a) & set(b)))

tr, te = read('problem_03/train.csv'), read('problem_03/test.csv')
t = pd.to_datetime(tr.timestamp)
report['P3'] = dict(train=len(tr), test=len(te), missing=int(tr.isna().sum().sum()),
                    duplicated_times=int(t.duplicated().sum()),
                    hourly_gaps=int((t.diff().dropna() != pd.Timedelta(hours=1)).sum()),
                    start=str(t.iloc[0]), end=str(t.iloc[-1]),
                    horizon_valid=bool(np.array_equal(te.horizon_hour, np.arange(1, 73))),
                    next_times_valid=bool(np.array_equal(pd.to_datetime(te.timestamp).to_numpy(),
                        pd.date_range(t.iloc[-1]+pd.Timedelta(hours=1), periods=72, freq='h').to_numpy())))

tr, lb = read('problem_04/train_series.csv'), read('problem_04/train_labels.csv')
channels = tr.columns[2:]
seq = {k: g.sort_values('step')[channels].to_numpy() for k, g in tr.groupby('sample_id')}
signatures = {}
for k, x in seq.items():
    sig = tuple(sorted(map(tuple, x)))
    signatures.setdefault(sig, []).append(k)
groups = [v for v in signatures.values() if len(v) > 1]
lookup = lb.set_index('sample_id').label.to_dict()
report['P4'] = dict(train=len(lb), labels=counts(lb.label),
                    complete_steps=all(np.array_equal(g.sort_values('step').step, np.arange(15)) for _, g in tr.groupby('sample_id')),
                    row_multiset_duplicate_groups=[{'ids': v, 'labels': [lookup[k] for k in v]} for v in groups],
                    ordered_duplicate_count=len(seq)-len({x.tobytes() for x in seq.values()}))

tr, te = read('problem_05/data/train.csv'), read('problem_05/data/test.csv')
a, modes, sizes = images([ROOT / 'problem_05' / x for x in tr.image_path])
b, _, _ = images([ROOT / 'problem_05' / x for x in te.image_path])
ctr = read('problem_05_codex/data/train.csv')
report['P5'] = dict(train=len(tr), test=len(te), labels=counts(tr.label), modes=modes, sizes=sizes,
                    train_duplicate_pixels=len(a)-len(set(a)), overlap_pixels=len(set(a)&set(b)),
                    codex_rows_identical=tr.equals(ctr))

tr, te = read('problem_06/data/train.csv'), read('problem_06/data/test.csv')
normalize = lambda s: re.sub(r'\s+', ' ', re.sub(r'\d+', '<NUM>', s.lower())).strip()
key, testkey = tr.message.map(normalize), te.message.map(normalize)
report['P6'] = dict(train=len(tr), test=len(te), labels=counts(tr.label),
                    missing=int(tr.isna().sum().sum()), raw_duplicate_rows=int(tr.message.duplicated().sum()),
                    normalized_duplicate_rows=int(key.duplicated().sum()),
                    normalized_conflicting_groups=int(tr.groupby(key).label.nunique().gt(1).sum()),
                    normalized_test_overlap_rows=int(testkey.isin(set(key)).sum()),
                    normalization='lowercase, digit runs -> <NUM>, whitespace collapsed; no punctuation removal')

tr, te = read('problem_07/data/train.csv'), read('problem_07/data/test.csv')
log, sessions = read('problem_07/data/link_log.csv'), read('problem_07/data/sessions.csv')
g = log.groupby('sample_id')
recent = g.relative_second.apply(lambda s: len(set(s) & set(range(-4, 1))))
report['P7'] = dict(train=len(tr), test=len(te), labels=counts(tr.label),
                    train_sessions=tr.session_id.nunique(), test_sessions=te.session_id.nunique(),
                    session_overlap=len(set(tr.session_id)&set(te.session_id)),
                    log_rows=len(log), seconds_range=[int(log.relative_second.min()), int(log.relative_second.max())],
                    duplicate_id_second=int(log.duplicated(['sample_id', 'relative_second']).sum()),
                    missing_recent5_train=int((recent.loc[tr.sample_id] < 5).sum()),
                    missing_recent5_test=int((recent.loc[te.sample_id] < 5).sum()),
                    train_per_session=counts(tr.session_id), test_per_session=counts(te.session_id),
                    train_mobility=counts(tr.merge(sessions).mobility),
                    test_mobility=counts(te.merge(sessions).mobility),
                    nonnumeric_sensor_counts={c: int(pd.to_numeric(log[c], errors='coerce').isna().sum())
                        for c in log.columns[2:] if c not in ['network_mode', 'transfer_state']})

tr, te = read('problem_08/data/train.csv'), read('problem_08/data/test.csv')
with np.load(ROOT / 'problem_08/data/sensor_data.npz') as data:
    pos = {k: i for i, k in enumerate(data['cycle_id'])}
    ti, si = tr.cycle_id.map(pos).to_numpy(), te.cycle_id.map(pos).to_numpy()
    pressure = data['pressure']
    crossed = pressure[:, 900:1200] > 20
    assert crossed.any(axis=1).all()
    rise = crossed.argmax(axis=1) + 900
    y = tr.label.to_numpy()
    fit = tr.split.eq('fit').to_numpy()
    ranges = [(int(rise[ti][fit & (y == k)].min()), int(rise[ti][fit & (y == k)].max())) for k in range(4)]
    bounds = [(ranges[k][1]+ranges[k+1][0])/2 for k in range(3)]
    stress = {str(shift): float(f1_score(y[~fit], np.digitize(rise[ti][~fit]+shift, bounds),
                              labels=range(4), average='macro', zero_division=0)) for shift in range(-3,4)}
    temps = data['temperature'].mean(axis=1)
    report['P8'] = dict(train=len(tr), test=len(te), labels=counts(y), splits=counts(tr.split),
                       arrays={k: list(data[k].shape) for k in data.files},
                       nonfinite={k: int((~np.isfinite(data[k])).sum()) for k in data.files if k != 'cycle_id'},
                       fit_rise_ranges=ranges, fit_bounds=bounds, valid_shift_stress=stress,
                       median_temperature_train=float(np.median(temps[ti])), median_temperature_test=float(np.median(temps[si])),
                       high_threshold_no_cross_train=int((~(pressure[ti,900:1200]>150).any(axis=1)).sum()),
                       high_threshold_no_cross_test=int((~(pressure[si,900:1200]>150).any(axis=1)).sum()))

(OUT / 'data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=int)+'\n')
print(json.dumps(report, ensure_ascii=False, indent=2, default=int))
