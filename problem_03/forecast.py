"""72-hour hourly temperature forecasts.

Every model is a function predict(hist) -> np.ndarray(72), where hist holds only
observations up to the forecast origin (a 23:00 timestamp).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H = 72


def day_matrix(hist, col):
    """(days, 24) matrix of one column. hist must start at 00:00 and end at 23:00."""
    assert hist.timestamp.iloc[0].hour == 0 and hist.timestamp.iloc[-1].hour == 23
    return hist[col].to_numpy(float).reshape(-1, 24)


def profile(T, k):
    """Row d = hourly mean of days d-k+1..d. Rows d < k-1 are NaN."""
    c = np.cumsum(np.vstack([np.zeros(24), T]), axis=0)
    out = np.full(T.shape, np.nan)
    out[k - 1:] = (c[k:] - c[:-k]) / k
    return out


# ---------------- M0: baselines ----------------
def persist(hist):
    return np.full(H, hist.temperature_c.iloc[-1], float)


def hour_mean(hist, k):
    return np.tile(day_matrix(hist, 'temperature_c')[-k:].mean(0), 3)


# ---------------- M1: diurnal profile + decaying anomaly ----------------
def m1_all(T, k, m, alpha, k_level=None):
    """M1 forecast from every origin day d (end of day d). Returns (days, 72).

    k_level (M1b): keep the k-day profile's shape but set its level to the
    mean of the last k_level days.
    """
    prof = profile(T, k)
    if k_level is not None:
        prof = prof - prof.mean(1, keepdims=True) + profile(T, k_level).mean(1, keepdims=True)
    anom = (T[:, 24 - m:] - prof[:, 24 - m:]).mean(1)
    return np.tile(prof, 3) + anom[:, None] * alpha ** np.arange(1, H + 1)


def m1(hist, k, m, alpha, k_level=None):
    return m1_all(day_matrix(hist, 'temperature_c'), k, m, alpha, k_level)[-1]


M1_BEST = dict(k=5, m=1, alpha=0.97)  # stage m1: A 2.671 (adopted, see PLAN.md 4절)


# ---------------- M2: direct multi-horizon regression on M1 residuals ----------------
def origin_features(hist, k):
    """Features known at the end of each day d. (days, n_feat); early rows contain NaN."""
    T, Td, P = (day_matrix(hist, c) for c in ('temperature_c', 'dew_point_c', 'pressure_hpa'))
    t, td, p = T.ravel(), Td.ravel(), P.ravel()
    end = np.arange(len(T)) * 24 + 23

    def lag(x, n):
        out = np.full(len(end), np.nan)
        ok = end - n >= 0
        out[ok] = x[end[ok] - n]
        return out

    def mean_last(x, n):
        c = np.concatenate([[0], np.cumsum(x)])
        out = np.full(len(end), np.nan)
        ok = end + 1 >= n
        out[ok] = (c[end[ok] + 1] - c[end[ok] + 1 - n]) / n
        return out

    doy = 2 * np.pi * hist.timestamp.dt.dayofyear.to_numpy()[end] / 365.25
    return np.column_stack([
        T - profile(T, k),  # last 24h anomalies
        mean_last(t, 24), mean_last(t, 72), mean_last(t, 168),
        t[end] - td[end], mean_last(t - td, 24),
        p[end], p[end] - lag(p, 3), p[end] - lag(p, 12), p[end] - lag(p, 24),
        td[end] - lag(td, 24), T.max(1) - T.min(1),
        np.sin(doy), np.cos(doy)])


def residual_data(hist, season_days=None):
    """Training rows = origin days whose 72 targets all lie inside hist; plus the last origin's row.

    Returns X (n, f), residual targets R (n, 72), x0 (f,), base0 (72,).
    """
    T = day_matrix(hist, 'temperature_c')
    N = len(T)
    base = m1_all(T, **M1_BEST)
    X = origin_features(hist, M1_BEST['k'])
    n = N - 3  # origin d uses days d+1..d+3, so d <= N-4
    Y = np.hstack([T[1:n + 1], T[2:n + 2], T[3:n + 3]])
    assert Y.shape == (n, H)
    ok = np.isfinite(X[:n]).all(1) & np.isfinite(base[:n]).all(1)
    if season_days is not None:
        doy = hist.timestamp.dt.dayofyear.to_numpy()[23::24]
        dist = np.abs(doy[:n] - doy[-1])
        ok &= np.minimum(dist, 365 - dist) <= season_days
    return X[:n][ok], (Y - base[:n])[ok], X[-1], base[-1]


M2_BEST = dict(alpha=1.0, season_days=None)  # stage m2: A 2.382, win 65.3% vs M1 (adopted)


def m2_ridge(hist, alpha, season_days=None):
    """Multi-output ridge: one coefficient set per horizon."""
    X, R, x0, b0 = residual_data(hist, season_days)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, R)
    return b0 + model.predict(x0[None])[0]


def stack_horizons(X):
    """Repeat each origin row 72 times and append horizon features (h, target hour sin/cos)."""
    h = np.arange(1, H + 1)
    hr = 2 * np.pi * ((h - 1) % 24) / 24
    hcols = np.column_stack([h, np.sin(hr), np.cos(hr)])
    return np.hstack([np.repeat(X, H, 0), np.tile(hcols, (len(X), 1))])


def m2_gbm(hist, loss, max_iter, max_leaf_nodes, season_days=None):
    X, R, x0, b0 = residual_data(hist, season_days)
    model = HistGradientBoostingRegressor(
        loss=loss, max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=50, early_stopping=False, random_state=0)
    model.fit(stack_horizons(X), R.ravel())
    return b0 + model.predict(stack_horizons(x0[None]))


# ---------------- M4: small neural nets on M1 residuals ----------------
def sequence_inputs(hist, k):
    """Per origin day d: last 168h of [T anomaly vs profile_d, T-Td, 1h pressure change] + doy sin/cos.
    Returns (days, 168, 5); early rows contain NaN."""
    T, Td, P = (day_matrix(hist, c) for c in ('temperature_c', 'dew_point_c', 'pressure_hpa'))
    N = len(T)
    t, td, p = T.ravel(), Td.ravel(), P.ravel()
    dp = np.concatenate([[np.nan], np.diff(p)])
    prof = profile(T, k)
    out = np.full((N, 168, 5), np.nan)
    doy = 2 * np.pi * hist.timestamp.dt.dayofyear.to_numpy()[23::24] / 365.25
    for d in range(7, N):
        s = slice((d - 6) * 24, (d + 1) * 24)
        out[d, :, 0] = t[s] - np.tile(prof[d], 7)
        out[d, :, 1] = t[s] - td[s]
        out[d, :, 2] = dp[s]
        out[d, :, 3], out[d, :, 4] = np.sin(doy[d]), np.cos(doy[d])
    return out


def _fit_net(make_net, Xtr, Rtr, Xva, Rva, seed, epochs=200, patience=20, batch=64):
    import torch
    torch.manual_seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr, Rtr, Xva, Rva = (torch.tensor(a, dtype=torch.float32) for a in (Xtr, Rtr, Xva, Rva))
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(batch):
            opt.zero_grad()
            (net(Xtr[idx]) - Rtr[idx]).abs().mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = (net(Xva) - Rva).abs().mean().item()
        if v < best:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return net


def m4(hist, kind, hidden, season_days=None, seeds=5):
    """kind='mlp' uses origin_features; kind='lstm' uses sequence_inputs. Averages `seeds` nets."""
    import torch
    from torch import nn
    T = day_matrix(hist, 'temperature_c')
    N = len(T)
    base = m1_all(T, **M1_BEST)
    feats = origin_features(hist, M1_BEST['k']) if kind == 'mlp' else sequence_inputs(hist, M1_BEST['k'])
    n = N - 3  # origin d uses days d+1..d+3, so d <= N-4
    R = np.hstack([T[1:n + 1], T[2:n + 2], T[3:n + 3]]) - base[:n]
    ok = np.isfinite(feats[:n].reshape(n, -1)).all(1) & np.isfinite(R).all(1)
    if season_days is not None:
        doy = hist.timestamp.dt.dayofyear.to_numpy()[23::24]
        dist = np.abs(doy[:n] - doy[-1])
        ok &= np.minimum(dist, 365 - dist) <= season_days
    d = np.flatnonzero(ok)
    # internal validation: last 10% of origins; training origins end >= 3 days before it
    va = d[-max(1, len(d) // 10):]
    tr = d[d + 3 <= va[0]]
    assert tr.max() + 3 <= va[0] and va.max() + 3 <= N - 1
    axes = (0,) if kind == 'mlp' else (0, 1)
    mu = feats[tr].mean(axes)
    sd = feats[tr].std(axes) + 1e-6
    Z = (feats - mu) / sd
    f = feats.shape[-1]

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(f, hidden, batch_first=True)
            self.out = nn.Linear(hidden, H)

        def forward(self, x):
            return self.out(self.lstm(x)[0][:, -1])

    def make_net():
        if kind == 'mlp':
            return nn.Sequential(nn.Linear(f, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, H))
        return LSTMNet()

    x0 = torch.tensor(Z[-1:], dtype=torch.float32)
    preds = []
    for seed in range(seeds):
        net = _fit_net(make_net, Z[tr], R[tr], Z[va], R[va], seed)
        with torch.no_grad():
            preds.append(net(x0)[0].numpy())
    return base[-1] + np.mean(preds, 0)


# ---------------- M3: blend of the two adopted models ----------------
def m3(hist, w):
    return w * m2_ridge(hist, **M2_BEST) + (1 - w) * m1(hist, **M1_BEST)
