"""Leakage-safe, fixed-origin 72-hour forecasting and winter validation."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from starter import load_data, save_submission

ROOT = Path(__file__).resolve().parent
MAIN = [pd.Timestamp(f'2014-12-{d} 23:00') for d in (19, 22, 25)]
EXTRA = [pd.Timestamp(f'2014-12-{d:02d} 23:00') for d in (1, 4, 7, 10, 13)]
HOURS = np.arange(1, 73)


def features(data, origin, auxiliary=False, linear=False):
    """Only rows at/before origin contribute observation features."""
    past = data.loc[:origin]
    temp = past.temperature_c
    targets = pd.date_range(origin + pd.Timedelta(hours=1), periods=72, freq='h')
    hour_phase = 2 * np.pi * targets.hour.to_numpy() / 24
    year_phase = 2 * np.pi * (targets.dayofyear.to_numpy() - 1) / np.where(
        targets.is_leap_year, 366, 365)
    state = [temp.iloc[-1 - k] for k in (0, 24, 48, 72)]
    state += [temp.iloc[-24:].mean(), temp.iloc[-72:].mean(),
              temp.iloc[-24:].min(), temp.iloc[-24:].max(),
              temp.iloc[-1] - temp.iloc[-25]]
    columns = [np.sin(hour_phase), np.cos(hour_phase), np.sin(year_phase),
               np.cos(year_phase), HOURS / 72]
    columns += [np.full(72, value) for value in state]
    if auxiliary:
        for name in ('dew_point_c', 'pressure_hpa'):
            series = past[name]
            columns += [np.full(72, value) for value in
                        (series.iloc[-1], series.iloc[-24:].mean(),
                         series.iloc[-1] - series.iloc[-25])]
        columns.append(np.full(72, temp.iloc[-1] - past.dew_point_c.iloc[-1]))
    if linear:
        # Current/lagged temperatures and their 24-hour change, not duplicate h.
        columns += [value * HOURS / 72 for value in state[:4] + [state[-1]]]
    return np.column_stack(columns)


def training_data(history, auxiliary=False, linear=False, winter=False):
    positions = np.arange(len(history))
    mask = (positions >= 72) & (positions + 72 < len(history))
    mask &= history.index.hour == 23
    if winter:
        mask &= history.index.month.isin([12, 1, 2])
    positions = positions[mask]
    origins = history.index[positions]
    x = np.concatenate([features(history, o, auxiliary, linear) for o in origins])
    y = np.concatenate([history.temperature_c.iloc[p + 1:p + 73].to_numpy()
                        for p in positions])
    assert (origins + pd.Timedelta(hours=72) <= history.index[-1]).all()
    return x, y, origins


def predict(history, candidate):
    kind = candidate['kind']
    target_hours = pd.date_range(history.index[-1] + pd.Timedelta(hours=1),
                                periods=72, freq='h').hour
    if kind == 'last':
        values = np.full(72, history.temperature_c.iloc[-1])
    elif kind == 'repeat':
        values = np.tile(history.temperature_c.iloc[-24:].to_numpy(), 3)
    elif kind == 'hourly':
        recent = history.iloc[-24 * candidate['days']:]
        means = recent.groupby(recent.index.hour).temperature_c.mean()
        values = means.loc[target_hours].to_numpy()
    elif kind == 'pattern':
        winter = history[history.index.month.isin([12, 1, 2])]
        base = winter.groupby(winter.index.hour).temperature_c.mean()
        recent = history.iloc[-candidate['window']:]
        delta = (recent.temperature_c.to_numpy() - base.loc[recent.index.hour].to_numpy()).mean()
        tau = candidate['tau']
        decay = np.ones(72) if tau is None else np.exp(-HOURS / tau)
        values = base.loc[target_hours].to_numpy() + delta * decay
    else:
        linear = kind == 'ridge'
        x, y, _ = training_data(history, candidate['auxiliary'], linear, candidate['winter'])
        if linear:
            model = make_pipeline(StandardScaler(), Ridge(alpha=candidate['alpha']))
        else:
            model = HistGradientBoostingRegressor(
                loss='absolute_error', learning_rate=0.05, max_iter=200,
                max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1,
                early_stopping=False, random_state=42)
        # Bound CPU usage and avoid oversubscription on this small dataset.
        with threadpool_limits(limits=2):
            model.fit(x, y)
            values = model.predict(features(history, history.index[-1],
                                            candidate['auxiliary'], linear))
    values = np.asarray(values, dtype=float)
    if values.shape != (72,) or not np.isfinite(values).all():
        raise ValueError('Forecast must have 72 finite values.')
    return values


def forecast_at(data, cutoff, candidate):
    return predict(data.loc[:cutoff].copy(), candidate)


def metrics(actual, predicted):
    error = np.abs(actual - predicted)
    return {'mae': error.mean(), 'rmse': np.sqrt(np.mean((actual - predicted) ** 2)),
            **{f'day{d + 1}_mae': error[d * 24:(d + 1) * 24].mean() for d in range(3)}}


def select_candidate(baseline, pattern, regression, scores):
    """Promote only when both periods improve and >=2 main folds improve."""
    def fold_scores(name, phase):
        return scores[(scores.candidate == name) & (scores.phase == phase)].set_index('cutoff').mae
    chosen = baseline
    decisions = []
    for challenger in (pattern, regression):
        current_main, next_main = fold_scores(chosen, 'main'), fold_scores(challenger, 'main')
        current_extra, next_extra = fold_scores(chosen, 'extra'), fold_scores(challenger, 'extra')
        wins = int((next_main < current_main).sum())
        accept = bool(next_main.mean() < current_main.mean()
                      and next_extra.mean() < current_extra.mean() and wins >= 2)
        decisions.append({'incumbent': chosen, 'challenger': challenger,
                          'main_fold_wins': wins, 'accepted': accept})
        if accept:
            chosen = challenger
    return chosen, decisions


def check_inputs(train, test, sample):
    if not np.isfinite(train[['temperature_c', 'dew_point_c', 'pressure_hpa']]).all().all():
        raise ValueError('Non-finite observations.')
    if not np.array_equal(test.horizon_hour.to_numpy(), HOURS):
        raise ValueError('Unexpected forecast horizons.')
    if not test.forecast_id.is_unique or not sample.forecast_id.equals(test.forecast_id):
        raise ValueError('Submission IDs/order mismatch.')


def check_submission(path, sample):
    result = pd.read_csv(path)
    assert result.columns.tolist() == ['forecast_id', 'temperature_c']
    assert len(result) == 72 and result.forecast_id.is_unique
    assert result.forecast_id.equals(sample.forecast_id)
    assert np.isfinite(result.temperature_c).all()
    return result


def write_report(summary):
    plan_path = ROOT / 'PLAN.md'
    plan = plan_path.read_text().split('\n## 12. 실행 결과')[0]
    plan += '\n## 12. 실행 결과\n\n'
    plan += '최종 선택: `' + summary['selected']['name'] + '`. 설정: `'
    plan += json.dumps(summary['selected'], ensure_ascii=False) + '`.\n\n'
    plan += '| 후보 | 구간 | 평균 MAE | 평균 구간 RMSE | 1일 MAE | 2일 MAE | 3일 MAE |\n'
    plan += '|---|---|---:|---:|---:|---:|---:|\n'
    for row in summary['summary']:
        plan += '| ' + row['candidate'] + ' | ' + row['phase'] + ' | '
        plan += ' | '.join(f'{row[key]:.4f}' for key in
                           ('mae', 'rmse', 'day1_mae', 'day2_mae', 'day3_mae')) + ' |\n'
    plan += '\nmain은 최근 3구간, extra는 이전 겨울 5구간이다. RMSE 열은 구간별 RMSE의 평균이다.\n\n'
    for decision in summary['decisions']:
        plan += '- ' + decision['incumbent'] + ' → ' + decision['challenger']
        plan += ': 주 검증 ' + str(decision['main_fold_wins']) + '/3구간 개선, '
        plan += ('채택' if decision['accepted'] else '미채택') + '.\n'
    plan += '\n전체 관측으로 최종 방법을 다시 계산하고 submission.csv를 생성했다. '
    plan += '이 방법은 겨울 시간대 평균과 최근 기온만 사용한다. 이슬점·기압은 회귀 비교에 사용했으나 최종 방법에는 포함하지 않는다.\n'
    plan += '\n재현: `python3 -m unittest -v test_forecast.py`, `python3 forecast.py`. '
    plan += '환경: Python 3, NumPy 1.26.4, pandas 2.3.2, scikit-learn 1.7.2. '
    plan += '설치된 SciPy가 NumPy 지원 범위 경고를 냈지만 필수 테스트와 모든 후보 실행은 완료됐다. 패키지는 변경하지 않았다.\n'
    plan += '\n검증 결과는 test의 실측 성능을 의미하지 않는다. extra도 독립적인 최종 평가가 아닌 안정성 점검이다.\n'
    plan_path.write_text(plan)


def main():
    train, test = load_data()
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    check_inputs(train, test, sample)
    data = train.set_index('timestamp')
    candidates, records, prediction_records = {}, [], []

    def evaluate(candidate, phase='main'):
        name = candidate['name']
        candidates[name] = candidate
        for cutoff in MAIN if phase == 'main' else EXTRA:
            predicted = forecast_at(data, cutoff, candidate)
            targets = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=72, freq='h')
            actual = data.loc[targets, 'temperature_c'].to_numpy()
            result = metrics(actual, predicted)
            records.append({'candidate': name, 'phase': phase, 'cutoff': cutoff, **result})
            prediction_records.extend({'candidate': name, 'phase': phase, 'cutoff': cutoff,
                                       'timestamp': t, 'horizon_hour': int(h),
                                       'actual': float(a), 'prediction': float(p)}
                                      for t, h, a, p in zip(targets, HOURS, actual, predicted))
            print(f'{phase} {name} {cutoff.date()} MAE={result["mae"]:.4f}', flush=True)
        pd.DataFrame(records).to_csv(ROOT / 'validation_results.csv', index=False)
        pd.DataFrame(prediction_records).to_csv(ROOT / 'validation_predictions.csv', index=False)

    def best(names):
        return min(names, key=lambda name: np.mean([r['mae'] for r in records
                                                   if r['candidate'] == name and r['phase'] == 'main']))

    baselines = [dict(name='last', kind='last'), dict(name='repeat24', kind='repeat')]
    baselines += [dict(name=f'hourly{d}d', kind='hourly', days=d) for d in (3, 7)]
    for candidate in baselines:
        evaluate(candidate)
    baseline = best([c['name'] for c in baselines])

    patterns = [dict(name=f'pattern_w{w}_tau{tau}', kind='pattern', window=w, tau=tau)
                for w in (24, 72) for tau in (24, 72, None)]
    for candidate in patterns:
        evaluate(candidate)
    pattern = best([c['name'] for c in patterns])

    regressions = [dict(name=f'ridge{alpha}_all_temp', kind='ridge', alpha=alpha,
                        auxiliary=False, winter=False) for alpha in (1, 100)]
    regressions.append(dict(name='boost_all_temp', kind='boost', auxiliary=False, winter=False))
    for candidate in regressions:
        evaluate(candidate)
    winning_model = candidates[best([c['name'] for c in regressions])]
    variants = [winning_model['name']]
    for auxiliary, winter in ((True, False), (False, True), (True, True)):
        candidate = {**winning_model, 'auxiliary': auxiliary, 'winter': winter}
        candidate['name'] = winning_model['name'].replace('all_temp',
                             ('winter' if winter else 'all') + ('_aux' if auxiliary else '_temp'))
        evaluate(candidate)
        variants.append(candidate['name'])
    regression = best(variants)

    shortlist = [baseline, pattern, regression]
    print('FROZEN SHORTLIST', shortlist, flush=True)
    for name in shortlist:
        evaluate(candidates[name], phase='extra')
    scores = pd.DataFrame(records)
    chosen, decisions = select_candidate(baseline, pattern, regression, scores)
    predicted = predict(data, candidates[chosen])
    save_submission(test, predicted, ROOT / 'submission.csv')
    check_submission(ROOT / 'submission.csv', sample)
    summary = {'selected': candidates[chosen], 'shortlist': shortlist, 'decisions': decisions,
               'all_candidates': candidates,
               'summary': scores.groupby(['candidate', 'phase'])[
                   ['mae', 'rmse', 'day1_mae', 'day2_mae', 'day3_mae']].mean().reset_index().to_dict('records')}
    write_report(summary)
    print('SELECTED', chosen, 'DECISIONS', decisions, flush=True)
    print('SUBMISSION', ROOT / 'submission.csv', flush=True)


if __name__ == '__main__':
    main()
