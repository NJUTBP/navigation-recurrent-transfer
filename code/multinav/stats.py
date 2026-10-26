"""Equal-source inference; never count target fits/trajectories as source replicates."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    n = len(x)
    if not n: return dict(n=0, mean=np.nan, sem=np.nan, ci_low=np.nan, ci_high=np.nan)
    m = float(x.mean())
    if n == 1: return dict(n=1, mean=m, sem=np.nan, ci_low=np.nan, ci_high=np.nan)
    se = float(x.std(ddof=1) / np.sqrt(n)); d = float(stats.t.ppf(.975, n-1) * se)
    return dict(n=n, mean=m, sem=se, ci_low=m-d, ci_high=m+d)


def sign_flip_p(values):
    """Two-sided source-block sign-flip test; symmetry/exchangeability required."""
    x = np.asarray(values, float)
    if not np.isfinite(x).all() or not len(x): return np.nan
    if len(x) > 20: raise ValueError("Exact sign-flip is capped at 20 source pairs")
    obs = abs(float(x.mean())); count = 0
    for signs in itertools.product((-1., 1.), repeat=len(x)):
        count += abs(float(np.dot(x, signs) / len(x))) >= obs - 1e-14
    return float(count / 2**len(x))


def sign_p(values):
    x = np.asarray(values, float); x = x[np.isfinite(x) & (x != 0)]
    return float(stats.binomtest(int((x > 0).sum()), len(x), .5).pvalue) if len(x) else 1.


def holm(pvalues):
    p = np.asarray(pvalues, float); out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p)); order = finite[np.argsort(p[finite])]
    if len(order): out[order] = np.minimum(1., np.maximum.accumulate(p[order] * (len(p)-np.arange(len(order)))))
    return out


def paired_source_effect(frame, left, right, *, expected_sources, expected_repeats,
                         bootstrap_samples=10000, seed=123):
    """Input: source_seed, target_seed, arm, value. Pair targets first, then sources.

    Percentile bootstrap resamples whole source blocks only. It does NOT also
    resample noisy source means internally (which would double-count variance).
    """
    x = frame[frame.arm.isin([left, right])].copy()
    keys = ["source_seed", "target_seed", "arm"]
    if x.duplicated(keys).any(): raise ValueError("Duplicate source/target/arm cells")
    p = x.pivot(index=["source_seed", "target_seed"], columns="arm", values="value")
    if left not in p or right not in p:
        pairs = pd.DataFrame(columns=["source_seed", "target_seed", "difference"])
    else:
        p = p[[left, right]].dropna(); p["difference"] = p[left] - p[right]
        pairs = p.reset_index()
    valid = []; detail = []
    for s in expected_sources:
        z = pairs.loc[pairs.source_seed == s, "difference"].to_numpy(float)
        ok = len(z) == expected_repeats and np.isfinite(z).all()
        detail.append(dict(source_seed=int(s), n_target_pairs=len(z), complete=ok,
                           difference=float(z.mean()) if ok else np.nan))
        if ok: valid.append(float(z.mean()))
    arr = np.asarray(valid, float); out = mean_ci(arr)
    complete = len(arr) == len(expected_sources)
    out.update(left=left, right=right, n_sources_planned=len(expected_sources),
               n_source_pairs_complete=len(arr), n_target_pairs=len(pairs),
               n_positive_sources=int((arr > 0).sum()), complete=complete,
               inference_status="confirmatory_complete" if complete else "incomplete_exploratory_only")
    # Effect estimates from incomplete sources are descriptive only; do not
    # output a confirmatory p value for a selectively completed cohort.
    if len(arr) >= 2 and complete:
        degenerate = bool(np.std(arr, ddof=1) <= 1e-14)
        p_t = np.nan if degenerate else float(stats.ttest_1samp(arr, 0).pvalue)
        out.update(p_t=p_t, degenerate_source_variance=degenerate,
                   p_sign_flip=sign_flip_p(arr) if len(arr) <= 20 else np.nan,
                   p_sign=sign_p(arr), df=len(arr)-1)
        rng = np.random.default_rng(seed)
        b = arr[rng.integers(0, len(arr), (int(bootstrap_samples), len(arr)))].mean(axis=1)
        out.update(bootstrap_low=float(np.quantile(b, .025)), bootstrap_high=float(np.quantile(b, .975)))
    else:
        out.update(p_t=np.nan, p_sign_flip=np.nan, p_sign=np.nan, df=max(0, len(arr)-1),
                   bootstrap_low=np.nan, bootstrap_high=np.nan)
    if len(arr) > 1:
        loo = (arr.sum()-arr)/(len(arr)-1)
        out.update(leave_one_source_out_min=float(loo.min()), leave_one_source_out_max=float(loo.max()))
    return out, detail, pairs


def normalized_log_auc(frame, *, budgets, group_columns=("source_seed", "target_seed", "arm"),
                       budget_column="budget", value_column="value"):
    budgets = sorted(set(int(x) for x in budgets))
    if len(budgets) < 2: raise ValueError("AUC requires at least two prespecified budgets")
    rows = []
    for key, group in frame.groupby(list(group_columns), dropna=False):
        z = group[group[budget_column].isin(budgets)].sort_values(budget_column)
        if z[budget_column].duplicated().any(): raise ValueError("Duplicate budget cell")
        if z[budget_column].tolist() != budgets or not np.isfinite(z[value_column]).all():
            continue  # incomplete grid is not silently interpolated
        t = np.log(np.asarray(budgets, float)); y = z[value_column].to_numpy(float)
        area = float(np.sum((y[1:]+y[:-1])*.5*np.diff(t))/(t[-1]-t[0]))
        rows.append({**dict(zip(group_columns, key)), "value": area})
    return pd.DataFrame(rows, columns=[*group_columns, "value"])
