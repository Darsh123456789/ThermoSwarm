"""Step 4 — Hypothesis 1 (storm-time model bias) and Hypothesis 2 (recovery τ vs altitude).

H1: samples = intervals ≤ 2 d containing the peak of storms with Kp ≥ 7, 300–550 km.
    Statistic: median R. 95% CI: cluster bootstrap over objects (2,000 resamples);
    second CI clustered by storm.
Exploratory: storm response relative to each object's own pre-storm baseline,
    X = R / R_pre − 1, R_pre = median R in [onset − 4 d, onset − 1 d].

H2: driven-relaxation model per 50-km band
    dδ/dt = α Q'(t) − δ/τ,  Q' = Newell coupling minus its pre-storm median,
    δ = 0 at onset − 4 d, integrated hourly; compared with X averaged over each
    TLE interval. τ by grid search with α solved by least squares; τ CI by
    bootstrap over storms. Storms used: those with OMNI coverage and calibrated B.
"""
import json
import numpy as np, pandas as pd
from scipy import stats
import spaceweather as sw

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
rng = np.random.default_rng(1)
res = {}

iv = pd.read_parquet(f"{OUT}/intervals.parquet")
st = pd.read_parquet(f"{OUT}/storm_samples.parquet")
storms = pd.read_csv(f"{OUT}/storms.csv", parse_dates=["onset", "end", "peak_time"])

# ---------------- H1 ----------------
h1 = st[(st.kp_max >= 7 - 1e-6) & st.alt.between(300, 550)]
med = h1.R.median()
objs = h1.norad.unique()
by_obj = {o: g.R.values for o, g in h1.groupby("norad")}
boot = [np.median(np.concatenate([by_obj[o] for o in rng.choice(objs, len(objs))]))
        for _ in range(2000)]
sids = h1.storm_id.unique()
by_storm = {s: g.R.values for s, g in h1.groupby("storm_id")}
boot_s = [np.median(np.concatenate([by_storm[s] for s in rng.choice(sids, len(sids))]))
          for _ in range(2000)]
res["H1"] = dict(n_samples=len(h1), n_objects=len(objs), storms=[int(s) for s in sids],
                 median_R=med, ci_obj=list(np.percentile(boot, [2.5, 97.5])),
                 ci_storm=list(np.percentile(boot_s, [2.5, 97.5])),
                 per_storm={int(k): v for k, v in h1.groupby("storm_id").R.median().round(3).to_dict().items()},
                 frac_R_gt_1=float((h1.R > 1).mean()))
lo, hi = res["H1"]["ci_obj"]
res["H1"]["verdict"] = ("falsified (CI within ±10% of 1)" if lo >= 0.9 and hi <= 1.1 else
                        "supported (>20% underestimate)" if lo > 1.2 else "inconclusive")

# all storms, by altitude band (for the paper)
st["band"] = pd.cut(st.alt, np.arange(200, 801, 50))
_t = st.groupby("band", observed=True).R.agg(["count", "median"]).round(3)
res["all_storms_by_band"] = {str(k): v for k, v in _t.to_dict("index").items()}

# ------------- storm-relative response (exploratory) -------------
rel = []
for _, s in storms.iterrows():
    pre = iv[(iv.t2 <= s.onset - pd.Timedelta("1D")) & (iv.t1 >= s.onset - pd.Timedelta("4D"))]
    base = pre.groupby("norad").R.median().rename("R_pre")
    win = iv[(iv.t1 >= s.onset - pd.Timedelta("4D")) & (iv.t2 <= s.onset + pd.Timedelta("7D"))].join(base, on="norad")
    win = win[win.R_pre.notna()].copy()
    win["X"] = win.R / win.R_pre - 1
    win["storm_id"], win["kp_max"], win["onset"] = s.storm_id, s.kp_max, s.onset
    rel.append(win)
rel = pd.concat(rel)
rel.to_parquet(f"{OUT}/storm_relative.parquet")
pk = rel[(rel.t1 <= rel.onset.map(lambda x: x) + pd.Timedelta("0D"))]
peakwin = []
for _, s in storms.iterrows():
    d = rel[(rel.storm_id == s.storm_id) & (rel.t1 <= s.peak_time) & (rel.t2 >= s.peak_time) & ((rel.jd2 - rel.jd1) <= 2)]
    peakwin.append(d)
peakwin = pd.concat(peakwin)
pre_R = rel.groupby(["storm_id"]).R_pre.median()
res["relative"] = dict(
    kp7_300_550_median_X=float(peakwin[(peakwin.kp_max >= 7 - 1e-6) & peakwin.alt.between(300, 550)].X.median()),
    all_median_X=float(peakwin.X.median()),
    per_storm_X={int(k): v for k, v in peakwin.groupby("storm_id").X.median().round(3).to_dict().items()},
    pre_storm_baseline_R={int(k): v for k, v in pre_R.round(3).to_dict().items()})
d = peakwin[(peakwin.kp_max >= 7 - 1e-6) & peakwin.alt.between(300, 550)]
objs2 = d.norad.unique(); g2 = {o: x.X.values for o, x in d.groupby("norad")}
b2 = [np.median(np.concatenate([g2[o] for o in rng.choice(objs2, len(objs2))])) for _ in range(2000)]
res["relative"]["kp7_ci"] = list(np.percentile(b2, [2.5, 97.5]))
# altitude dependence of the relative response (all storms, Kp>=6-)
pw = peakwin.dropna(subset=["X"])
sl = stats.theilslopes(pw.X, pw.alt)
res["relative"]["X_vs_alt_theilsen_per100km"] = [sl.slope * 100, sl.low_slope * 100, sl.high_slope * 100]

# ---------------- H2: relaxation fit ----------------
omni = sw.load_omni()
Q = omni.newell.interpolate(limit=6)
omni_end = Q.dropna().index.max()
fit_storms = []
for _, s in storms.iterrows():
    t0, t1 = s.onset - pd.Timedelta("4D"), s.onset + pd.Timedelta("7D")
    if t1 > omni_end or Q[t0:t1].isna().mean() > 0.1:
        continue
    if (rel.storm_id == s.storm_id).sum() < 200:
        continue
    fit_storms.append(s.storm_id)
res["H2_storms"] = [int(x) for x in fit_storms]


def _response(storm_id, tau_h):
    """Unit-α hourly δ series for one storm and its cumulative sum."""
    s = storms[storms.storm_id == storm_id].iloc[0]
    t0 = s.onset - pd.Timedelta("4D")
    q = Q[t0:s.onset + pd.Timedelta("7D")].fillna(0)
    qpre = Q[t0:s.onset - pd.Timedelta("1D")].median()
    qq = (q - qpre).values
    dec = np.exp(-1 / tau_h)
    dlt = np.zeros(len(qq))
    for k in range(1, len(qq)):
        dlt[k] = dlt[k - 1] * dec + qq[k - 1] * tau_h * (1 - dec)
    return q.index, np.r_[0, np.cumsum(dlt)]


TAUS = np.r_[np.arange(2, 24, 1), np.arange(24, 73, 3)].astype(float)
_cache = {}


def response(storm_id, tau_h):
    if (storm_id, tau_h) not in _cache:
        _cache[(storm_id, tau_h)] = _response(storm_id, tau_h)
    return _cache[(storm_id, tau_h)]


def storm_design(d, s_id):
    """Interval-averaged unit-α model response for every τ: array (len(TAUS), n)."""
    ok = np.isfinite(d.X.values) & (np.abs(d.X.values) < 3)
    d = d[ok]
    rows = []
    for tau in TAUS:
        idx, cs = response(s_id, tau)
        tt = idx.values.astype("datetime64[s]").astype("int64")
        a = np.clip(np.searchsorted(tt, d.t1.values.astype("datetime64[s]").astype("int64")), 0, len(tt) - 1)
        b = np.clip(np.searchsorted(tt, d.t2.values.astype("datetime64[s]").astype("int64")), 1, len(tt))
        rows.append((cs[b] - cs[np.minimum(a, b - 1)]) / np.maximum(b - a, 1))
    return np.array(rows), d.X.values


def fit(parts):
    """parts: list of (Xmat, y). Grid-search τ, α by least squares, robust SSE."""
    Xm = np.concatenate([p[0] for p in parts], axis=1)
    y = np.concatenate([p[1] for p in parts])
    best = None
    for k, tau in enumerate(TAUS):
        x = Xm[k]
        alpha = (x @ y) / (x @ x)
        r = y - alpha * x
        sse = np.sum(np.minimum(r ** 2, (3 * 1.4826 * np.median(np.abs(r))) ** 2))
        if best is None or sse < best[2]:
            best = (tau, alpha, sse, 1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2))
    return best


bands = [(350, 400), (400, 450), (450, 500), (500, 550), (550, 600), (600, 650), (650, 700)]
H2 = []
for lo_, hi_ in bands:
    d = rel[rel.alt.between(lo_, hi_) & rel.storm_id.isin(fit_storms)]
    parts = {s_id: storm_design(d[d.storm_id == s_id], s_id) for s_id in fit_storms}
    parts = {k: v for k, v in parts.items() if v[1].size > 0}
    ids = list(parts)
    tau, alpha, sse, r2 = fit([parts[k] for k in ids])
    bt = [fit([parts[k] for k in rng.choice(ids, len(ids))])[0] for _ in range(500)]
    H2.append(dict(band=f"{lo_}-{hi_}", h_mid=(lo_ + hi_) / 2, n=len(d), objects=d.norad.nunique(),
                   tau_h=tau, alpha=alpha, r2=r2, tau_ci=list(np.percentile(bt, [2.5, 97.5])),
                   tau_boot=bt))
    print(H2[-1]["band"], tau, round(alpha, 4), round(r2, 3), H2[-1]["tau_ci"], flush=True)
h2 = pd.DataFrame(H2)
h2.drop(columns="tau_boot").to_csv(f"{OUT}/h2_tau_by_band.csv", index=False)
reg = stats.linregress(h2.h_mid, h2.tau_h)
# bootstrap slope using per-band bootstrap draws
bs = [stats.linregress(h2.h_mid, [np.asarray(b)[k] for b in h2.tau_boot]).slope for k in range(500)]
res["H2"] = dict(table=h2.drop(columns="tau_boot").round(4).to_dict("records"),
                 slope_h_per_100km=reg.slope * 100, slope_p_two_sided=reg.pvalue,
                 slope_p_one_sided_negative=(reg.pvalue / 2 if reg.slope < 0 else 1 - reg.pvalue / 2),
                 slope_boot_ci_per100km=list(np.percentile(np.array(bs) * 100, [2.5, 97.5])))
res["H2"]["verdict"] = "supported" if res["H2"]["slope_p_one_sided_negative"] < 0.05 else "falsified (slope not significantly negative)"
# constrained bands only: R² ≥ 0.1 and bootstrap CI not reaching the grid edges
c = h2[(h2.r2 >= 0.1)]
rc = stats.linregress(c.h_mid, c.tau_h) if len(c) >= 3 else None
res["H2"]["constrained_bands"] = c.band.tolist()
if rc is not None:
    res["H2"]["constrained_slope_per100km"] = rc.slope * 100
    res["H2"]["constrained_p_one_sided"] = rc.pvalue / 2 if rc.slope < 0 else 1 - rc.pvalue / 2
json.dump(res, open(f"{OUT}/results_h1_h2.json", "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "all_storms_by_band"}, indent=1, default=float))
