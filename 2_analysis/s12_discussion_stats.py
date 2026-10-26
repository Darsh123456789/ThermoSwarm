"""Step 12 — supporting statistics quoted in Sections 6 and 7 of the paper.

Reads outputs of steps 3-9 and prints:
  6.1  agreement with Swarm vs number of objects averaged per satellite-day
  6.2  slow (15-day) vs fast parts of the daily network ratio; storm-time ratio vs
       storm strength and vs pre-storm baseline across the 18 storms
  6.3/7 TLE interval lengths, coverage below 300 km, width of the H1 intervals
  6.4  forecast error by altitude band, error tails, per-storm gains vs storm traits,
       re-entry cases with a large correction
Writes results_discussion.json.
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
res = {}

# ---- 6.1 averaging: median |network/Swarm - 1| for k objects per satellite-day ----
rng = np.random.default_rng(0)
m = pd.read_parquet(f"{OUT}/tle_vs_swarm.parquet", columns=["norad", "tmid", "R", "sw_rel", "sat"])
m["day"] = m.tmid.dt.floor("D")
avg = {}
for k in [1, 3, 5, 10, 20]:
    errs = []
    for _, g in m.groupby(["sat", "day"]):
        if len(g) < 20:
            continue
        for _ in range(3):
            sub = g.sample(k, random_state=int(rng.integers(1e9)))
            errs.append(abs(sub.R.median() / g.sw_rel.median() - 1))
    avg[k] = float(100 * np.median(errs))
res["swarm_diff_pct_vs_objects"] = avg
res["objects_per_satellite_day_median"] = float(m.groupby(["sat", "day"]).size().median())

iv = pd.read_parquet(f"{OUT}/intervals.parquet").reset_index(drop=True)
alt_obj = iv.groupby("norad").alt.median()
res["objects_per_100km_band"] = {str(k): int(v) for k, v in
                                 pd.cut(alt_obj, [190, 300, 400, 500, 600, 700, 800]).value_counts().sort_index().items()}

# ---- 6.2 slow vs fast parts of the daily network ratio (300-550 km) ----
s = iv[iv.alt.between(300, 550)]
d = s.groupby(s.tmid.dt.floor("D")).R.median().asfreq("D").interpolate()
slow = d.rolling(15, center=True, min_periods=8).median()
x = pd.concat([d, slow], axis=1).dropna(); x.columns = ["d", "s"]
res["slow_variance_fraction"] = float(x.s.var() / x.d.var())
res["slow_p5_p95"] = x.s.quantile([0.05, 0.95]).tolist()
res["slow_min"] = [float(x.s.min()), str(x.s.idxmin().date())]
res["slow_max"] = [float(x.s.max()), str(x.s.idxmax().date())]

# ---- 6.2 storm-time ratio vs storm strength and pre-storm baseline ----
rel = json.load(open(f"{OUT}/results_h1_h2.json"))["relative"]
pre = pd.Series({int(k): v for k, v in rel["pre_storm_baseline_R"].items()})
X = pd.Series({int(k): v for k, v in rel["per_storm_X"].items()})
st = pd.read_parquet(f"{OUT}/storm_samples.parquet", columns=["storm_id", "R", "alt", "kp_max"]).reset_index(drop=True)
j = pd.concat([pre.rename("pre"), st.groupby("storm_id").R.median().rename("peak"), X.rename("X"),
               st.groupby("storm_id").kp_max.first().rename("kp")], axis=1).dropna()
res["storms"] = len(j)
res["peak_vs_kp_spearman"] = list(stats.spearmanr(j.peak, j.kp))
res["peak_vs_pre_pearson"] = list(stats.pearsonr(j.pre, j.peak))
res["peak_vs_pre_slope"] = float(stats.linregress(j.pre, j.peak).slope)
res["X_vs_kp_spearman"] = list(stats.spearmanr(j.X, j.kp))
res["highest_storm_ratio"] = [float(j.peak.max()), int(j.peak.idxmax())]

# ---- 6.3 / 7 interval lengths, coverage, H1 interval widths ----
dur = iv.jd2 - iv.jd1
res["interval_days_p10_p50_p90"] = dur.quantile([0.1, 0.5, 0.9]).tolist()
res["frac_intervals_le_1.1d"] = float((dur <= 1.1).mean())
res["storm_samples_below_300km"] = [int((st.alt < 300).sum()), len(st)]
h1 = json.load(open(f"{OUT}/results_h1_h2.json"))["H1"]
res["H1_ci_width_ratio_storm_over_object"] = float(np.diff(h1["ci_storm"])[0] / np.diff(h1["ci_obj"])[0])

# ---- 6.4 forecast by altitude band, tails, per-storm gains ----
f = pd.read_parquet(f"{OUT}/forecast_results.parquet").reset_index(drop=True)
S = f[f.stormtime].copy()


def red(dd):
    return float(1 - np.abs(dd.e_gb_phys).median() / np.abs(dd.e_uncorrected).median())


S["band"] = pd.cut(S.alt, [200, 300, 400, 500, 600, 700, 800])
res["forecast_by_band"] = {str(b): dict(n=len(dd), uncorrected_median_km=float(np.abs(dd.e_uncorrected).median()),
                                        reduction_median=red(dd))
                           for b, dd in S.groupby("band", observed=True)}
res["error_quantiles_km"] = {str(q): [float(np.abs(S.e_uncorrected).quantile(q)), float(np.abs(S.e_gb_phys).quantile(q))]
                             for q in (0.5, 0.9, 0.99)}
big = np.abs(S.e_uncorrected) > 10
res["windows_over_10km"] = dict(fraction=float(big.mean()),
                                improved=float((np.abs(S.e_gb_phys[big]) < np.abs(S.e_uncorrected[big])).mean()),
                                median_reduction=red(S[big]))
res["fraction_windows_improved"] = float((np.abs(S.e_gb_phys) < np.abs(S.e_uncorrected)).mean())
h3 = json.load(open(f"{OUT}/results_h3.json"))
ps = pd.Series({int(k): v for k, v in h3["per_storm_reduction_gbphys_median"].items()})
storms = pd.read_csv(f"{OUT}/storms.csv").set_index("storm_id")
jj = pd.concat([ps.rename("red"), storms.kp_max, pre.rename("pre"), S.groupby("group").size().rename("n")], axis=1).dropna()
res["per_storm_gain_vs_kp_spearman"] = list(stats.spearmanr(jj.red, jj.kp_max))
res["per_storm_gain_vs_pre_spearman"] = list(stats.spearmanr(jj.red, jj.pre))
res["windows_in_failed_storms"] = {int(k): int(v) for k, v in jj[jj.red < 0].n.items()}

# ---- 6.4 re-entry cases with a large correction ----
C = pd.read_parquet(f"{OUT}/reentry_cases.parquet").dropna(subset=["err_uncorrected", "err_corrected"])
bigc = (C.c0 - 1).abs() > 0.1
res["reentry_large_correction"] = dict(n=int(bigc.sum()), of=len(C),
                                       median_abs_uncorrected=float(C.err_uncorrected[bigc].abs().median()),
                                       median_abs_corrected=float(C.err_corrected[bigc].abs().median()),
                                       corrected_better=float((C.err_corrected[bigc].abs() < C.err_uncorrected[bigc].abs()).mean()))
ree = json.load(open(f"{OUT}/results_reentry.json"))["lead_7d_storm"]
res["reentry_7d_storm"] = ree

json.dump(res, open(f"{OUT}/results_discussion.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
