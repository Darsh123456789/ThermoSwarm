"""Step 8 — independent validation against Swarm A/B/C precise-orbit densities.

For each Swarm sample (2-min cadence, valid flag 0, fill values removed) evaluate
NRLMSIS 2.1 at the same time/position with the same drivers used in the inversion.
Swarm ratio R_sw = ρ_Swarm / ρ_MSIS. Averaged over the same TLE intervals, then
compared with TLE-derived R of objects within ±25 km of the Swarm altitude.

Two comparisons:
 (a) absolute:  median R_sw (does NRLMSIS have a bias at all?)
 (b) relative:  R_sw normalised by its own 30-day quiet-time median, i.e. the same
                quantity the TLE method measures after quiet-time B calibration.
"""
import json, numpy as np, pandas as pd, pymsis
from scipy import stats
from spaceweather import MsisDrivers, load_gfz, kp_label

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
SW = f"{ROOT}/swarm"
drv = MsisDrivers()
three, daily = load_gfz()
rng = np.random.default_rng(3)
res = {}

frames = []
for s in "ABC":
    d = pd.read_csv(f"{SW}/swarm{s}_density.csv.gz")
    d["time"] = pd.to_datetime(d.time).dt.tz_localize(None)
    d = d[(d.validity_flag == 0) & (d.density > 1e-16) & (d.density < 1e-9)]
    unix = d.time.values.astype("datetime64[s]").astype("int64").astype(float)
    f107, f107a, aps = drv(unix)
    rho = pymsis.calculate(d.time.values.astype("datetime64[us]"), d.Longitude_GD.values,
                           d.Latitude_GD.values, d.Height_GD.values / 1e3, f107s=f107,
                           f107as=f107a, aps=aps, version=2.1)[:, 0]
    d["rho_msis"] = rho
    d["sat"] = s
    d["alt"] = d.Height_GD / 1e3
    frames.append(d[["time", "sat", "alt", "Latitude_GD", "local_solar_time", "density", "rho_msis"]])
    print(s, len(d), "valid samples", flush=True)
sw = pd.concat(frames).sort_values("time").reset_index(drop=True)
sw["R_abs"] = sw.density / sw.rho_msis
sw.to_parquet(f"{OUT}/swarm_msis.parquet")

# ap for quiet flag
ap = three.ap
sw["ap24max"] = ap.rolling(8, min_periods=1).max().reindex(sw.time.dt.floor("3h")).values
res["absolute"] = {s: dict(median=float(g.R_abs.median()),
                           quiet_median=float(g.R_abs[g.ap24max <= 15].median()),
                           alt_median=float(g.alt.median()))
                   for s, g in sw.groupby("sat")}

# ---- relative normalisation, identical in spirit to TLE calibration ----
# orbit/3-h means first (TLE B is an orbit-average quantity)
sw["bin"] = sw.time.dt.floor("3h")
b3 = sw.groupby(["sat", "bin"]).agg(dens=("density", "mean"), msis=("rho_msis", "mean"),
                                    alt=("alt", "mean"), ap24max=("ap24max", "max")).reset_index()
b3["Rabs"] = b3.dens / b3.msis
out = []
for s, g in b3.groupby("sat"):
    g = g.sort_values("bin").copy()
    t = g.bin.values.astype("datetime64[h]").astype("int64") / 24.0
    q = (g.ap24max <= 15).values
    base = np.full(len(g), np.nan)
    for k, tk in enumerate(t):
        m = q & (np.abs(t - tk) <= 30)
        if m.sum() >= 40:
            base[k] = np.median(g.Rabs.values[m])
    g["Rrel"] = g.Rabs / base
    out.append(g)
b3 = pd.concat(out)
b3.to_parquet(f"{OUT}/swarm_3h.parquet")

# ---- match to TLE intervals ----
iv = pd.read_parquet(f"{OUT}/intervals.parquet")
iv = iv[iv.t2 <= sw.time.max()]
rows = []
for s, g in b3.groupby("sat"):
    g = g.dropna(subset=["Rrel"])
    alt_s = g.alt.median()
    tt = g.bin.values.astype("datetime64[s]").astype("int64")
    cs_r = np.r_[0, np.cumsum(g.Rrel.values)]
    cs_a = np.r_[0, np.cumsum(g.Rabs.values)]
    sel = iv[(iv.alt - alt_s).abs() <= 25]
    a = np.searchsorted(tt, sel.t1.values.astype("datetime64[s]").astype("int64"))
    b = np.searchsorted(tt, sel.t2.values.astype("datetime64[s]").astype("int64"))
    ok = (b - a) >= 6
    d = sel[ok].copy()
    d["sw_rel"] = (cs_r[b[ok]] - cs_r[a[ok]]) / (b[ok] - a[ok])
    d["sw_abs"] = (cs_a[b[ok]] - cs_a[a[ok]]) / (b[ok] - a[ok])
    d["sat"] = s
    rows.append(d)
m = pd.concat(rows)
m.to_parquet(f"{OUT}/tle_vs_swarm.parquet")

# daily network median vs Swarm daily
m["day"] = m.tmid.dt.floor("D")
dd = m.groupby(["sat", "day"]).agg(tle=("R", "median"), sw=("sw_rel", "median"), n=("R", "size")).reset_index()
dd = dd[dd.n >= 5]


def corr_block(x, y):
    r, p = stats.pearsonr(x, y)
    rs = stats.spearmanr(x, y).statistic
    lr = np.log(x / y)
    return dict(n=len(x), pearson_r=float(r), spearman=float(rs), median_ratio=float(np.median(x / y)),
                rms_log_diff=float(np.sqrt(np.mean(lr ** 2))), mad_pct=float(100 * np.median(np.abs(x / y - 1))))


res["matched_intervals"] = corr_block(m.R.values, m.sw_rel.values)
res["matched_intervals"]["objects"] = int(m.norad.nunique())
res["daily_network"] = corr_block(dd.tle.values, dd.sw.values)
res["daily_by_sat"] = {s: corr_block(g.tle.values, g.sw.values) for s, g in dd.groupby("sat")}
res["per_object_scatter_vs_swarm_pct"] = float(100 * 1.4826 * np.median(np.abs(m.R / m.sw_rel - 1)))

# bootstrap CI for daily r (resample days)
bs = []
for _ in range(1000):
    k = rng.integers(0, len(dd), len(dd))
    bs.append(stats.pearsonr(dd.tle.values[k], dd.sw.values[k])[0])
res["daily_network"]["r_ci"] = list(np.percentile(bs, [2.5, 97.5]))

# ---- storm-time check (the H1 question with independent truth) ----
storms = pd.read_csv(f"{OUT}/storms.csv", parse_dates=["onset", "end", "peak_time"])
st_rows = []
for _, s in storms.iterrows():
    if s.peak_time > sw.time.max():
        continue
    g = b3[(b3.bin >= s.peak_time - pd.Timedelta("12h")) & (b3.bin <= s.peak_time + pd.Timedelta("12h"))]
    pre = b3[(b3.bin >= s.onset - pd.Timedelta("4D")) & (b3.bin <= s.onset - pd.Timedelta("1D"))]
    tl = m[(m.t1 <= s.peak_time) & (m.t2 >= s.peak_time) & ((m.jd2 - m.jd1) <= 2)]
    # Swarm averaged over the same windows as those TLE intervals
    st_rows.append(dict(storm_id=int(s.storm_id), peak=str(s.peak_time.date()), kp=kp_label(s.kp_max),
                        kp_max=float(s.kp_max),
                        swarm_peak_abs=float(g.Rabs.max()), swarm_peak12h_abs=float(g.Rabs.median()),
                        swarm_pre_abs=float(pre.Rabs.median()),
                        swarm_peak_rel=float(g.Rrel.median()),
                        swarm_same_window_rel=float(tl.sw_rel.median()) if len(tl) else np.nan,
                        tle_same_window=float(tl.R.median()) if len(tl) else np.nan, n_tle=len(tl)))
sr = pd.DataFrame(st_rows)
sr.to_csv(f"{OUT}/swarm_storms.csv", index=False)
k7 = sr[sr.kp_max >= 7 - 1e-6]
res["storms"] = st_rows
res["kp7_swarm_peak_abs_median"] = float(k7.swarm_peak12h_abs.median())
res["kp7_swarm_max3h_abs_median"] = float(k7.swarm_peak_abs.median())

# smoothing: how much does a 1-2 day TLE window dilute the Swarm 3-h storm peak?
dil = []
for _, s in k7.iterrows():
    g = b3[(b3.bin >= pd.Timestamp(s.peak) - pd.Timedelta("1D")) & (b3.bin <= pd.Timestamp(s.peak) + pd.Timedelta("2D"))]
    for sat, gg in g.groupby("sat"):
        if gg.Rrel.notna().sum() < 8:
            continue
        pk = gg.Rrel.max()
        sm = gg.Rrel.rolling(12, center=True, min_periods=6).mean().max()  # 36-h mean, ~TLE interval
        dil.append(dict(storm=s.peak, sat=sat, peak3h=pk, smoothed36h=sm))
dil = pd.DataFrame(dil)
res["smoothing"] = dict(median_peak3h_rise=float((dil.peak3h - 1).median()),
                        median_smoothed36h_rise=float((dil.smoothed36h - 1).median()))
json.dump(res, open(f"{OUT}/results_swarm.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "storms"}, indent=1, default=float))
print(sr.to_string())
