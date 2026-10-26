"""Step 6 — 72-h density-correction forecast, storm-wise holdout, along-track error (H3).

Everything at issue time t0 is causal:
  B_c   = median f over the object's quiet intervals in [t0 − 30 d, t0)
  R72   = f72 / B_c                     (target: mean density ratio over next 72 h)
  R_last= f / B_c of the last interval ending ≤ t0
Model ladder (prediction of R72):
  0 uncorrected  : R = 1  (NRLMSIS 2.1 with quiet-calibrated B)
  1 persistence  : R = R_last
  2 physics      : R_last · (1 + α δ̄_fc) / (1 + α δ̄_last), δ from the relaxation model
                   driven by Newell coupling (τ, α fit on training storms only)
  3 GB           : XGBoost on drivers + R_last + altitude
  4 GB + physics : 3 plus the physics prediction and δ state
Along-track error of a prediction: e = a · (B_c·R_pred − f72) · J  (km).
Groups: each window assigned to the nearest storm (storms 3–18, OMNI coverage);
4-fold GroupKFold, so each storm is predicted by a model that never saw it.
"""
import json, numpy as np, pandas as pd, warnings
import xgboost as xgb
from sklearn.model_selection import GroupKFold
import spaceweather as sw

warnings.filterwarnings("ignore")
import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
rng = np.random.default_rng(7)

w = pd.read_parquet(f"{OUT}/windows72.parquet")
iv = pd.read_parquet(f"{OUT}/intervals.parquet")
storms = pd.read_csv(f"{OUT}/storms.csv", parse_dates=["onset", "end", "peak_time"])
three, daily = sw.load_gfz()
omni = sw.load_omni()
Q = omni.newell.interpolate(limit=12)
omni_end = Q.dropna().index.max()

# ---------- causal per-object quantities ----------
rows = []
for norad, g in w.groupby("norad"):
    q = iv[(iv.norad == norad)].sort_values("jd2")
    if q.empty:
        continue
    qj2, qj1, qf, qq = q.jd2.values, q.jd1.values, q.f.values, q.quiet.values
    for k, t in zip(g.index, g.jd1.values):
        m = qq & (qj2 <= t) & (qj1 >= t - 30)
        if m.sum() < 5:
            continue
        B = np.median(qf[m])
        j = np.searchsorted(qj2, t, side="right") - 1
        if j < 0 or t - qj2[j] > 3:
            continue
        m7 = (qj2 <= t) & (qj2 >= t - 7)
        rows.append((k, B, qf[j] / B, np.median(qf[m7]) / B, qj1[j], qj2[j]))
c = pd.DataFrame(rows, columns=["idx", "B", "R_last", "R_base7", "last_jd1", "last_jd2"]).set_index("idx")
w = w.join(c, how="inner")
w["R72"] = w.f72 / w.B
w = w[(w.t0 >= storms.onset.min() - pd.Timedelta("10D")) & (w.t0 <= omni_end - pd.Timedelta("3D"))]
w = w[np.isfinite(w.R72) & (w.R72 > 0.2) & (w.R72 < 5) & (w.R_last > 0.2) & (w.R_last < 5)]

# ---------- global driver features (hourly) ----------
hr = pd.DataFrame(index=pd.date_range("2025-09-01", omni_end, freq="1h"))
hr["Q"] = Q.reindex(hr.index)
ap3 = three.ap.reindex(hr.index, method="ffill")
kp3 = three.kp.reindex(hr.index, method="ffill")
for h in (3, 6, 24, 48):
    hr[f"Q_{h}h"] = hr.Q.rolling(h, min_periods=1).mean()
hr["ap_now"] = ap3
hr["ap_max24"] = ap3.rolling(24, min_periods=1).max()
hr["kp_max24"] = kp3.rolling(24, min_periods=1).max()
hr["symh_min24"] = omni.symh.reindex(hr.index).rolling(24, min_periods=1).min()
fd = daily.f107.shift(1).reindex(hr.index.floor("D")).values
hr["f107"] = fd
hr["f107_dev"] = fd - daily.f107a.reindex(hr.index.floor("D")).values
on = (kp3 >= 14 / 3 - 1e-6)
last_on = pd.Series(np.where(on, hr.index.values.astype("datetime64[h]").astype("int64"), np.nan), index=hr.index).ffill()
hr["h_since_storm"] = np.minimum(hr.index.values.astype("datetime64[h]").astype("int64") - last_on, 240).fillna(240)
Qp = (hr.Q - hr.Q.rolling(24 * 30, min_periods=24 * 5).median()).fillna(0).values

TAUS = np.array([3, 5, 8, 11, 15, 20, 30, 45.0])
hrt = hr.index.values.astype("datetime64[h]").astype("int64")
DEL = {}
for tau in TAUS:  # unit-α δ(t) on the hourly grid, and its cumulative sum
    dec = np.exp(-1 / tau); d = np.zeros(len(Qp))
    for k in range(1, len(Qp)):
        d[k] = d[k - 1] * dec + Qp[k - 1] * tau * (1 - dec)
    DEL[tau] = (d, np.r_[0, np.cumsum(d)])


def hidx(jd):
    return np.clip(np.searchsorted(hrt, ((jd - 2440587.5) * 24).astype("int64")), 0, len(hrt) - 1)


i0 = hidx(w.jd1.values); la, lb = hidx(w.last_jd1.values), hidx(w.last_jd2.values)
lb = np.maximum(lb, la + 1)
T = 72.0
for tau in TAUS:
    d, cs = DEL[tau]
    w[f"d0_{tau}"] = d[i0]
    w[f"dfc_{tau}"] = d[i0] * tau / T * (1 - np.exp(-T / tau))
    w[f"dlast_{tau}"] = (cs[lb] - cs[la]) / (lb - la)
feat_drv = ["Q_3h", "Q_6h", "Q_24h", "Q_48h", "ap_now", "ap_max24", "kp_max24", "symh_min24",
            "f107", "f107_dev", "h_since_storm"]
w = w.join(pd.DataFrame(hr[feat_drv].values[i0], index=w.index, columns=feat_drv))

# ---------- storm groups ----------
gs = storms[storms.peak_time <= omni_end].reset_index(drop=True)
pk = gs.peak_time.values.astype("datetime64[s]").astype("int64")
t0s = w.t0.values.astype("datetime64[s]").astype("int64")
w["group"] = gs.storm_id.values[np.abs(t0s[:, None] - pk[None, :]).argmin(1)]
sm = np.zeros(len(w), bool)
for _, s in gs.iterrows():
    sm |= ((w.t0 >= s.onset - pd.Timedelta("1D")) & (w.t0 <= s.end + pd.Timedelta("2D"))).values
w["stormtime"] = sm


def phys_pred(d, tau, alpha):
    return d.R_last * (1 + alpha * d[f"dfc_{tau}"]) / (1 + alpha * d[f"dlast_{tau}"])


def fit_phys(d):
    best = None
    y = np.log(d.R72)
    for tau in TAUS:
        for alpha in np.linspace(0, 0.3, 31):
            p = phys_pred(d, tau, alpha)
            ok = p > 0.05
            e = np.median(np.abs(y[ok] - np.log(p[ok])))
            if best is None or e < best[2]:
                best = (tau, alpha, e)
    return best[:2]


feats_gb = ["alt", "R_last", "R_base7"] + feat_drv
gkf = GroupKFold(n_splits=4)
pred = {m: np.full(len(w), np.nan) for m in ["uncorrected", "persistence", "physics", "gb", "gb_phys"]}
phys_params, models = [], []
for fold, (tr, te) in enumerate(gkf.split(w, groups=w.group)):
    dtr, dte = w.iloc[tr], w.iloc[te]
    tau, alpha = fit_phys(dtr.sample(min(len(dtr), 40000), random_state=fold))
    phys_params.append((float(tau), float(alpha)))
    for d in (dtr, dte):
        pass
    p_tr = phys_pred(dtr, tau, alpha).clip(0.1, 10)
    p_te = phys_pred(dte, tau, alpha).clip(0.1, 10)
    pred["uncorrected"][te] = 1.0
    pred["persistence"][te] = dte.R_last
    pred["physics"][te] = p_te
    Xtr = dtr[feats_gb].copy(); Xte = dte[feats_gb].copy()
    sub = rng.choice(len(Xtr), min(len(Xtr), 80000), replace=False)
    params = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, n_jobs=2, objective="reg:absoluteerror")
    m1 = xgb.XGBRegressor(**params).fit(Xtr.iloc[sub], np.log(dtr.R72.values[sub]))
    pred["gb"][te] = np.exp(m1.predict(Xte))
    Xtr["phys"], Xte["phys"] = np.log(p_tr), np.log(p_te)
    Xtr["delta0"], Xte["delta0"] = alpha * dtr[f"d0_{tau}"], alpha * dte[f"d0_{tau}"]
    m2 = xgb.XGBRegressor(**params).fit(Xtr.iloc[sub], np.log(dtr.R72.values[sub]))
    pred["gb_phys"][te] = np.exp(m2.predict(Xte))
    models.append((m2, Xte.sample(min(2000, len(Xte)), random_state=0)))
    print("fold", fold, "storms", sorted(dte.group.unique()), "phys τ, α:", tau, alpha, flush=True)

# ---------- along-track error ----------
res = {"n_windows": len(w), "n_objects": int(w.norad.nunique()), "n_storm_windows": int(w.stormtime.sum()),
       "phys_params_per_fold": phys_params, "groups": [int(x) for x in sorted(w.group.unique())]}
for m, p in pred.items():
    w[f"e_{m}"] = w.a_km * (w.B * p - w.f72) * w.J
    w[f"lr_{m}"] = np.abs(np.log(p) - np.log(w.R72))
w.to_parquet(f"{OUT}/forecast_results.parquet")


def summary(d):
    out = {}
    for m in pred:
        e = np.abs(d[f"e_{m}"])
        out[m] = dict(median_km=float(e.median()), rms_km=float(np.sqrt((e ** 2).mean())),
                      p90_km=float(e.quantile(0.9)), mae_logR=float(d[f"lr_{m}"].mean()))
    return out


res["all"] = summary(w)
res["stormtime"] = summary(w[w.stormtime])
S = w[w.stormtime]
grp = {g: d for g, d in S.groupby("group")}


def red(d, a="uncorrected", b="gb_phys", stat="median"):
    ea, eb = np.abs(d[f"e_{a}"]), np.abs(d[f"e_{b}"])
    return 1 - (eb.median() / ea.median() if stat == "median" else np.sqrt((eb ** 2).mean() / (ea ** 2).mean()))


for b in ["persistence", "physics", "gb", "gb_phys"]:
    for stat in ["median", "rms"]:
        boots = []
        keys = list(grp)
        for _ in range(1000):
            d = pd.concat([grp[k] for k in rng.choice(keys, len(keys))])
            boots.append(red(d, b=b, stat=stat))
        res[f"reduction_{b}_{stat}"] = dict(value=float(red(S, b=b, stat=stat)),
                                           ci=list(np.percentile(boots, [2.5, 97.5])))
res["per_storm_reduction_gbphys_median"] = {int(k): round(float(red(d)), 3) for k, d in grp.items()}
v = res["reduction_gb_phys_median"]
res["H3_verdict"] = ("supported (≥30%)" if v["value"] >= 0.30 and v["ci"][0] > 0.10 else
                     "falsified (<10%)" if v["value"] < 0.10 else "partially supported (10–30%)")

# SHAP
import shap
imp = []
for m2, Xs in models:
    sv = shap.TreeExplainer(m2).shap_values(Xs)
    imp.append(pd.Series(np.abs(sv).mean(0), index=Xs.columns))
res["shap_mean_abs"] = pd.concat(imp, axis=1).mean(1).sort_values(ascending=False).round(4).to_dict()
json.dump(res, open(f"{OUT}/results_h3.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
