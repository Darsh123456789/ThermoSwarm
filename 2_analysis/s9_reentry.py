"""Step 9 — re-entry date prediction at 7, 3 and 1 day lead (hindcast).

Objects: every object that re-entered during the window and has TLEs up to ≤ 1.5 d
before decay (first download's re-entry set + second download's low-altitude set).
Actual decay epoch: Space-Track decay message (Historical), else SATCAT DECAY date.

Model (same for both methods): circular orbit, da/dt = −B ρ √(μa), ρ = orbit
average of NRLMSIS 2.1 over 24 points along the orbit (object's inclination),
observed F10.7/ap (hindcast: both methods see the same drivers). Re-entry = 120 km.
B is fitted so the model reproduces the object's own decay over the 5 days before
issue (standard practice).

Methods:
  uncorrected : ρ = ρ_MSIS
  corrected   : ρ = ρ_MSIS · c(t), c = network density ratio over the last 1.5 d
                divided by its mean over the 5-d B-fit window (other objects only),
                relaxing to 1 with a 2-day e-fold (fixed before looking at results).
  persistence variant (sensitivity): c held constant.
"""
import json, gzip, numpy as np, pandas as pd, pymsis, warnings
from spaceweather import MsisDrivers

warnings.filterwarnings("ignore")
import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
MU, RE = 398600.4418e9, 6378.137e3
drv = MsisDrivers()
rng = np.random.default_rng(5)

# ---------- objects and actual decay epochs ----------
t1 = pd.read_parquet(f"{OUT}/tles_reentry.parquet")
t2 = pd.read_parquet(f"{OUT}/tles_clean_2.parquet")
tle = pd.concat([t1, t2]).drop_duplicates(["norad", "jd"]).sort_values(["norad", "jd"])
tle = tle[tle.ecc < 0.02]
dec = pd.read_csv(f"{ROOT}/data/decay.csv")
dec = dec[dec.MSG_TYPE == "Historical"]
dec["epoch"] = pd.to_datetime(dec.DECAY_EPOCH).dt.floor("D") + pd.Timedelta("12h")
dec = dec.groupby("NORAD_CAT_ID").epoch.last()
sc = pd.read_csv(f"{ROOT}/data2/selected_objects_2.csv").set_index("NORAD_CAT_ID")
sc_dec = pd.to_datetime(sc.DECAY, errors="coerce") + pd.Timedelta("12h")
truth = dec.combine_first(sc_dec).dropna()
truth_jd = truth.map(lambda t: t.value / 86400e9 + 2440587.5)

iv = pd.read_parquet(f"{OUT}/intervals.parquet", columns=["norad", "jd1", "jd2", "R", "alt"])
iv = iv[iv.alt < 500]
ivj2, ivR, ivN = iv.jd2.values, iv.R.values, iv.norad.values

LEADS = [7, 3, 1]
cases = []
for norad, g in tle.groupby("norad"):
    if norad not in truth_jd.index:
        continue
    jd_dec = truth_jd[norad]
    jd, a = g.jd.values, g.a_km.values * 1e3
    if jd.max() < jd_dec - 1.5 or jd.min() > jd_dec - 7 - 5:
        continue
    for L in LEADS:
        k = np.searchsorted(jd, jd_dec - L, side="right") - 1       # last TLE before issue
        if k < 0 or jd_dec - jd[k] > L + 0.5:
            continue
        k0 = np.searchsorted(jd, jd[k] - 5)                         # start of fit window
        if jd[k] - jd[k0] < 3 or a[k0] <= a[k]:
            continue
        m_rec = (ivN != norad) & (ivj2 <= jd[k]) & (ivj2 > jd[k] - 1.5)
        m_fit = (ivN != norad) & (ivj2 <= jd[k]) & (ivj2 > jd[k0])
        if m_rec.sum() < 20 or m_fit.sum() < 20:
            continue
        c0 = np.median(ivR[m_rec]) / np.median(ivR[m_fit])
        cases.append(dict(norad=norad, lead=L, jd_issue=jd[k], jd_fit0=jd[k0], a_fit0=a[k0], a_issue=a[k],
                          incl=g.incl.values[k], jd_dec=jd_dec, c0=c0))
C = pd.DataFrame(cases)
print(len(C), "cases from", C.norad.nunique(), "objects", flush=True)

NP = 24
u = np.linspace(0, 2 * np.pi, NP, endpoint=False)


def orbit_rho(jd, a, incl):
    """Orbit-averaged NRLMSIS density for arrays of cases (n,) → (n,)."""
    n = len(jd)
    lat = np.degrees(np.arcsin(np.sin(np.radians(incl))[:, None] * np.sin(u)[None, :]))
    lon = (np.degrees(np.arctan2(np.cos(np.radians(incl))[:, None] * np.sin(u), np.cos(u)))
           + (jd[:, None] * 360.98) % 360) % 360 - 180
    alt = np.repeat(((a - RE) / 1e3)[:, None], NP, 1)
    t = np.repeat(jd[:, None], NP, 1) + u[None, :] / (2 * np.pi) * 0.06
    unix = ((t - 2440587.5) * 86400).ravel()
    f107, f107a, aps = drv(unix)
    rho = pymsis.calculate((unix * 1e6).astype("int64").astype("datetime64[us]"), lon.ravel(), lat.ravel(),
                           np.clip(alt.ravel(), 80, 1000), f107s=f107, f107as=f107a, aps=aps, version=2.1)[:, 0]
    return rho.reshape(n, NP).mean(1)


def integrate(jd0, a0, incl, B, c0, efold, jd_stop, a_stop=RE + 120e3):
    """Vectorised decay. Returns final a (if jd_stop given) and time of reaching a_stop."""
    jd, a = jd0.copy(), a0.copy()
    hit = np.full(len(a), np.nan)
    active = np.ones(len(a), bool)
    while active.any():
        idx = np.where(active)[0]
        h = (a[idx] - RE) / 1e3
        dt = np.where(h > 250, 1 / 24, np.where(h > 180, 1 / 96, 1 / 480))     # days
        dt = np.minimum(dt, jd_stop[idx] - jd[idx]) if jd_stop is not None else dt
        rho = orbit_rho(jd[idx], a[idx], incl[idx])
        el = jd[idx] - jd0[idx]
        c = 1 + (c0[idx] - 1) * (np.exp(-el / efold) if efold else 1.0)
        a[idx] -= B[idx] * rho * c * np.sqrt(MU * a[idx]) * dt * 86400
        jd[idx] += dt
        done = a[idx] <= a_stop
        hit[idx[done]] = jd[idx[done]]
        stop = done | ((jd[idx] >= jd_stop[idx] - 1e-9) if jd_stop is not None else False) | (jd[idx] - jd0[idx] > 40)
        active[idx[stop]] = False
    return a, hit


# ---------- fit B over the 5-d window by one Newton/secant step (model linear in B) ----------
ones = np.ones(len(C))
a_end, _ = integrate(C.jd_fit0.values, C.a_fit0.values, C.incl.values, ones, ones, 0, C.jd_issue.values)
B = (C.a_fit0.values - C.a_issue.values) / (C.a_fit0.values - a_end)
for it in range(2):   # refine (decay accelerates, so not exactly linear)
    a_end, _ = integrate(C.jd_fit0.values, C.a_fit0.values, C.incl.values, B, ones, 0, C.jd_issue.values)
    B *= (C.a_fit0.values - C.a_issue.values) / np.maximum(C.a_fit0.values - a_end, 1)
C["B"] = B
print("B fitted", flush=True)

res = {}
for name, cc, ef in [("uncorrected", ones, 0), ("corrected", C.c0.values, 2.0), ("persistence", C.c0.values, None)]:
    if ef is None:
        _, hit = integrate(C.jd_issue.values, C.a_issue.values, C.incl.values, B, cc, 0, None)
    else:
        _, hit = integrate(C.jd_issue.values, C.a_issue.values, C.incl.values, B, cc, ef, None)
    C[f"err_{name}"] = hit - C.jd_dec.values            # days, + = predicted late
    print(name, "done", flush=True)
C.to_parquet(f"{OUT}/reentry_cases.parquet")

for L in LEADS:
    d = C[C.lead == L].dropna(subset=["err_uncorrected", "err_corrected"])
    eu, ec, ep = d.err_uncorrected.abs(), d.err_corrected.abs(), d.err_persistence.abs()
    objs = d.norad.values
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(d), len(d))
        bs.append(np.median(eu.values[k]) - np.median(ec.values[k]))
    res[f"lead_{L}d"] = dict(
        n=len(d), objects=int(d.norad.nunique()),
        median_abs_err_uncorrected_d=float(eu.median()), median_abs_err_corrected_d=float(ec.median()),
        median_abs_err_persistence_d=float(ep.median()),
        improvement_days=float(eu.median() - ec.median()), improvement_ci=list(np.percentile(bs, [2.5, 97.5])),
        mean_abs_uncorrected=float(eu.mean()), mean_abs_corrected=float(ec.mean()),
        frac_corrected_better=float((ec < eu).mean()),
        median_signed_uncorrected=float(d.err_uncorrected.median()), median_signed_corrected=float(d.err_corrected.median()),
        median_c0=float(d.c0.median()), frac_c0_off_by_10pct=float((np.abs(d.c0 - 1) > 0.1).mean()))
# storm subset: issue within 3 d before a Kp>=6- storm peak or during it
storms = pd.read_csv(f"{OUT}/storms.csv", parse_dates=["peak_time"])
pk = storms.peak_time.map(lambda t: t.value / 86400e9 + 2440587.5).values
C["storm_affected"] = [bool(np.any((p >= ji - 0.5) & (p <= jd_))) for ji, jd_ in zip(C.jd_issue, C.jd_dec) for p in [pk]]
for L in LEADS:
    d = C[(C.lead == L) & C.storm_affected].dropna(subset=["err_uncorrected", "err_corrected"])
    res[f"lead_{L}d_storm"] = dict(n=len(d), unc=float(d.err_uncorrected.abs().median()) if len(d) else None,
                                   cor=float(d.err_corrected.abs().median()) if len(d) else None)
json.dump(res, open(f"{OUT}/results_reentry.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
