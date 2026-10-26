"""Step 3 — quiet-time ballistic-coefficient calibration and density ratios.

Quiet interval: max 3-hourly ap ≤ 15 from 24 h before t1 to t2.
B(t): running median of quiet f within ±30 days (≥ 8 quiet intervals).
R = f / B  =  ρ_obs / ρ_MSIS  (relative to the model's own quiet-time level).
Per-measurement uncertainty: σ_R = 1.4826·MAD of quiet R for that object
(empirical scatter from TLE noise and B variation).
"""
import numpy as np, pandas as pd
import spaceweather as sw

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"

iv = pd.read_parquet(f"{OUT}/intervals_raw.parquet")
three, daily = sw.load_gfz()
ap = three.ap
cum = ap.cumsum()

# max/mean ap over each interval (and 24 h lead-in for the quiet flag)
apv, apt = ap.values, ap.index.values.astype("datetime64[s]").astype("int64")
t1 = iv.t1.values.astype("datetime64[s]").astype("int64")
t2 = iv.t2.values.astype("datetime64[s]").astype("int64")
lo = np.searchsorted(apt, t1 - 86400 - 10800)
lo2 = np.searchsorted(apt, t1 - 10800)
hi = np.searchsorted(apt, t2, side="right")
from numpy.lib.stride_tricks import sliding_window_view
iv["ap_max_lead"] = [apv[a:b].max() for a, b in zip(lo, hi)]
iv["ap_mean"] = [apv[a:b].mean() for a, b in zip(lo2, hi)]
iv["ap_max"] = [apv[a:b].max() for a, b in zip(lo2, hi)]
iv["quiet"] = iv.ap_max_lead <= 15

# sanity: positive decay, finite
n0 = len(iv)
iv = iv[np.isfinite(iv.f) & (iv.I > 0)].copy()


def calibrate(g):
    g = g.sort_values("jd1").copy()
    q = g[g.quiet & (g.f > 0)]
    B = np.full(len(g), np.nan)
    if len(q) >= 8:
        qt, qf = q.jd1.values, q.f.values
        for k, t in enumerate(g.jd1.values):
            m = np.abs(qt - t) <= 30
            if m.sum() >= 8:
                B[k] = np.median(qf[m])
    g["B"] = B
    g["R"] = g.f / g.B
    qr = g.R[g.quiet].dropna()
    g["sigR"] = 1.4826 * np.median(np.abs(qr - qr.median())) if qr.notna().sum() >= 8 else np.nan
    return g


iv = pd.concat([calibrate(g) for _, g in iv.groupby("norad")])
iv = iv[iv.R.notna()]
# drop objects whose quiet-time scatter is too large to be informative
iv = iv[iv.sigR < 0.5]
iv.to_parquet(f"{OUT}/intervals.parquet")

storms = sw.find_storms(three[three.index >= "2025-10-01"])
storms.to_csv(f"{OUT}/storms.csv", index=False)

# --- storm-time samples: intervals ≤ 2 d long that contain the storm peak ---
rows = []
for _, s in storms.iterrows():
    m = (iv.t1 <= s.peak_time) & (iv.t2 >= s.peak_time) & ((iv.jd2 - iv.jd1) <= 2.0)
    d = iv[m].copy()
    d["storm_id"], d["kp_max"] = s.storm_id, s.kp_max
    rows.append(d)
st = pd.concat(rows)
st.to_parquet(f"{OUT}/storm_samples.parquet")

print("intervals:", n0, "->", len(iv), "objects:", iv.norad.nunique())
print("quiet fraction:", iv.quiet.mean().round(3))
print("quiet R median / MAD:", iv.R[iv.quiet].median().round(3),
      (1.4826 * (iv.R[iv.quiet] - iv.R[iv.quiet].median()).abs().median()).round(3))
print("median per-object sigR:", iv.groupby("norad").sigR.first().median().round(3))
print("alt range (5–95%):", iv.alt.quantile([0.05, 0.95]).round(0).tolist(),
      "min/max", iv.alt.min().round(0), iv.alt.max().round(0))
print(st.groupby(["storm_id", "kp_max"]).R.agg(["count", "median"]).round(3).to_string())
