"""Step 5a — 72-hour prediction windows for the forecast / along-track test.

For each object, TLE pairs separated by 66–78 h (issue times ≥ 24 h apart) give
  f72 = Δn_obs / I        (observed B·R over the window)
  J   = ∫0^T (T − t) g(t) dt   so that the drag-induced along-track phase error of a
        prediction using B·R_pred is  Δθ = (B·R_pred − f72) · J,  error = a · Δθ.
g(t) = (3/2) n ρ_MSIS v_rel²/v along the SGP4 trajectory (as in step 2).
"""
import numpy as np, pandas as pd, warnings
from multiprocessing import Pool
from sgp4.api import Satrec
from sgp4.propagation import gstime
import pymsis
from s2_inversion import drv, RE, FLAT, OMEGA

warnings.filterwarnings("ignore")
import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
NS = 144


def integrals(l1, l2, jd1, jd2):
    s = Satrec.twoline2rv(l1, l2)
    jd = jd1 + (np.arange(NS) + 0.5) * (jd2 - jd1) / NS
    e, r, v = s.sgp4_array(np.floor(jd), jd - np.floor(jd))
    if np.any(e):
        return None
    r, v = r * 1e3, v * 1e3
    rn = np.linalg.norm(r, axis=1)
    gm = np.array([gstime(x) for x in jd])
    lon = (np.degrees(np.arctan2(r[:, 1], r[:, 0]) - gm) + 180) % 360 - 180
    lat = np.arctan(np.tan(np.arcsin(r[:, 2] / rn)) / (1 - FLAT) ** 2)
    alt = rn / 1e3 - RE * (1 - FLAT * np.sin(lat) ** 2)
    vrel = v.copy(); vrel[:, 0] += OMEGA * r[:, 1]; vrel[:, 1] -= OMEGA * r[:, 0]
    unix = (jd - 2440587.5) * 86400
    f107, f107a, aps = drv(unix)
    rho = pymsis.calculate((unix * 1e6).astype("int64").astype("datetime64[us]"), lon,
                           np.degrees(lat), alt, f107s=f107, f107as=f107a, aps=aps, version=2.1)[:, 0]
    g = 1.5 * (s.no_kozai / 60) * rho * np.linalg.norm(vrel, axis=1) ** 2 / np.linalg.norm(v, axis=1)
    T = (jd2 - jd1) * 86400
    dt = T / NS
    tt = (np.arange(NS) + 0.5) * dt
    return g.sum() * dt, ((T - tt) * g).sum() * dt, alt.mean()


def run(groups):
    rows = []
    for norad, g in groups:
        g = g.sort_values("jd").reset_index(drop=True)
        jd = g.jd.values
        last = -1e9
        for i in range(len(g)):
            if jd[i] < last + 1.0:
                continue
            j = np.searchsorted(jd, jd[i] + 66 / 24)
            if j >= len(g) or jd[j] > jd[i] + 78 / 24:
                continue
            res = integrals(g.l1[i], g.l2[i], jd[i], jd[j])
            if res is None:
                continue
            I, J, alt = res
            rows.append((norad, jd[i], jd[j], (g.n[j] - g.n[i]) / 60, I, J, alt, g.a_km[i]))
            last = jd[i]
    return rows


if __name__ == "__main__":
    tle = pd.read_parquet(f"{OUT}/tles_clean_all.parquet")
    keep = pd.read_parquet(f"{OUT}/intervals.parquet", columns=["norad"]).norad.unique()
    tle = tle[tle.norad.isin(keep)]
    groups = list(tle.groupby("norad"))
    with Pool(2) as p:
        res = sum(p.map(run, [groups[k::20] for k in range(20)]), [])
    w = pd.DataFrame(res, columns=["norad", "jd1", "jd2", "dn", "I", "J", "alt", "a_km"])
    w["t0"] = pd.to_datetime(w.jd1 - 2440587.5, unit="D")
    w["f72"] = w.dn / w.I
    w.to_parquet(f"{OUT}/windows72.parquet")
    print(len(w), "72-h windows,", w.norad.nunique(), "objects")
