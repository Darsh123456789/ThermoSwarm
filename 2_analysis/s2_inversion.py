"""Step 2 — orbit-averaged density inversion (Picone, Emmert & Lean 2005 approach).

For each object, successive TLE pairs 1–3 days apart define an interval [t1, t2].
Observed:   dn_obs = n(t2) - n(t1)                         (Brouwer mean motion, rad/s)
Modelled:   I = ∫ (3/2) n ρ_MSIS v_rel² / v dt              (per unit ballistic coeff.)
along the SGP4 trajectory, ρ_MSIS from NRLMSIS 2.1 driven by observed F10.7 and ap.
Then f = dn_obs / I = B_eff · (ρ_true / ρ_MSIS) averaged over the interval.

Quiet-time calibration (cf. Doornbos et al. 2008): B is the running median of f
over quiet intervals (max ap ≤ 15 from 24 h before the interval to its end) within
±30 days. The density ratio is then R = f / B.
"""
import sys, warnings
import numpy as np, pandas as pd, pymsis
from multiprocessing import Pool
from sgp4.api import Satrec
from sgp4.propagation import gstime
from spaceweather import MsisDrivers

warnings.filterwarnings("ignore")
import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
RE, FLAT, OMEGA = 6378.137, 1 / 298.257223563, 7.2921159e-5
NSAMP = 96
drv = MsisDrivers()


def trajectory_integral(l1, l2, jd1, jd2):
    s = Satrec.twoline2rv(l1, l2)
    jd = np.linspace(jd1, jd2, NSAMP + 1)
    jd = 0.5 * (jd[1:] + jd[:-1])                     # midpoints
    e, r, v = s.sgp4_array(np.floor(jd), jd - np.floor(jd))
    if np.any(e):
        return None
    r, v = r * 1e3, v * 1e3                           # m, m/s
    rn = np.linalg.norm(r, axis=1)
    gm = np.array([gstime(x) for x in jd])
    lon = np.degrees(np.arctan2(r[:, 1], r[:, 0]) - gm)
    lon = (lon + 180) % 360 - 180
    latc = np.arcsin(r[:, 2] / rn)
    lat = np.arctan(np.tan(latc) / (1 - FLAT) ** 2)
    alt = rn / 1e3 - RE * (1 - FLAT * np.sin(lat) ** 2)
    vrel = v.copy()
    vrel[:, 0] += OMEGA * r[:, 1]
    vrel[:, 1] -= OMEGA * r[:, 0]
    vabs, vr = np.linalg.norm(v, axis=1), np.linalg.norm(vrel, axis=1)
    unix = (jd - 2440587.5) * 86400
    f107, f107a, aps = drv(unix)
    dates = (unix * 1e6).astype("int64").astype("datetime64[us]")
    rho = pymsis.calculate(dates, lon, np.degrees(lat), alt, f107s=f107, f107as=f107a,
                           aps=aps, version=2.1)[:, 0]
    n = s.no_kozai / 60.0
    g = 1.5 * n * rho * vr ** 2 / vabs
    dt = (jd2 - jd1) * 86400
    return g.mean() * dt, alt.mean(), np.average(alt, weights=rho), rho.mean()


def object_intervals(g):
    g = g.sort_values("jd").reset_index(drop=True)
    jd = g.jd.values
    rows, i = [], 0
    while i < len(g) - 1:
        j = np.searchsorted(jd, jd[i] + 1.0)
        if j >= len(g):
            break
        if jd[j] - jd[i] > 3.0:
            i = j; continue
        res = trajectory_integral(g.l1[i], g.l2[i], jd[i], jd[j])
        if res is not None:
            I, alt, alt_eff, rho_m = res
            dn = (g.n[j] - g.n[i]) / 60.0
            rows.append((g.norad[i], jd[i], jd[j], dn, I, alt, alt_eff, rho_m))
        i = j
    return rows


def run(chunk):
    out = []
    for _, g in chunk:
        out.extend(object_intervals(g))
    return out


if __name__ == "__main__":
    tle = pd.read_parquet(f"{OUT}/tles_clean.parquet")
    groups = list(tle.groupby("norad"))
    chunks = [groups[k::40] for k in range(40)]
    with Pool(2) as p:
        res = []
        for k, r in enumerate(p.imap_unordered(run, chunks)):
            res.extend(r)
            print(f"chunk {k + 1}/40", flush=True)
    iv = pd.DataFrame(res, columns=["norad", "jd1", "jd2", "dn", "I", "alt", "alt_eff", "rho_msis"])
    iv["t1"] = pd.to_datetime(iv.jd1 - 2440587.5, unit="D")
    iv["t2"] = pd.to_datetime(iv.jd2 - 2440587.5, unit="D")
    iv["tmid"] = iv.t1 + (iv.t2 - iv.t1) / 2
    iv["f"] = iv.dn / iv.I
    iv.to_parquet(f"{OUT}/intervals_raw.parquet")
    print(len(iv), "intervals from", iv.norad.nunique(), "objects")
