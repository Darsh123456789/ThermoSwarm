"""Step 1 — parse TLE history and quality-control it.

Output: tles_clean.parquet with one row per accepted TLE:
norad, epoch (UTC), n_rad_min (Brouwer mean motion), a_km, ecc, incl, bstar, alt_km
"""
import gzip, sys
import numpy as np, pandas as pd
from sgp4.api import Satrec

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
DATA = f"{ROOT}/data"
OUT = f"{ROOT}/thermoswarm/out"
RE, MU = 6378.137, 398600.4418


def parse(path):
    rows, seen = [], set()
    with gzip.open(path, "rt") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    for l1, l2 in zip(lines[0::2], lines[1::2]):
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        try:
            s = Satrec.twoline2rv(l1, l2)
        except Exception:
            continue
        if s.error:
            continue
        key = (s.satnum, round(s.jdsatepoch + s.jdsatepochF, 8))
        if key in seen:          # Space-Track returns each TLE twice (alpha-5 + 5-digit)
            continue
        seen.add(key)
        n = s.no_kozai           # rad/min, Brouwer (un-Kozai) mean motion
        a = (MU / (n / 60.0) ** 2) ** (1 / 3)
        rows.append((s.satnum, s.jdsatepoch + s.jdsatepochF, n, a, s.ecco,
                     np.degrees(s.inclo), s.bstar, l1, l2))
    df = pd.DataFrame(rows, columns=["norad", "jd", "n", "a_km", "ecc", "incl",
                                     "bstar", "l1", "l2"])
    df["epoch"] = pd.to_datetime(df.jd - 2440587.5, unit="D")
    df["alt_km"] = df.a_km - RE
    return df.sort_values(["norad", "jd"]).reset_index(drop=True)


def qc(df, log):
    n0 = len(df)
    df = df[df.ecc < 0.01]                                 # near-circular
    log["ecc_cut"] = n0 - len(df)
    df = df.drop_duplicates(["norad", "jd"])
    keep, bad_objs = [], 0
    for norad, g in df.groupby("norad"):
        g = g.sort_values("jd")
        # drop TLEs closer than 0.1 d (duplicate fits) — keep the later
        g = g[np.r_[np.diff(g.jd.values) > 0.1, True]]
        # robust outlier rejection on a(t): residual from 7-point running median
        med = g.a_km.rolling(7, center=True, min_periods=3).median()
        resid = (g.a_km - med).abs()
        mad = 1.4826 * resid.rolling(31, center=True, min_periods=5).median()
        g = g[(resid <= np.maximum(5 * mad, 0.05)) | mad.isna()]
        if len(g) < 30:
            continue
        # manoeuvre detection: drag can only lower a; a rise > 0.3 km between
        # successive TLEs (beyond noise) marks a manoeuvre
        da = np.diff(g.a_km.values)
        rises = np.where(da > 0.3)[0]
        if len(rises) > 3:            # repeatedly manoeuvring -> drop object
            bad_objs += 1
            continue
        mask = np.ones(len(g), bool)
        for k in rises:               # drop a 2-day window around isolated jumps
            t = g.jd.values[k + 1]
            mask &= ~((g.jd.values > t - 2) & (g.jd.values < t + 2))
        g = g[mask]
        if len(g) >= 30:
            keep.append(g)
    log["manoeuvring_objects_dropped"] = bad_objs
    out = pd.concat(keep)
    log["tles_after_qc"] = len(out)
    log["objects_after_qc"] = out.norad.nunique()
    return out


if __name__ == "__main__":
    log = {}
    df = parse(f"{DATA}/tle_history.txt.gz")
    log["tles_unique"] = len(df); log["objects"] = df.norad.nunique()
    clean = qc(df, log)
    clean.to_parquet(f"{OUT}/tles_clean.parquet")
    re = parse(f"{DATA}/tle_reentry.txt.gz")
    re.to_parquet(f"{OUT}/tles_reentry.parquet")
    log["reentry_tles"] = len(re); log["reentry_objects"] = re.norad.nunique()
    print(pd.Series(log).to_string())
    print(clean.groupby("norad").alt_km.mean().describe())
