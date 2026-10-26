"""Load space-weather drivers: Kp/ap/F10.7 (GFZ), Hp30 (GFZ), solar wind (OMNI 5-min).

Provides:
  load_gfz()      -> 3-hourly DataFrame (kp, ap) and daily DataFrame (Ap, f107, f107a)
  msis_inputs(t)  -> f107 (previous day), f107a (81-day centred), ap[7] for NRLMSIS
  load_omni()     -> hourly solar wind with Newell coupling function
  find_storms()   -> storm list from Kp
"""
import gzip
import numpy as np, pandas as pd

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
DATA = f"{ROOT}/data"


def _read_gfz(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            rows.append([int(p[0]), int(p[1]), int(p[2])] + [float(x) for x in p[7:15]]
                        + [float(x) for x in p[15:23]] + [float(p[23]), float(p[25])])
    cols = ["y", "m", "d"] + [f"kp{i}" for i in range(8)] + [f"ap{i}" for i in range(8)] + ["Ap", "f107"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
    return df.set_index("date")


def load_gfz():
    d = _read_gfz(f"{DATA}/kp_ap_f107_definitive.txt.gz")
    d = d[d.index >= "2025-01-01"]
    nc = _read_gfz(f"{DATA}/kp_ap_f107_nowcast.txt.gz")
    for day, row in nc.iterrows():          # fill days/values missing (-1) in definitive
        if day not in d.index:
            d.loc[day] = row
        else:
            for c in d.columns:
                if d.at[day, c] < 0 <= row[c]:
                    d.at[day, c] = row[c]
    d = d.sort_index()
    d = d.replace(-1.0, np.nan)
    daily = pd.DataFrame({"Ap": d.Ap, "f107": d.f107})
    daily["f107"] = daily.f107.where(daily.f107 > 0).interpolate().ffill()
    daily["f107a"] = daily.f107.rolling(81, center=True, min_periods=41).mean()
    kp = d[[f"kp{i}" for i in range(8)]].stack(future_stack=True).values
    ap = d[[f"ap{i}" for i in range(8)]].stack(future_stack=True).values
    idx = pd.date_range(d.index[0], periods=len(kp), freq="3h")
    three = pd.DataFrame({"kp": kp, "ap": ap}, index=idx)
    three = three.ffill()
    daily["Ap"] = daily.Ap.fillna(three.ap.resample("1D").mean())
    return three, daily


class MsisDrivers:
    """Vectorised NRLMSIS driver lookup (F10.7, F10.7A, 7-element ap array)."""

    def __init__(self):
        self.three, self.daily = load_gfz()
        self.t0 = self.three.index[0].value / 1e9
        self.ap3 = self.three.ap.values
        self.d0 = self.daily.index[0].value / 1e9
        self.f107 = self.daily.f107.values
        self.f107a = self.daily.f107a.values
        self.Ap = self.daily.Ap.values

    def __call__(self, unix):
        unix = np.asarray(unix, float)
        k = ((unix - self.t0) // 10800).astype(int)
        di = ((unix - self.d0) // 86400).astype(int)
        ap = self.ap3
        def A(j):
            return ap[np.clip(j, 0, len(ap) - 1)]
        aps = np.empty((len(unix), 7))
        aps[:, 0] = self.Ap[np.clip(di, 0, len(self.Ap) - 1)]
        for c, lag in zip(range(1, 5), range(0, 4)):
            aps[:, c] = A(k - lag)
        aps[:, 5] = np.mean([A(k - j) for j in range(4, 12)], axis=0)
        aps[:, 6] = np.mean([A(k - j) for j in range(12, 20)], axis=0)
        f107 = self.f107[np.clip(di - 1, 0, len(self.f107) - 1)]
        f107a = self.f107a[np.clip(di, 0, len(self.f107a) - 1)]
        return f107, f107a, aps


def load_hp30():
    rows = []
    with gzip.open(f"{DATA}/hp30.txt.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if int(p[0]) < 2025:
                continue
            rows.append((int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[7]), float(p[8])))
    df = pd.DataFrame(rows, columns=["y", "m", "d", "h", "hp30", "ap30"])
    t = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d)) + pd.to_timedelta(df.h, unit="h")
    df.index = t
    return df[["hp30", "ap30"]].replace(-1.0, np.nan)


def load_omni():
    frames = []
    for yr in (2025, 2026):
        a = np.loadtxt(f"{DATA}/omni_5min{yr}.asc.gz", usecols=(0, 1, 2, 3, 17, 18, 21, 25, 41))
        frames.append(a)
    a = np.vstack(frames)
    t = (pd.to_datetime(a[:, 0].astype(int).astype(str), format="%Y")
         + pd.to_timedelta(a[:, 1] - 1, unit="D") + pd.to_timedelta(a[:, 2], unit="h")
         + pd.to_timedelta(a[:, 3], unit="m"))
    df = pd.DataFrame(a[:, 4:], index=t, columns=["by", "bz", "v", "n", "symh"])
    df.loc[df.by > 9000, ["by", "bz"]] = np.nan
    df.loc[df.v > 9000, "v"] = np.nan
    df.loc[df.n > 900, "n"] = np.nan
    df.loc[df.symh > 9000, "symh"] = np.nan
    bt = np.hypot(df.by, df.bz)
    theta = np.arctan2(df.by, df.bz)
    # Newell et al. (2007) coupling function, scaled by 1e-4 (units ~ kV-equivalent)
    df["newell"] = 1e-4 * df.v ** (4 / 3) * bt ** (2 / 3) * np.abs(np.sin(theta / 2)) ** (8 / 3)
    hourly = df.resample("1h").mean()
    return hourly


def find_storms(three, kp_on=14 / 3, kp_keep=17 / 3, merge_h=24):
    """Storms: contiguous Kp >= 5- intervals, merged if <24 h apart, kept if max Kp >= 6-."""
    s = three.kp >= kp_on - 1e-6
    ev, start, last = [], None, None
    for t, flag in s.items():
        if flag:
            if start is None:
                start = t
            elif last is not None and (t - last) > pd.Timedelta(hours=merge_h):
                ev.append((start, last)); start = t
            last = t
    if start is not None:
        ev.append((start, last))
    out = []
    for a, b in ev:
        seg = three.kp[a:b + pd.Timedelta(hours=3)]
        if seg.max() >= kp_keep - 1e-6:
            out.append(dict(onset=a, end=b + pd.Timedelta(hours=3), peak_time=seg.idxmax(),
                            kp_max=seg.max(), ap_max=three.ap[a:b + pd.Timedelta(hours=3)].max()))
    df = pd.DataFrame(out)
    df["storm_id"] = np.arange(1, len(df) + 1)
    return df


def kp_label(k):
    n = int(round(k * 3))
    base = (n + 1) // 3
    rem = n - 3 * base
    return f"{base}{'-' if rem < 0 else '+' if rem > 0 else 'o'}"
