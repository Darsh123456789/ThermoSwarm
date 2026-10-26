"""Step 2b — run the step-2 inversion on the low-altitude objects (second download)."""
import pandas as pd
from multiprocessing import Pool
from s2_inversion import run, OUT

if __name__ == "__main__":
    tle = pd.read_parquet(f"{OUT}/tles_clean_2.parquet")
    groups = list(tle.groupby("norad"))
    with Pool(2) as p:
        res = sum(p.map(run, [groups[k::20] for k in range(20)]), [])
    iv = pd.DataFrame(res, columns=["norad", "jd1", "jd2", "dn", "I", "alt", "alt_eff", "rho_msis"])
    iv["t1"] = pd.to_datetime(iv.jd1 - 2440587.5, unit="D")
    iv["t2"] = pd.to_datetime(iv.jd2 - 2440587.5, unit="D")
    iv["tmid"] = iv.t1 + (iv.t2 - iv.t1) / 2
    iv["f"] = iv.dn / iv.I
    old = pd.read_parquet(f"{OUT}/intervals_raw.parquet")
    old.to_parquet(f"{OUT}/intervals_raw_v1.parquet")
    pd.concat([old, iv]).to_parquet(f"{OUT}/intervals_raw.parquet")
    print(len(iv), "new intervals from", iv.norad.nunique(), "objects")
