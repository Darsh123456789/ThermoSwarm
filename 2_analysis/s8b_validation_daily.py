"""Step 8b — write validation_daily.csv, the input for Figure 3.

Input:  out/tle_vs_swarm.parquet (written by s8_swarm_validation.py): one row per
        network TLE interval matched to Swarm, with the network ratio R and the
        Swarm ratio sw_rel averaged over the same interval, for satellite sat.
Output: validation_daily.csv with columns date, network, swarm.

Method (same as the daily statistics in s8_swarm_validation.py):
  1. assign each matched interval to the UTC day of its midpoint (tmid);
  2. per Swarm satellite and day, take the median network R and median Swarm
     ratio, keeping satellite-days with at least 5 matched intervals;
  3. per day, take the median across satellites (A, B, C).

This is the code originally run inline (not saved as a file) on 13 Oct 2026 to
produce the CSV used for Figure 3; saved here unchanged except for the paths.
"""
import os as _os
import sys
import pandas as pd

ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
dest = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/validation_daily.csv"

m = pd.read_parquet(f"{OUT}/tle_vs_swarm.parquet")
m["day"] = m.tmid.dt.floor("D")
dd = m.groupby(["sat", "day"]).agg(tle=("R", "median"), sw=("sw_rel", "median"), n=("R", "size")).reset_index()
dd = dd[dd.n >= 5]
d = dd.groupby("day").agg(network=("tle", "median"), swarm=("sw", "median")).reset_index()
d["date"] = d.day.dt.strftime("%Y-%m-%d")
d[["date", "network", "swarm"]].round(3).to_csv(dest, index=False)
print(f"wrote {dest}: {len(d)} days, Pearson r = {d.network.corr(d.swarm):.3f}")
