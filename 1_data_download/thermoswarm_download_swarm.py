# ThermoSwarm — Swarm accelerometer/GPS density download (run in Google Colab)
# Source: ESA VirES for Swarm, public HAPI server (no login needed).
# Product: SW_OPER_DNSxPOD_2_ — thermosphere density from Swarm precise orbits,
# 30-s samples plus orbit-averaged density. Swarm A and C (~440-460 km),
# Swarm B (~500-510 km).
# Output: swarm_density.zip

import os, io, time, zipfile, requests
import pandas as pd

START, STOP = "2025-10-01", "2026-10-12"   # server returns what exists (to ~Jul 2026)
SATS = {"A": "SW_OPER_DNSAPOD_2_", "B": "SW_OPER_DNSBPOD_2_", "C": "SW_OPER_DNSCPOD_2_"}
PARAMS = "Latitude_GD,Longitude_GD,Height_GD,local_solar_time,density,density_orbitmean,validity_flag"
HAPI = "https://vires.services/hapi"
OUT = "swarm_density"
os.makedirs(OUT, exist_ok=True)

months = pd.date_range(START, STOP, freq="MS").tolist() + [pd.Timestamp(STOP)]
for sat, ds in SATS.items():
    try:
        info = requests.get(f"{HAPI}/info", params={"dataset": ds}, timeout=60).json()
        last = pd.Timestamp(info["stopDate"]).tz_localize(None)
        print(f"Swarm {sat}: data available until {last:%Y-%m-%d}")
    except Exception as e:
        print(f"Swarm {sat}: dataset not available ({e}), skipping"); continue
    frames = []
    for a, b in zip(months[:-1], months[1:]):
        if a > last:
            break
        for attempt in range(3):
            try:
                r = requests.get(f"{HAPI}/data", timeout=600, params={
                    "dataset": ds, "parameters": PARAMS, "format": "csv",
                    "start": a.strftime("%Y-%m-%dT00:00:00Z"),
                    "stop": min(b, last + pd.Timedelta("1D")).strftime("%Y-%m-%dT00:00:00Z")})
                r.raise_for_status()
                break
            except Exception as e:
                print(f"   retry {a:%Y-%m} ({e})"); time.sleep(20)
        else:
            print(f"   FAILED {a:%Y-%m}"); continue
        if r.text.strip():
            df = pd.read_csv(io.StringIO(r.text), header=None,
                             names=["time"] + PARAMS.split(","))
            # keep every 4th sample (2-min cadence) to keep the file small
            frames.append(df.iloc[::4])
            print(f"   {a:%Y-%m}: {len(df):,} samples")
    if frames:
        out = pd.concat(frames)
        out.to_csv(f"{OUT}/swarm{sat}_density.csv.gz", index=False)
        print(f"Swarm {sat}: saved {len(out):,} rows")

zf = "swarm_density.zip"
with zipfile.ZipFile(zf, "w") as z:
    for fn in os.listdir(OUT):
        z.write(f"{OUT}/{fn}", fn)
print(f"\nDone: {zf} ({os.path.getsize(zf)/1e6:.1f} MB)")
try:
    from google.colab import files
    files.download(zf)
except ImportError:
    pass
