# ThermoSwarm — data download script
# Run this in Google Colab (Runtime > Run all). It asks for your Space-Track
# login (typed into Colab only, never stored), downloads one year of TLE
# history for ~1,500 near-circular LEO objects, the re-entry list, and the
# space-weather indices, then zips everything as thermoswarm_data.zip.

import os, io, json, time, gzip, zipfile, getpass, requests
import pandas as pd

START, END = "2025-10-01", "2026-10-12"   # study window
MAX_OBJECTS = 1500                        # drag-probe objects
MAX_REENTRY = 150                         # re-entry case-study objects
OUT = "thermoswarm_data"
os.makedirs(OUT, exist_ok=True)
BASE = "https://www.space-track.org"
log = {"window": [START, END], "failed_downloads": []}

# ---------- Space-Track login ----------
s = requests.Session()
r = s.post(BASE + "/ajaxauth/login",
           data={"identity": input("Space-Track email: "),
                 "password": getpass.getpass("Space-Track password: ")})
r.raise_for_status()
if "Failed" in r.text:
    raise SystemExit("Login failed — check email/password.")

def st(query, fmt="csv"):
    """One Space-Track query, with retries and a pause to respect rate limits."""
    url = f"{BASE}/basicspacedata/query/{query}/format/{fmt}"
    for attempt in range(4):
        resp = s.get(url, timeout=600)
        if resp.status_code == 200:
            time.sleep(3)
            return resp.text
        print(f"  HTTP {resp.status_code}, retrying in 60 s…")
        time.sleep(60)
    resp.raise_for_status()

# ---------- 1. Candidate objects from the satellite catalogue ----------
print("Querying SATCAT…")
q = "class/satcat/CURRENT/Y/PERIGEE/>200/APOGEE/<700/LAUNCH/<2025-09-01"
alive = pd.read_csv(io.StringIO(st(q + "/DECAY/null-val")))
died = pd.read_csv(io.StringIO(st(q + f"/DECAY/>{START}")))
cat = pd.concat([alive, died], ignore_index=True).drop_duplicates("NORAD_CAT_ID")

# near-circular only (apogee - perigee < 50 km)
cat = cat[(cat.APOGEE - cat.PERIGEE) < 50].copy()
# drop actively manoeuvring constellations and crewed stations
skip = "STARLINK|ONEWEB|KUIPER|QIANFAN|GUOWANG|ISS |ZARYA|TIANHE|WENTIAN|MENGTIAN|CSS"
cat = cat[~cat.OBJECT_NAME.str.upper().str.contains(skip, na=False)]
cat["ALT"] = (cat.APOGEE + cat.PERIGEE) / 2
cat["BIN"] = (cat.ALT // 25) * 25
# prefer debris and rocket bodies (no propulsion), then payloads
prio = {"DEBRIS": 0, "ROCKET BODY": 1, "PAYLOAD": 2}
cat["PRIO"] = cat.OBJECT_TYPE.map(prio).fillna(3)
cat = cat.sample(frac=1, random_state=42).sort_values("PRIO")

# stratify evenly across 25 km altitude bins
per_bin = max(1, MAX_OBJECTS // cat.BIN.nunique())
sel = cat.groupby("BIN").head(per_bin)
if len(sel) < MAX_OBJECTS:  # top up from bins with spare objects
    sel = pd.concat([sel, cat.drop(sel.index).head(MAX_OBJECTS - len(sel))])
sel = sel.head(MAX_OBJECTS)
sel.to_csv(f"{OUT}/selected_objects.csv", index=False)
print(f"Selected {len(sel)} objects from {len(cat)} candidates")
print(sel.groupby("BIN").size().to_string())

def pull_history(ids, path):
    """Download TLE history for a list of NORAD IDs, 100 per request."""
    with gzip.open(path, "wt") as f:
        for i in range(0, len(ids), 100):
            chunk = ",".join(str(x) for x in ids[i:i + 100])
            print(f"  TLEs {i + 1}–{min(i + 100, len(ids))} of {len(ids)}")
            f.write(st(f"class/gp_history/NORAD_CAT_ID/{chunk}/EPOCH/{START}--{END}"
                       "/orderby/NORAD_CAT_ID,EPOCH", fmt="tle"))

# ---------- 2. TLE history for the drag probes ----------
print("Downloading TLE history (this takes a while)…")
pull_history(sel.NORAD_CAT_ID.tolist(), f"{OUT}/tle_history.txt.gz")

# ---------- 3. Re-entries during the window ----------
print("Downloading re-entry list…")
dec = pd.read_csv(io.StringIO(st(f"class/decay/DECAY_EPOCH/{START}--{END}"
                                 "/orderby/DECAY_EPOCH")))
dec.to_csv(f"{OUT}/decay.csv", index=False)
uncontrolled = dec[~dec.OBJECT_NAME.str.upper().str.contains(skip, na=False)]
re_ids = uncontrolled.NORAD_CAT_ID.drop_duplicates().head(MAX_REENTRY).tolist()
if re_ids:
    pull_history(re_ids, f"{OUT}/tle_reentry.txt.gz")

# ---------- 4. Space-weather indices ----------
sw = {
    "kp_ap_f107_definitive.txt": "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt",
    "kp_ap_f107_nowcast.txt": "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt",
    "hp30.txt": "https://kp.gfz-potsdam.de/app/files/Hp30_ap30_complete_series.txt",
    "omni_5min2025.asc": "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min2025.asc",
    "omni_5min2026.asc": "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min2026.asc",
}
for name, url in sw.items():
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with gzip.open(f"{OUT}/{name}.gz", "wb") as f:
            f.write(r.content)
        print(f"OK   {name}")
    except Exception as e:
        print(f"FAIL {name}: {e}")
        log["failed_downloads"].append(name)

# ---------- 5. Manifest and zip ----------
log.update({"n_selected": int(len(sel)), "n_candidates": int(len(cat)),
            "n_reentry_objects": len(re_ids),
            "selection": "SATCAT current, perigee>200 km, apogee<700 km, "
                         "apogee-perigee<50 km, launched before 2025-09-01, "
                         "constellations/stations excluded, 25 km altitude strata"})
json.dump(log, open(f"{OUT}/manifest.json", "w"), indent=2)

zf = "thermoswarm_data.zip"
with zipfile.ZipFile(zf, "w") as z:
    for fn in os.listdir(OUT):
        z.write(f"{OUT}/{fn}", fn)
print(f"\nDone: {zf} ({os.path.getsize(zf) / 1e6:.1f} MB)")
if log["failed_downloads"]:
    print("Failed (download these by hand):", log["failed_downloads"])

try:
    from google.colab import files
    files.download(zf)
except ImportError:
    pass
