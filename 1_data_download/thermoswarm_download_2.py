# ThermoSwarm — second data download (run in Google Colab, same as before)
# Adds:
#  1. Low-altitude drag probes (180–400 km): objects that re-entered during the
#     window and objects launched up to Sep 2026 — the first download excluded
#     both, which is why the network had < 30 objects below 300 km.
#  2. Solar EUV indices (S10.7, M10.7, Y10.7) and the Dst-based DTC file used by
#     the JB2008 density model, from Space Environment Technologies.
# Output: thermoswarm_data_2.zip

import os, io, json, time, gzip, zipfile, getpass, requests
import pandas as pd

START, END = "2025-10-01", "2026-10-12"
MAX_NEW = 800
OUT = "thermoswarm_data_2"
os.makedirs(OUT, exist_ok=True)
BASE = "https://www.space-track.org"
log = {"window": [START, END], "failed_downloads": []}

s = requests.Session()
r = s.post(BASE + "/ajaxauth/login",
           data={"identity": input("Space-Track email: "),
                 "password": getpass.getpass("Space-Track password: ")})
r.raise_for_status()
if "Failed" in r.text:
    raise SystemExit("Login failed — check email/password.")


def st(query, fmt="csv"):
    url = f"{BASE}/basicspacedata/query/{query}/format/{fmt}"
    for attempt in range(4):
        resp = s.get(url, timeout=600)
        if resp.status_code == 200:
            time.sleep(3)
            return resp.text
        print(f"  HTTP {resp.status_code}, retrying in 60 s…")
        time.sleep(60)
    resp.raise_for_status()


skip = "STARLINK|ONEWEB|KUIPER|QIANFAN|GUOWANG|ISS |ZARYA|TIANHE|WENTIAN|MENGTIAN|CSS|DRAGON|SOYUZ|PROGRESS|CYGNUS|SHENZHOU|TIANZHOU"

# 1a. objects that re-entered during the window (any final perigee)
print("Querying SATCAT…")
dec = pd.read_csv(io.StringIO(st(f"class/satcat/CURRENT/Y/DECAY/{START}--{END}")))
# 1b. low objects launched recently (not in the first download)
new = pd.read_csv(io.StringIO(st("class/satcat/CURRENT/Y/PERIGEE/>180/APOGEE/<450"
                                 "/LAUNCH/2025-09-01--2026-09-01/DECAY/null-val")))
cand = pd.concat([dec, new], ignore_index=True).drop_duplicates("NORAD_CAT_ID")
cand = cand[~cand.OBJECT_NAME.str.upper().str.contains(skip, na=False)]
cand = cand[cand.OBJECT_TYPE.isin(["DEBRIS", "ROCKET BODY", "PAYLOAD", "UNKNOWN"])]
prio = {"DEBRIS": 0, "ROCKET BODY": 1, "UNKNOWN": 2, "PAYLOAD": 3}
cand["PRIO"] = cand.OBJECT_TYPE.map(prio)
cand = cand.sort_values("PRIO").head(MAX_NEW)
cand.to_csv(f"{OUT}/selected_objects_2.csv", index=False)
print(f"Selected {len(cand)} extra objects")
print(cand.OBJECT_TYPE.value_counts().to_string())

with gzip.open(f"{OUT}/tle_history_2.txt.gz", "wt") as f:
    ids = cand.NORAD_CAT_ID.tolist()
    for i in range(0, len(ids), 100):
        chunk = ",".join(str(x) for x in ids[i:i + 100])
        print(f"  TLEs {i + 1}–{min(i + 100, len(ids))} of {len(ids)}")
        f.write(st(f"class/gp_history/NORAD_CAT_ID/{chunk}/EPOCH/{START}--{END}"
                   "/orderby/NORAD_CAT_ID,EPOCH", fmt="tle"))

# 2. JB2008 solar and geomagnetic index files
for name, url in {
    "SOLFSMY.TXT": "https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT",
    "DTCFILE.TXT": "https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT",
}.items():
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        open(f"{OUT}/{name}", "wb").write(r.content)
        print(f"OK   {name}")
    except Exception as e:
        print(f"FAIL {name}: {e}")
        log["failed_downloads"].append(name)

log["n_extra_objects"] = int(len(cand))
json.dump(log, open(f"{OUT}/manifest_2.json", "w"), indent=2)
zf = "thermoswarm_data_2.zip"
with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
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
