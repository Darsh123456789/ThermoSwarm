"""Still frames (1920x1080) for the 90-second project video."""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import matplotlib.image as mpimg

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"; FIG = f"{ROOT}/thermoswarm/figures"
V = f"{ROOT}/video_frames"; os.makedirs(V, exist_ok=True)
for f in ["/usr/share/fonts/opentype/inter/Inter-Regular.otf", "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
          "/usr/share/fonts/opentype/inter/Inter-Bold.otf"]:
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 22, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False})
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, RED, GREY, LBLUE = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#b9b8b0", "#cde2fb"
MONO = "DejaVu Sans Mono"


def frame(n, title, kicker):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    fig.text(0.05, 0.93, kicker.upper(), fontsize=20, color=BLUE, fontweight="semibold")
    fig.text(0.05, 0.865, title, fontsize=44, color=INK, fontweight="bold")
    fig.text(0.95, 0.04, f"ThermoSwarm  ·  {n}/9", fontsize=16, color=INK2, ha="right")
    return fig


def save(fig, n):
    fig.savefig(f"{V}/frame_{n:02d}.png", facecolor=BG); plt.close(fig)


def img_ax(fig, path, rect):
    ax = fig.add_axes(rect); ax.imshow(mpimg.imread(path)); ax.axis("off"); return ax


def stat(fig, x, y, big, small, color=BLUE):
    fig.text(x, y, big, fontsize=64, fontweight="bold", color=color)
    fig.text(x, y - 0.025, small, fontsize=21, color=INK2, va="top", linespacing=1.35)


iv = pd.read_parquet(f"{OUT}/intervals.parquet")
r3 = json.load(open(f"{OUT}/results_h3.json")); rs = json.load(open(f"{OUT}/results_swarm.json"))
n_obj, n_meas = iv.norad.nunique(), len(iv)

# 1 Title -------------------------------------------------------------------
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
ax = fig.add_axes([0.52, 0.0, 0.48, 1.0]); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.axis("off"); ax.set_aspect("equal")
ax.add_patch(Circle((0, 0), 1.0, color=LBLUE, alpha=0.6)); ax.add_patch(Circle((0, 0), 0.82, color=BLUE))
rng = np.random.default_rng(3)
for _ in range(260):
    r, th = rng.uniform(0.86, 1.2), rng.uniform(0, 2 * np.pi)
    ax.plot(r * np.cos(th), r * np.sin(th), "o", ms=3, color=INK2, alpha=0.7)
fig.text(0.06, 0.62, "ThermoSwarm", fontsize=88, fontweight="bold", color=INK)
fig.text(0.06, 0.56, "Turning 1,137 tracked objects into a\nsensor for storm-time atmospheric drag", fontsize=34, color=INK2, linespacing=1.4, va="top")
fig.text(0.06, 0.30, "Engineering Mechanics · Aerospace", fontsize=22, color=BLUE, fontweight="semibold")
fig.text(0.06, 0.25, "Public data, October 2025 – October 2026", fontsize=22, color=INK2)
save(fig, 1)

# 2 Problem -------------------------------------------------------------------
fig = frame(2, "Solar storms swell the upper atmosphere", "The problem")
ax = fig.add_axes([0.04, 0.08, 0.46, 0.72]); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.1, 1.6); ax.axis("off"); ax.set_aspect("equal")
for r, a in [(1.42, 0.12), (1.25, 0.18)]:
    ax.add_patch(Wedge((0, -0.9), r, 20, 160, color=ORANGE, alpha=a))
ax.add_patch(Wedge((0, -0.9), 1.1, 20, 160, color=LBLUE, alpha=0.8))
ax.add_patch(Wedge((0, -0.9), 0.9, 0, 180, color=BLUE))
for th in np.linspace(35, 145, 7):
    t = np.radians(th); ax.plot(1.18 * np.cos(t), -0.9 + 1.18 * np.sin(t), "o", ms=10, color=INK)
ax.text(0, 0.62, "storm-heated thermosphere expands", ha="center", fontsize=20, color=ORANGE, fontweight="semibold")
ax.text(0, -1.05, "Earth", ha="center", fontsize=18, color=BG)
fig.text(0.54, 0.70, "More air at satellite altitudes  →  more drag", fontsize=28, color=INK, fontweight="semibold")
stat(fig, 0.54, 0.54, "38 of 49", "Starlink satellites lost to a moderate storm, Feb 2022")
stat(fig, 0.54, 0.36, "55 km", "RMS 72-hour along-track error from drag alone,\nduring storms, with the standard model (this study)")
fig.text(0.54, 0.12, "Density models such as NRLMSIS are least reliable exactly\nwhen collision and re-entry predictions matter most.",
         fontsize=22, color=INK2, linespacing=1.4)
save(fig, 2)

# 3 Idea -----------------------------------------------------------------------
fig = frame(3, "Every tracked object is a drag probe", "The idea")
fig.text(0.05, 0.70, r"$\dfrac{da}{dt} = -\rho\,B\,\sqrt{\mu a}$", fontsize=56, color=INK)
fig.text(0.05, 0.60, "How fast an orbit shrinks (from public TLEs)\ntells you the air density ρ it flies through.", fontsize=24, color=INK2, linespacing=1.4, va="top")
stat(fig, 0.05, 0.38, f"{n_obj:,}", "objects calibrated (mostly debris & rocket bodies)")
stat(fig, 0.05, 0.20, f"{n_meas:,}", "orbit-averaged density measurements")
alt = iv.groupby("norad").alt.median()
ax = fig.add_axes([0.53, 0.12, 0.42, 0.66]); ax.set_facecolor(BG)
ax.hist(alt, bins=np.arange(200, 801, 25), orientation="horizontal", color=BLUE, edgecolor=BG, linewidth=1.5)
ax.set_ylabel("Altitude (km)"); ax.set_xlabel("Objects per 25-km band"); ax.grid(axis="x", color=GRID)
ax.set_title("The network: 197–786 km", loc="left", fontsize=24, color=INK, fontweight="semibold")
save(fig, 3)

# 4 Method + code ----------------------------------------------------------------
fig = frame(4, "From raw orbits to a calibrated density ratio", "The method")
steps = [("Quality control", "drop duplicates, outliers,\nmanoeuvring objects"),
         ("Density inversion", "Δn observed vs NRLMSIS 2.1\nalong the SGP4 orbit"),
         ("Quiet-time calibration", "ballistic coefficient B from\ngeomagnetically quiet days"),
         ("Ratio R = ρ_obs / ρ_MSIS", "feeds every test")]
for i, (h, b) in enumerate(steps):
    y = 0.70 - i * 0.165
    fig.patches.append(FancyBboxPatch((0.05, y - 0.06), 0.34, 0.12, boxstyle="round,pad=0.005,rounding_size=0.012",
                                      transform=fig.transFigure, fc=LBLUE if i == 3 else BG, ec=BLUE if i == 3 else INK2, lw=2))
    fig.text(0.07, y + 0.012, h, fontsize=24, fontweight="semibold", color=INK)
    fig.text(0.07, y - 0.042, b, fontsize=17, color=INK2)
    if i < 3: fig.text(0.215, y - 0.093, "↓", fontsize=26, color=INK2, ha="center")
code = ("# s2_inversion.py — drag along the orbit\n"
        "rho = pymsis.calculate(dates, lon, lat, alt,\n"
        "        f107s=f107, f107as=f107a, aps=aps,\n"
        "        version=2.1)[:, 0]\n"
        "g = 1.5 * n * rho * vr**2 / vabs\n"
        "I = g.mean() * dt\n"
        "f = dn_obs / I        # = B x (rho_true / rho_MSIS)\n\n"
        "# s3_calibrate.py — quiet-time B\n"
        "quiet = ap_max_24h <= 15\n"
        "B = median(f[quiet & within_30_days])\n"
        "R = f / B             # density ratio")
fig.patches.append(FancyBboxPatch((0.45, 0.13), 0.50, 0.66, boxstyle="round,pad=0.005,rounding_size=0.012",
                                  transform=fig.transFigure, fc="#1a1a19", ec="#1a1a19"))
fig.text(0.47, 0.75, code, fontsize=19, family=MONO, color="#f0efec", va="top", linespacing=1.55)
save(fig, 4)

# 5 Map -------------------------------------------------------------------------
fig = frame(5, "A year of model error, mapped by altitude", "Result 1")
img_ax(fig, f"{FIG}/fig1_altitude_time_map.png", [0.03, 0.07, 0.94, 0.74])
save(fig, 5)

# 6 Validation ------------------------------------------------------------------
fig = frame(6, "Checked against ESA's Swarm satellites", "Validation")
d = pd.read_csv(f"{ROOT}/validation_daily.csv", parse_dates=["date"])
ax = fig.add_axes([0.06, 0.12, 0.60, 0.64]); ax.set_facecolor(BG)
ax.plot(d.date, d.swarm, color=GREY, lw=3.5, label="Swarm A/B/C (independent)")
ax.plot(d.date, d.network, color=BLUE, lw=2.2, label="ThermoSwarm network")
ax.axhline(1, color=INK2, ls="--", lw=1.2); ax.set_ylabel("ρ_obs / ρ_MSIS"); ax.grid(axis="y", color=GRID)
ax.legend(frameon=False, fontsize=19, loc="upper left")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
stat(fig, 0.71, 0.62, f"r = {rs['daily_network']['pearson_r']:.2f}", "daily correlation over\n716 satellite-days")
fig.text(0.71, 0.48, "", fontsize=1)
stat(fig, 0.71, 0.34, f"{rs['daily_network']['mad_pct']:.0f}%", "median daily difference")
save(fig, 6)

# 7 H1 explanation -------------------------------------------------------------
fig = frame(7, "Why storms looked smaller than expected", "Hypothesis 1")
img_ax(fig, f"{FIG}/fig2_storm_case_studies.png", [0.02, 0.10, 0.58, 0.70])
fig.text(0.63, 0.74, "H1 (>20% underestimate): not supported", fontsize=24, color=RED, fontweight="semibold")
stat(fig, 0.63, 0.58, "+73%", "storm peaks the model misses\n(Swarm, 3-hour resolution)", ORANGE)
stat(fig, 0.63, 0.38, "67% → 32%", "day-long TLE averaging halves\nthe visible storm rise", INK)
stat(fig, 0.63, 0.18, "24% too high", "model baseline just before\nthe Jan 2026 G4 storm", BLUE)
save(fig, 7)

# 8 Forecast ---------------------------------------------------------------------
fig = frame(8, "Forecasting the error cuts orbit-prediction error", "Hypothesis 3")
st = r3["stormtime"]
names = [("uncorrected", "NRLMSIS 2.1"), ("persistence", "Persistence"), ("physics", "Physics model"),
         ("gb", "Gradient boosting"), ("gb_phys", "Boosting + physics")]
ax = fig.add_axes([0.20, 0.14, 0.40, 0.60]); ax.set_facecolor(BG)
vals = [st[k]["rms_km"] for k, _ in names][::-1]; labs = [l for _, l in names][::-1]
cols = [BLUE if l == "Boosting + physics" else GREY for l in labs]
ax.barh(labs, vals, color=cols, height=0.6)
for i, v in enumerate(vals): ax.text(v + 0.8, i, f"{v:.1f} km", va="center", fontsize=20, color=INK)
ax.set_xlabel("RMS 72-h along-track error, held-out storms (km)"); ax.set_xlim(0, 66); ax.grid(axis="x", color=GRID)
stat(fig, 0.66, 0.60, f"−{100 * r3['reduction_gb_phys_rms']['value']:.0f}%", "RMS error, 15 unseen storms")
stat(fig, 0.66, 0.42, f"−{100 * r3['reduction_gb_phys_median']['value']:.0f}%", "median error (3.30 → 2.68 km)")
fig.text(0.66, 0.20, "H3 (≥30%): partially supported\nRe-entry dates: no measurable gain", fontsize=22, color=INK2, linespacing=1.5)
save(fig, 8)

# 9 Impact ---------------------------------------------------------------------
fig = frame(9, "A free, validated monitor for space safety", "Impact")
items = [("Collision screening", "narrower storm-time uncertainty in miss distances"),
         ("Emerging space programmes", "e.g. ISRO's IS4OM: no density satellite needed"),
         ("Constellation launches", "real-time bias to time orbit raising"),
         ("Better models", "a year-long, altitude-resolved record of NRLMSIS error")]
for i, (h, b) in enumerate(items):
    y = 0.70 - i * 0.13
    fig.text(0.06, y, "●", fontsize=22, color=BLUE); fig.text(0.09, y, h, fontsize=28, fontweight="semibold", color=INK)
    fig.text(0.09, y - 0.045, b, fontsize=21, color=INK2)
fig.text(0.62, 0.70, "Next steps", fontsize=28, fontweight="semibold", color=INK)
fig.text(0.62, 0.66, "• faster orbit updates to\n   capture storm peaks\n• 2024 solar-maximum storms\n• JB2008 comparison\n• daily open service",
         fontsize=22, color=INK2, linespacing=1.5, va="top")
fig.text(0.06, 0.10, "Data: Space-Track.org, GFZ Potsdam, NASA OMNIWeb, ESA Swarm/VirES.  Code developed with AI assistance (Claude).",
         fontsize=16, color=INK2)
save(fig, 9)
print("frames written")
