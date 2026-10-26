"""Step 7 — paper figures (PNG, 200 dpi)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import spaceweather as sw

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"
FIG = f"{ROOT}/thermoswarm/figures"
import os; os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAG, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#e87ba4", "#008300", "#6250d6", "#e34948")
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "lines.linewidth": 2, "legend.frameon": False})
div = LinearSegmentedColormap.from_list("bwr", ["#104281", "#6da7ec", "#f0efec", "#ec8a8a", "#a32222"])

iv = pd.read_parquet(f"{OUT}/intervals.parquet").reset_index(drop=True)
storms = pd.read_csv(f"{OUT}/storms.csv", parse_dates=["onset", "end", "peak_time"])
three, daily = sw.load_gfz()
r12 = json.load(open(f"{OUT}/results_h1_h2.json"))
r3 = json.load(open(f"{OUT}/results_h3.json"))
short = iv[(iv.jd2 - iv.jd1) <= 2].copy()

# ---- Fig 1: altitude-time map ----
short["day"] = short.tmid.dt.floor("D")
short["band"] = (short.alt // 25) * 25
m = short[short.band.between(300, 700)].groupby(["band", "day"]).R.median().unstack()
m = m.loc[:, m.columns >= "2025-10-15"]
fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
im = a0.pcolormesh(m.columns, m.index + 12.5, m.values, cmap=div, norm=TwoSlopeNorm(1, 0.5, 1.5), shading="nearest")
a0.set_ylabel("Altitude (km)"); a0.grid(False)
a0.set_title("Daily median density ratio ρ_obs / ρ_NRLMSIS 2.1 by altitude (TLE inversion, 1,137 objects)", loc="left")
cb = fig.colorbar(im, ax=[a0, a1], fraction=0.03, pad=0.01); cb.set_label("ρ_obs / ρ_MSIS")
ap = three.ap["2025-10-15":"2026-10-12"]
a1.fill_between(ap.index, ap.values, step="post", color=BLUE, lw=0)
a1.set_ylabel("ap (3-h)")
for _, s in storms.iterrows():
    if s.kp_max >= 7 - 1e-6 and s.peak_time >= pd.Timestamp("2025-10-15"):
        a1.annotate(f"Kp {sw.kp_label(s.kp_max)}", (s.peak_time, ap.max() * 0.92), fontsize=7, color=INK2, ha="center")
fig.savefig(f"{FIG}/fig1_altitude_time_map.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 2: storm case studies ----
fig, axes = plt.subplots(2, 2, figsize=(10, 5.4), sharex="col", gridspec_kw=dict(height_ratios=[2.2, 1]))
bands = [(350, 450, BLUE, "350–450 km"), (450, 550, AQUA, "450–550 km"), (550, 650, ORANGE, "550–650 km"), (650, 750, VIOLET, "650–750 km")]
for col, (a, b, title) in enumerate([("2026-01-13", "2026-01-28", "G4 storm, 19–20 Jan 2026 (Kp 9-)"),
                                     ("2026-03-15", "2026-03-30", "Storm of 20–23 Mar 2026 (Kp 7o)")]):
    ax = axes[0, col]
    d = short[(short.tmid >= a) & (short.tmid < b)]
    for lo, hi, c, lab in bands:
        s = d[d.alt.between(lo, hi)].groupby(d.tmid.dt.floor("12h")).R.median()
        ax.plot(s.index, s.values, color=c, label=lab, marker="o", ms=3)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_title(title, loc="left"); ax.set_ylabel("ρ_obs / ρ_MSIS")
    if col == 0: ax.legend(ncol=2, fontsize=8)
    ap2 = three.ap[a:b]
    axes[1, col].fill_between(ap2.index, ap2.values, step="post", color=BLUE, lw=0)
    axes[1, col].set_ylabel("ap"); axes[1, col].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b")); axes[1, col].xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3))
fig.savefig(f"{FIG}/fig2_storm_case_studies.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 3: H1 — storm-time ratio per Kp>=7 storm, and relative response ----
st = pd.read_parquet(f"{OUT}/storm_samples.parquet")
h1 = st[(st.kp_max >= 7 - 1e-6) & st.alt.between(300, 550)]
rel = pd.read_parquet(f"{OUT}/storm_relative.parquet")
fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.6))
ids = sorted(h1.storm_id.unique())
labels = [f"{storms.set_index('storm_id').peak_time[i]:%d %b %y}\nKp {sw.kp_label(storms.set_index('storm_id').kp_max[i])}" for i in ids]
data = [h1[h1.storm_id == i].R.values for i in ids]
bp = a0.boxplot(data, tick_labels=labels, showfliers=False, widths=0.5, patch_artist=True,
                medianprops=dict(color=INK, lw=2), boxprops=dict(facecolor="#b7d3f6", edgecolor=INK2),
                whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
a0.axhline(1, color=INK2, ls="--", lw=1); a0.axhspan(0.9, 1.1, color="#f0efec", zorder=0)
a0.axhline(1.2, color=RED, lw=1); a0.text(5.45, 1.23, "H1 threshold (+20%)", color=RED, fontsize=7, ha="right")
a0.set_ylabel("ρ_obs / ρ_MSIS (300–550 km)"); a0.set_title("H1: storm-time density ratio, Kp ≥ 7 storms", loc="left")
a0.tick_params(axis="x", labelsize=7)
pre = r12["relative"]["pre_storm_baseline_R"]; px = r12["relative"]["per_storm_X"]
sid = sorted(int(k) for k in px)
kp = storms.set_index("storm_id").kp_max
cols = [RED if kp[i] >= 7 - 1e-6 else BLUE for i in sid]
a1.scatter([pre[str(i)] for i in sid], [px[str(i)] for i in sid], c=cols, s=40, edgecolor=SURF, lw=1.5, zorder=3)
for i in sid:
    if kp[i] >= 7 - 1e-6:
        a1.annotate(f"{storms.set_index('storm_id').peak_time[i]:%b %y}", (pre[str(i)], px[str(i)]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points", color=INK2)
a1.axhline(0, color=INK2, lw=1); a1.axvline(1, color=INK2, lw=1, ls="--")
a1.set_xlabel("Pre-storm baseline ρ_obs/ρ_MSIS"); a1.set_ylabel("Storm rise vs own baseline")
a1.set_title("Storm response relative to pre-storm level", loc="left")
a1.scatter([], [], c=RED, label="Kp ≥ 7"); a1.scatter([], [], c=BLUE, label="Kp 6- to 7-"); a1.legend(fontsize=8)
fig.savefig(f"{FIG}/fig3_H1_storm_ratios.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 4: storm-time ratio vs altitude ----
st["band"] = (st.alt // 50) * 50 + 25
g = st[st.alt.between(300, 750)].groupby("band").R
q = g.quantile([0.25, 0.5, 0.75]).unstack()
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.fill_betweenx(q.index, q[0.25], q[0.75], color="#b7d3f6", lw=0, label="interquartile range")
ax.plot(q[0.5], q.index, color=BLUE, marker="o", ms=5, label="median")
ax.axvline(1, color=INK2, ls="--", lw=1)
ax.set_xlabel("Storm-time ρ_obs / ρ_MSIS (all 18 calibrated storms)"); ax.set_ylabel("Altitude (km)")
ax.set_title("Storm-time model bias vs altitude", loc="left"); ax.legend(fontsize=8)
fig.savefig(f"{FIG}/fig4_ratio_vs_altitude.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 5: tau vs altitude (H2) ----
t = pd.DataFrame(r12["H2"]["table"])
fig, ax = plt.subplots(figsize=(5, 3.6))
con = t.r2 >= 0.1
for msk, c, lab in [(con, BLUE, "constrained (R² ≥ 0.1)"), (~con, "#9a9994", "unconstrained (R² < 0.1)")]:
    tt = t[msk]
    lo = [x[0] for x in tt.tau_ci]; hi = [x[1] for x in tt.tau_ci]
    ax.errorbar(tt.tau_h, tt.h_mid, xerr=[tt.tau_h - lo, np.array(hi) - tt.tau_h], fmt="o", color=c,
                ms=6, capsize=3, lw=1.5, label=lab)
ax.set_xlabel("Recovery time constant τ (h), 95% bootstrap CI over storms"); ax.set_ylabel("Altitude (km)")
ax.set_title("H2: thermospheric recovery time vs altitude", loc="left"); ax.legend(fontsize=8, loc="upper right")
fig.savefig(f"{FIG}/fig5_H2_tau_vs_altitude.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 6: forecast ladder ----
names = {"uncorrected": "NRLMSIS 2.1\n(uncorrected)", "persistence": "Persistence", "physics": "Physics\nmodel",
         "gb": "Gradient\nboosting", "gb_phys": "GB +\nphysics"}
fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.6))
ks = list(names)
for ax, stat, lab in [(a0, "median_km", "Median |along-track error| (km)"), (a1, "rms_km", "RMS along-track error (km)")]:
    vals = [r3["stormtime"][k][stat] for k in ks]
    bars = ax.bar(range(len(ks)), vals, color=[ "#9a9994", BLUE, BLUE, BLUE, ORANGE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(ks)), [names[k] for k in ks], fontsize=8); ax.set_ylabel(lab)
a0.set_title("72-h drag-induced along-track error, held-out storm windows", loc="left")
fig.savefig(f"{FIG}/fig6_forecast_ladder.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 7: per-storm reduction + SHAP ----
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(wspace=0.75))
ps = r3["per_storm_reduction_gbphys_median"]
ids = sorted(int(k) for k in ps)
v = [100 * ps[str(i)] for i in ids]
lab = [f"{storms.set_index('storm_id').peak_time[i]:%d %b}" for i in ids]
a0.bar(range(len(ids)), v, color=[BLUE if x >= 0 else RED for x in v], width=0.6)
a0.axhline(0, color=INK2, lw=1)
a0.set_xticks(range(len(ids)), lab, rotation=60, fontsize=7); a0.set_ylabel("Median error reduction (%)")
a0.set_title("Per-storm improvement of GB+physics over NRLMSIS", loc="left")
sh = pd.Series(r3["shap_mean_abs"]).sort_values().tail(10)
nice = {"phys": "physics-model prediction", "R_base7": "7-day baseline ratio", "f107_dev": "F10.7 − 81-day mean",
        "f107": "F10.7", "h_since_storm": "hours since storm onset", "R_last": "last observed ratio",
        "symh_min24": "SYM-H min (24 h)", "alt": "altitude", "Q_48h": "Newell coupling (48 h)",
        "Q_24h": "Newell coupling (24 h)", "ap_max24": "ap max (24 h)", "ap_now": "ap now"}
a1.barh([nice.get(k, k) for k in sh.index], sh.values, color=BLUE, height=0.6)
a1.set_xlabel("Mean |SHAP| (log density ratio)"); a1.set_title("What the model learned (SHAP)", loc="left")
a1.grid(axis="y", visible=False)
fig.savefig(f"{FIG}/fig7_per_storm_and_shap.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 8: data coverage ----
obj = iv.groupby("norad").alt.median()
types = pd.read_csv(f"{ROOT}/data/selected_objects.csv").set_index("NORAD_CAT_ID").OBJECT_TYPE
fig, ax = plt.subplots(figsize=(5, 3.6))
bins = np.arange(225, 800, 25)
stack = [obj[types.reindex(obj.index).values == t] for t in ["DEBRIS", "ROCKET BODY", "PAYLOAD"]]
ax.hist(stack, bins=bins, stacked=True, color=[BLUE, ORANGE, AQUA], label=["Debris", "Rocket body", "Payload"],
        edgecolor=SURF, lw=1)
ax.set_xlabel("Median altitude (km)"); ax.set_ylabel("Objects")
ax.set_title(f"Drag-probe network: {len(obj)} objects", loc="left"); ax.legend(fontsize=8)
fig.savefig(f"{FIG}/fig8_coverage.png", dpi=200, bbox_inches="tight"); plt.close()
print("figures written")
