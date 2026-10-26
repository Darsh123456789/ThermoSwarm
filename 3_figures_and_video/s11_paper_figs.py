"""Journal-style vector figures for the research paper (Times font, PDF)."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats
import spaceweather as sw

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
OUT = f"{ROOT}/thermoswarm/out"; P = f"{ROOT}/paper_tex/figs"; os.makedirs(P, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
BLUE, ORANGE, GREY, RED, GREEN = "#2a6fbf", "#d9622b", "#9a9994", "#c0392b", "#2e8b57"
W1, W2 = 3.4, 6.9  # single / double column widths (in)
div = LinearSegmentedColormap.from_list("bwr", ["#104281", "#6da7ec", "#f4f3ef", "#ec8a8a", "#a32222"])


def lab(ax, s):
    ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


iv = pd.read_parquet(f"{OUT}/intervals.parquet").reset_index(drop=True)
three, daily = sw.load_gfz()
storms = pd.read_csv(f"{OUT}/storms.csv", parse_dates=["onset", "end", "peak_time"])

# ---- Fig 2: altitude-time map -------------------------------------------------
s = iv[(iv.jd2 - iv.jd1) <= 2].copy()
s["day"] = s.tmid.dt.floor("D"); s["band"] = (s.alt // 25) * 25
m = s[s.band.between(300, 700)].groupby(["band", "day"]).R.median().unstack()
m = m.loc[:, m.columns >= "2025-10-25"]
fig, (a0, a1) = plt.subplots(2, 1, figsize=(W2, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
im = a0.pcolormesh(m.columns, m.index + 12.5, m.values, cmap=div, norm=TwoSlopeNorm(1, 0.55, 1.45), shading="nearest", rasterized=True)
a0.set_ylabel("Altitude (km)")
cb = fig.colorbar(im, ax=[a0, a1], fraction=0.025, pad=0.01); cb.set_label(r"$\rho_{\mathrm{obs}}/\rho_{\mathrm{MSIS}}$")
ap = three.ap["2025-10-25":"2026-10-12"]
a1.fill_between(ap.index, ap.values, step="post", color="0.25", lw=0)
a1.set_ylabel(r"$a_p$")
for _, r in storms.iterrows():
    if r.kp_max >= 7 - 1e-6 and r.peak_time >= pd.Timestamp("2025-10-25"):
        a1.annotate(f"Kp {sw.kp_label(r.kp_max).replace('-', '−')}", (r.peak_time, 300), fontsize=7, ha="center", va="top")
a1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); a1.set_ylim(0, 320)
fig.savefig(f"{P}/fig2_map.pdf", dpi=300); plt.close()

# ---- Fig 3: validation ----------------------------------------------------------
d = pd.read_csv(f"{ROOT}/validation_daily.csv", parse_dates=["date"])
fig = plt.figure(figsize=(W2, 2.6))
a0 = fig.add_axes([0.07, 0.15, 0.60, 0.78]); a1 = fig.add_axes([0.76, 0.15, 0.23, 0.78])
a0.plot(d.date, d.swarm, color=GREY, lw=2.0, label="Swarm A/B/C (independent)")
a0.plot(d.date, d.network, color=BLUE, lw=0.9, label="TLE network")
a0.axhline(1, color="0.4", lw=0.6, ls="--"); a0.set_ylabel(r"$R$ (daily median)")
a0.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); a0.legend(loc="upper left", ncol=2)
lab(a0, "(a)")
r, _ = stats.pearsonr(d.network, d.swarm)
a1.scatter(d.swarm, d.network, s=5, color=BLUE, alpha=0.6, lw=0)
lo, hi = 0.55, 1.45; a1.plot([lo, hi], [lo, hi], color="0.3", lw=0.7, ls="--")
a1.set_xlim(lo, hi); a1.set_ylim(lo, hi); a1.set_aspect("equal")
a1.set_xlabel("Swarm"); a1.set_ylabel("TLE network")
a1.text(0.05, 0.92, f"$r$ = {r:.2f}\n$n$ = {len(d)} days", transform=a1.transAxes, fontsize=8, va="top")
lab(a1, "(b)")
fig.savefig(f"{P}/fig3_validation.pdf"); plt.close()

# ---- Fig 4: storm ratio vs altitude; storm rise vs pre-storm baseline ----------
st = pd.read_parquet(f"{OUT}/storm_samples.parquet")
st = st[st.alt.between(250, 750)].copy(); st["b"] = (st.alt // 50) * 50 + 25
q = st.groupby("b").R.agg(["median", "count", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
q.columns = ["med", "n", "q1", "q3"]
res = json.load(open(f"{OUT}/results_h1_h2.json"))["relative"]
X = pd.Series(res["per_storm_X"]).astype(float); X.index = X.index.astype(int)
Pb = pd.Series(res["pre_storm_baseline_R"]).astype(float); Pb.index = Pb.index.astype(int)
dd = pd.concat([X.rename("X"), Pb.rename("pre")], axis=1).dropna().join(storms.set_index("storm_id")[["kp_max", "peak_time"]])
fig, (a0, a1) = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(wspace=0.35))
a0.errorbar(q.med, q.index, xerr=[q.med - q.q1, q.q3 - q.med], fmt="o", color=BLUE, ms=4, lw=0.8, capsize=2)
a0.axvline(1, color="0.4", lw=0.6, ls="--")
a0.set_xlabel(r"Storm-time $R$ (median, IQR)"); a0.set_ylabel("Altitude (km)")
ts = stats.theilslopes(st.R, st.alt)
a0.text(0.97, 0.97, f"Theil–Sen slope\n{100 * 100 * ts.slope:.1f}% per 100 km", transform=a0.transAxes, ha="right", va="top", fontsize=8, bbox=dict(fc="white", ec="none", pad=1.5))
lab(a0, "(a)")
strong = dd.kp_max >= 7 - 1e-6
a1.scatter(dd.pre[~strong], 100 * dd.X[~strong], s=22, color=GREY, label=r"Kp 6− to 7−", zorder=3)
a1.scatter(dd.pre[strong], 100 * dd.X[strong], s=26, color=RED, label=r"Kp $\geq$ 7o", zorder=3)
lr = stats.linregress(dd.pre, 100 * dd.X); xx = np.linspace(dd.pre.min(), dd.pre.max(), 10)
a1.plot(xx, lr.intercept + lr.slope * xx, color="0.3", lw=0.8)
a1.axhline(0, color="0.4", lw=0.6, ls="--"); a1.axvline(1, color="0.4", lw=0.6, ls=":")
a1.set_xlabel(r"Pre-storm baseline $R$"); a1.set_ylabel("Storm rise above baseline (%)")
a1.text(0.97, 0.95, f"$r$ = {lr.rvalue:.2f}, $p$ = {lr.pvalue:.2f}\n$n$ = {len(dd)} storms", transform=a1.transAxes, ha="right", va="top", fontsize=8)
a1.legend(loc="lower left"); lab(a1, "(b)")
fig.savefig(f"{P}/fig4_storm_bias.pdf"); plt.close()

# ---- Fig 5: case studies ----------------------------------------------------------
short = iv[(iv.jd2 - iv.jd1) <= 2]
fig, axes = plt.subplots(2, 2, figsize=(W2, 3.6), sharex="col", gridspec_kw=dict(height_ratios=[2.3, 1], hspace=0.1, wspace=0.22))
bands = [(350, 450, "#2a6fbf", "350–450 km"), (450, 550, "#2e8b57", "450–550 km"), (550, 650, "#d9622b", "550–650 km"), (650, 750, "#7b4fbf", "650–750 km")]
for col, (a, b, title) in enumerate([("2026-01-13", "2026-01-28", "(a) 19–20 January 2026, Kp 9− (G4)"),
                                     ("2026-03-15", "2026-03-30", "(b) 20–23 March 2026, Kp 7o")]):
    ax = axes[0, col]; dsel = short[(short.tmid >= a) & (short.tmid < b)]
    for lo_, hi_, c, l in bands:
        ser = dsel[dsel.alt.between(lo_, hi_)].groupby(dsel.tmid.dt.floor("12h")).R.median()
        ax.plot(ser.index, ser.values, color=c, lw=1.1, marker="o", ms=2.2, label=l)
    ax.axhline(1, color="0.4", lw=0.6, ls="--"); ax.set_title(title, loc="left", fontsize=9)
    if col == 0: ax.set_ylabel(r"$R$ (12-h median)"); ax.legend(ncol=2, fontsize=7, loc="lower right")
    a2 = axes[1, col]; aps = three.ap[a:b]
    a2.fill_between(aps.index, aps.values, step="post", color="0.25", lw=0)
    if col == 0: a2.set_ylabel(r"$a_p$")
    a2.xaxis.set_major_locator(mdates.DayLocator(interval=4)); a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.savefig(f"{P}/fig5_cases.pdf"); plt.close()

# ---- Fig 6: tau ---------------------------------------------------------------------
t = pd.DataFrame(json.load(open(f"{OUT}/results_h1_h2.json"))["H2"]["table"])
fig, ax = plt.subplots(figsize=(W1, 2.5))
for _, r in t.iterrows():
    c = BLUE if r.r2 >= 0.1 else GREY
    ax.errorbar(r.tau_h, r.h_mid, xerr=[[r.tau_h - r.tau_ci[0]], [r.tau_ci[1] - r.tau_h]], fmt="o", color=c, ms=4, lw=0.9, capsize=2)
    ax.text(74, r.h_mid, f"$R^2$={r.r2:.3f}", fontsize=7, va="center", color="0.3")
ax.set_xlim(0, 86); ax.set_xlabel(r"Recovery time constant $\tau$ (h)"); ax.set_ylabel("Altitude (km)")
ax.scatter([], [], color=BLUE, s=12, label=r"constrained ($R^2\geq0.1$)"); ax.scatter([], [], color=GREY, s=12, label=r"unconstrained")
ax.legend(loc="upper center", fontsize=7, ncol=2, bbox_to_anchor=(0.45, 1.13))
fig.savefig(f"{P}/fig6_tau.pdf"); plt.close()

# ---- Fig 7: forecast -------------------------------------------------------------------
r3 = json.load(open(f"{OUT}/results_h3.json"))
names = [("uncorrected", "NRLMSIS 2.1"), ("persistence", "Persistence"), ("physics", "Physics model"),
         ("gb", "Gradient boosting"), ("gb_phys", "Boosting + physics")]
f = pd.read_parquet(f"{OUT}/forecast_results.parquet"); S = f[f.stormtime].copy()
S["band"] = pd.cut(S.alt, [300, 400, 500, 600, 700], labels=["300–400", "400–500", "500–600", "600–700"])
g = S.groupby("band", observed=True).apply(lambda x: pd.Series({"unc": np.abs(x.e_uncorrected).median(), "gbp": np.abs(x.e_gb_phys).median()}))
fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw=dict(wspace=0.55, width_ratios=[1.25, 1.25, 1]))
y = np.arange(len(names))[::-1]
for ax, key, xl, l in [(a0, "median_km", "Median |error| (km)", "(a)"), (a1, "rms_km", "RMS error (km)", "(b)")]:
    v = [r3["stormtime"][k][key] for k, _ in names]
    ax.barh(y, v, color=[BLUE if k == "gb_phys" else GREY for k, _ in names], height=0.6)
    for yy, vv in zip(y, v): ax.text(vv * 1.02, yy, f"{vv:.2f}" if key == "median_km" else f"{vv:.1f}", va="center", fontsize=7)
    ax.set_yticks(y, [n for _, n in names] if ax is a0 else [""] * len(y)); ax.set_xlabel(xl)
    ax.set_xlim(0, max(v) * 1.3); lab(ax, l)
xb = np.arange(len(g)); a2.bar(xb - 0.18, g.unc, 0.36, color=GREY, label="NRLMSIS 2.1"); a2.bar(xb + 0.18, g.gbp, 0.36, color=BLUE, label="Boosting + physics")
a2.set_yscale("log"); a2.set_xticks(xb, g.index, rotation=30, fontsize=7); a2.set_ylabel("Median |error| (km)"); a2.set_xlabel("Altitude band (km)")
for i, (u, b_) in enumerate(zip(g.unc, g.gbp)): a2.text(i, max(u, b_) * 1.25, f"−{100 * (1 - b_ / u):.0f}%", ha="center", fontsize=7)
a2.legend(fontsize=6.5, loc="upper right"); a2.set_ylim(0.3, 80); lab(a2, "(c)")
n_storms = S.group.nunique()          # storms with storm-time windows (14); not len(r3["groups"]), which is 15
fig.suptitle(f"Drag-induced along-track error on {len(S):,} storm-time windows from {n_storms} held-out storms",
             x=0.01, y=1.06, ha="left", fontsize=8, color="0.35")
fig.savefig(f"{P}/fig7_forecast.pdf"); plt.close()
print("figures done")
