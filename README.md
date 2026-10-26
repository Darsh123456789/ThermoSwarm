# ThermoSwarm

Analysis code for the paper **"ThermoSwarm: Altitude-Resolved Storm-Time Thermospheric Density From Multi-Object Orbital Decay, With Physics-Informed Drag Forecasting"** by Darsh Chaudhary (submitted to *Space Weather*).

The scripts turn about 1,100 already-tracked objects in low Earth orbit into a network of drag sensors. They invert public two-line element sets (TLEs) for thermospheric density against NRLMSIS 2.1, validate the result against ESA Swarm densities, test storm-time model bias and recovery, and train a physics-informed gradient-boosting drag forecaster.

Study window: 1 October 2025 to 12 October 2026.

## Contents

| Folder | What it does |
|---|---|
| `1_data_download/` | Downloads all input data (run in Google Colab) |
| `2_analysis/` | Quality control, density inversion, calibration, hypothesis tests H1–H3, Swarm validation, re-entry |
| `3_figures_and_video/` | Paper figures and presentation frames |
| `4_paper/` | LaTeX source of the paper and a Word converter |

No data is included in this repository. See "Data" below.

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

`4_paper/make_docx.py` also needs [pandoc](https://pandoc.org).

## Data

All inputs are free and public. Run the three download scripts in Google Colab (Runtime > Run all). Each produces a zip file.

| Script | Downloads | Login |
|---|---|---|
| `thermoswarm_download.py` | Space-Track TLE history for about 1,500 drag probes, the re-entry list, GFZ Kp/ap/F10.7/Hp30, NASA OMNI 5-minute solar wind | Free Space-Track account |
| `thermoswarm_download_2.py` | 800 low-altitude and re-entered objects; JB2008 solar indices from Space Environment Technologies | Free Space-Track account |
| `thermoswarm_download_swarm.py` | ESA Swarm A/B/C densities (`SW_OPER_DNSxPOD_2_`) from the public VirES HAPI server | None |

The scripts ask for your Space-Track login when run; no credentials are stored in this repository.

Space-Track's user agreement does not allow redistributing TLEs, so the raw orbital data must be downloaded by each user. The derived densities, calibrated ballistic coefficients, storm list and forecast outputs are archived separately on Zenodo: DOI to be added.

### Folder layout

Set the environment variable `THERMOSWARM_ROOT` to a working folder (default `/home/claude`). Unzip the downloads into:

```
$THERMOSWARM_ROOT/
├── data/              # thermoswarm_data.zip
├── data2/             # thermoswarm_data_2.zip
├── swarm/             # swarm_density.zip
└── thermoswarm/out/   # analysis outputs (create this folder)
```

```bash
export THERMOSWARM_ROOT=/path/to/workdir
mkdir -p $THERMOSWARM_ROOT/thermoswarm/out
```

## Running the analysis

Run from inside `2_analysis/`, in this order:

| Step | Script | Purpose | Main outputs |
|---|---|---|---|
| — | `spaceweather.py` | Shared helpers: index loading, NRLMSIS drivers, Newell coupling, storm detection | — |
| 1 | `s1_parse_qc.py` | Parse TLEs; remove duplicates and outliers; reject manoeuvres | `tles_clean.parquet`, `tles_reentry.parquet` |
| 2 | `s2_inversion.py` | Picone–Emmert–Lean density inversion against NRLMSIS 2.1 | `intervals_raw.parquet` |
| 2b | `s2b_inversion_lowalt.py` | Same inversion for the low-altitude set (see note 1) | `intervals_raw.parquet` (appended) |
| 3 | `s3_calibrate.py` | Quiet-time ballistic-coefficient calibration → density ratio R; storm list | `intervals.parquet`, `storms.csv`, `storm_samples.parquet` |
| 4 | `s4_storms.py` | H1 (storm bias) and H2 (recovery time τ) | `results_h1_h2.json`, `h2_tau_by_band.csv` |
| 5 | `s5_windows72.py` | 72-hour forecast windows and drag integrals | `windows72.parquet` |
| 6 | `s6_forecast.py` | H3: model ladder, storm-wise cross-validation, along-track error, SHAP | `results_h3.json`, `forecast_results.parquet` |
| 8 | `s8_swarm_validation.py` | Independent validation against Swarm | `results_swarm.json`, `swarm_storms.csv` |
| 9 | `s9_reentry.py` | Re-entry date predictions at 7, 3 and 1 day | `results_reentry.json` |

Then, from `3_figures_and_video/`:

- `s11_paper_figs.py`: Figures 2–7 of the paper (see note 2). Figure 1 (pipeline diagram) is a drawn image, `fig1_pipeline.png`.
- `s7_figures.py`: earlier draft figures.
- `s10_video_frames.py`: frames for the project video.


### Notes

1. **Low-altitude set.** Before step 2b, run the `parse` and `qc` functions from `s1_parse_qc.py` on `data2/tle_history_2.txt.gz` and save the result as `tles_clean_2.parquet`. Then concatenate `tles_clean.parquet` and `tles_clean_2.parquet` into `tles_clean_all.parquet` (used by step 5).
2. **Figure 3 input.** `s11_paper_figs.py` and `s10_video_frames.py` read `validation_daily.csv` (daily network and Swarm density ratios) from `$THERMOSWARM_ROOT`. The script that writes this file is not yet included.

### Key parameters

| Stage | Setting |
|---|---|
| Quality control | eccentricity < 0.01; manoeuvre = rise in semi-major axis > 0.3 km; at least 30 clean TLEs per object |
| Inversion | TLE pairs 1–3 days apart; 96 integration points per interval |
| Calibration | quiet = ap ≤ 15 over the interval and previous 24 h; ±30-day window with at least 8 quiet intervals; objects rejected if quiet-time scatter > 0.5 |
| Storms | Kp ≥ 5−, merged within 24 h, maximum Kp ≥ 6− |
| Statistics | bootstrap resamples 2,000 (H1), 500 (H2), 1,000 (H3); τ grid 2–72 h |
| Validation | network objects within ±25 km of Swarm altitude, identical intervals |
| Forecast | 72 h horizon (TLE pairs 66–78 h apart); 4-fold cross-validation grouped by storm; XGBoost with 400 trees, depth 5, learning rate 0.05, absolute-error loss on log R |
| Re-entry | B fitted over previous 5 days; correction relaxes with 2-day e-folding time; re-entry at 120 km |

Random seeds are set in each script, so reruns on the same input data give the same results.

## Data sources

- **TLEs and decay data:** Space-Track.org (18th and 19th Space Defense Squadrons)
- **Kp, ap, F10.7, Hp30:** GFZ Helmholtz Centre for Geosciences, Potsdam (Matzka et al., 2021)
- **Solar wind and SYM-H:** NASA OMNIWeb, Space Physics Data Facility (King & Papitashvili, 2005)
- **Swarm densities:** European Space Agency, via VirES for Swarm (van den IJssel et al., 2020)
- **JB2008 indices:** Space Environment Technologies
- **Atmosphere model:** NRLMSIS 2.1 (Emmert et al., 2022) through the `pymsis` package

## Citation

If you use this code, please cite the paper and this software (see `CITATION.cff`). DOI: to be added after Zenodo archiving.

## License

MIT. See `LICENSE`.
