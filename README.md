# Lung Cancer Detection — CC+RF Clinical Decision Support

Advanced clinical-software Streamlit frontend for the explainable
two-stage classifier defended in the NUST MS Thesis (2026, §6.5.2):
**Web-Based Explainable Machine Learning Detection of Lung Cancer
Through Free Circulating Serum miRNAs**.

The deployed pipeline is **ClusterCentroids + Random Forest (CC+RF)**.

> **Research use only. Not a clinical decision-making instrument.**

---

## What this app shows

A rich patient-report layout for each prediction:

1. **Radial probability gauge** — Plotly-rendered with color zones
   (green → amber → red) and the Youden τ\* threshold marker baked in.
2. **Clinical verdict tile** — large badge with risk category, decision
   margin, and raw RF probability.
3. **SHAP waterfall plot** — cumulative attribution showing how each
   miRNA's contribution accumulates from a 0.5 baseline to the final
   prediction.
4. **SHAP force diagram** — horizontal red/blue stack showing net
   positive vs negative contributions.
5. **Per-miRNA density grid** — 10 small-multiples plots showing each
   miRNA's training-distribution density with the patient's value marked
   and percentile rank annotated.
6. **Diagnostic detail table** — every miRNA with patient value, training
   median, IQR, percentile, distribution band, and SHAP contribution.

## Four interaction modes (tabs)

| Tab | Use case |
|---|---|
| 📁 **CSV Upload** | Batch prediction; one row per patient; bulk results table + detailed expandable reports per patient |
| ✏️ **Manual Entry** | Single-patient form; 10 number-input fields ordered by SHAP importance |
| 🎛️ **Demo + Presets** | Interactive sliders with 4 pre-loaded patient profiles (Healthy / Borderline / Elevated / Strong-signal) |
| 🔬 **Side-by-Side Comparison** | Two patients in parallel; per-miRNA delta table; clinical-discussion mode |

## Achieved performance (this trained bundle)

| Metric | Internal test | External (GSE113486 lung) |
|---|---|---|
| AUC | 99.97% | **98.65%** |
| Sensitivity @ τ\* | 99.4% | 70.0% |
| Specificity @ τ\* | 99.5% | 100.0% |
| Train-test overfit gap | 0.03% | — |

**Calibration:** Platt scaling on internal validation (A = +6.5307,
B = −3.3269). Youden-J operating threshold **τ\* = 0.2204** (J_max = 0.9890).

---

## Repository structure

```
.
├── app.py                    # Streamlit frontend (advanced)
├── model_bundle.joblib       # Trained CC+RF + Platt + Youden (≈75 KB)
├── requirements.txt          # Inference dependencies (Streamlit Cloud)
├── .gitignore
└── README.md
```

The notebook that produced the bundle is kept separately — not needed
for deployment.

---

## Visual design

- **Theme:** clinical-software aesthetic (deep blue `#1E3A5F` + slate
  `#475569` + medical green/amber/red semantic colors).
- **Typography:** Inter / system-ui font stack.
- **Components:** card-surface layout with subtle shadows, rounded
  corners (12px), animated hover states, custom-styled tabs and metrics.
- **Hero:** SVG DNA double-helix on a clinical gradient background.
- **Plots:** matplotlib for SHAP (waterfall, force, density grid) +
  Plotly for the interactive gauge.

---

## Deployment

### From notebook to live URL

1. Run `G140_with_save_best_model.ipynb` through Cell 23 in Colab.
   Optionally apply the SHAP-importance patch cell if the run produced
   an empty `shap_importance` field (known SHAP 0.44+ API issue).
2. Download `outputs/g140_v10_default/model_bundle.joblib` from the
   Colab file panel.
3. On GitHub, create a public repo and upload these files at the root:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `model_bundle.joblib`
4. Sign in to **streamlit.io/cloud** with GitHub, click **Create app**,
   point at this repo, branch `main`, main file `app.py`, **Deploy**.

First build takes 4–7 minutes (most of it is `shap` pulling in `numba`).
Subsequent redeploys after a push take ~30 seconds.

### Test locally first (optional)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501.

---

## Input format

**Exactly 10 miRNA expression values per patient.** Required column
headers are displayed in the app's CSV Upload tab (read live from the
loaded bundle). One row per patient.

- Order doesn't matter — the app aligns by column name.
- Extra columns (Patient_ID, Label, other miRNAs) are silently ignored.
- Missing panel miRNAs are imputed with training medians and flagged.
- Values: 3D-Gene Toray platform-processed, quantile-normalised log₂
  expression (same scale as training).

In-app template download: one click in the CSV Upload tab produces a
CSV with the bundle's exact 10 column headers and training-median
pre-filled values. Replace each row with measured patient data.

---

## What's in the joblib bundle

| Key | Purpose |
|---|---|
| `feature_names` (10) | SHAP-shortlisted miRNA names |
| `stage2_model` | Trained Random Forest (n=200, depth=6) |
| `platt_A`, `platt_B`, `platt_scaler` | Calibration parameters |
| `youden_threshold` | Operating threshold τ\* |
| `train_medians` | Per-feature median (imputation) |
| `train_quartiles` | Per-feature q25/q50/q75/min/max/mean/std |
| `shap_importance` | Mean \|SHAP\| per feature (panel rank) |
| `internal_metrics`, `external_metrics` | Performance audit dicts |
| `training_date`, `run_id`, `random_seed` | Provenance |

---

## What was added beyond the original notebook

The notebook outputs **raw Random Forest probabilities** at a default
**0.5 threshold**. The deployment adds two transformations from thesis
§3.10:

1. **Platt scaling** — 1-parameter logistic fitted on the internal
   validation split. Calibrates the probability distribution.
2. **Youden-J threshold τ\*** — replaces the arbitrary 0.5 cutoff with
   the threshold maximising sensitivity + specificity − 1 on the
   calibrated validation curve.

Both are monotonic 1-D transforms, so internal AUC and pipeline ranking
are unchanged — only the calibrated probability scale and the binary
verdict at τ\* differ.

---

## Common issues

| Symptom | Cause / Fix |
|---|---|
| `model_bundle.joblib not found` | Bundle missing or in a subfolder. Confirm at repo root via GitHub. |
| Panel composition shows all 0.0000 for SHAP | Bundle's `shap_importance` is empty. Either re-run the patch cell in the notebook, or just let the app's runtime SHAP recovery compute it on first load. |
| Build fails on `shap` install | `numba` install sometimes times out. Click **Reboot app** in the Streamlit Cloud manage panel. |
| Predictions all return ≈0.5 | Column names don't match training schema. Check case-sensitivity (e.g. `hsa-miR-5100`). |
| Demo sliders feel sluggish | Each slider drag re-renders 4 matplotlib figures plus a Plotly gauge. Throttle by releasing the slider between adjustments. |
| Comparison tab shows identical patients | Click "Load preset → Patient A" / "Patient B" buttons to refresh from selected preset. |

---

## Citation

> [Author Name]. *Web-Based Explainable Machine Learning Detection of
> Lung Cancer Through Free Circulating Serum miRNAs.* MS Thesis,
> National University of Sciences and Technology (NUST), Islamabad,
> 2026.

---

## License

Specify your license (MIT, Apache-2.0, or institutional default).
