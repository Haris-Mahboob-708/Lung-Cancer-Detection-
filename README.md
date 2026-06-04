# Lung Cancer Detection — CC+RF Web Inference

Streamlit web frontend for the explainable two-stage classifier developed
in the NUST MS Thesis (2026): **Web-Based Explainable Machine Learning
Detection of Lung Cancer Through Free Circulating Serum miRNAs**.

The deployed pipeline is **ClusterCentroids + Random Forest (CC+RF)**,
selected as the deployable model per Chapter 6 §6.5.2 by a four-criterion
rule (high external AUC, named-feature interpretability, smallest overfit
gap among interpretable candidates, 100% external specificity).

> **Research use only. Not a clinical decision-making instrument.**

---

## Achieved performance (from the bundled training run)

| Metric | Internal test (GSE137140) | External (GSE113486 lung subset) |
|---|---|---|
| AUC | 99.97% | **98.65%** |
| Sensitivity at τ\* | 99.4% | 70.0% |
| Specificity at τ\* | 99.5% | 100.0% |
| Train-test overfit gap | 0.03% | — |

**Calibration parameters:** Platt scaling fitted on internal validation
(n=375). A = +6.5307, B = −3.3269. Youden-J operating threshold
**τ\* = 0.2204** (J_max = 0.9890).

---

## Repository contents

```
.
├── app.py                    # Streamlit frontend
├── model_bundle.joblib       # Trained CC+RF + Platt + Youden (≈75 KB)
├── requirements.txt          # Inference dependencies
├── .gitignore
└── README.md                 # This file
```

The notebook that produced the bundle (`G140_with_save_best_model.ipynb`)
is kept separately — it's not needed for deployment.

---

## End-to-end deployment

### 1. Produce the model bundle (in Colab)

Open `G140_with_save_best_model.ipynb` in Colab:

1. Mount Google Drive (Cell 6).
2. Run cells 0–22 (full 16-pipeline training loop).
3. Run **Cell 23** (the save-best-model cell). This writes
   `./outputs/g140_v10_default/model_bundle.joblib`.
4. **If you saw a SHAP error in Cell 23** (`only length-1 arrays can be
   converted to Python scalars`), the bundle was saved but its
   `shap_importance` field is empty. Run the patch cell from
   `notebook_patch_cell.py` immediately after Cell 23 to fix it in place.
5. Download the bundle from Colab's file panel (it's at
   `outputs/g140_v10_default/model_bundle.joblib`).

### 2. Create the GitHub repository

```bash
git clone https://github.com/<your-username>/lung-mirna-detector.git
cd lung-mirna-detector

# Copy the files from this deployment package into the repo root
cp <path-to>/app.py .
cp <path-to>/requirements.txt .
cp <path-to>/.gitignore .
cp <path-to>/README.md .

# Add the bundle you downloaded from Colab
cp ~/Downloads/model_bundle.joblib .

git add .
git commit -m "Initial deploy of CC+RF lung-cancer miRNA classifier"
git push origin main
```

### 3. Test locally (optional, 5 minutes)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. You should see your dark-blue header, the
sidebar showing the bundle's metrics, and three tabs (CSV upload,
Manual entry, Demo sliders).

### 4. Deploy on Streamlit Community Cloud

1. Sign in at https://streamlit.io/cloud with your GitHub account.
2. Click **New app**.
3. Fill in:
   - **Repository:** `<your-username>/lung-mirna-detector`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**.

First build takes 3–6 minutes (most of it is installing `shap`, which
pulls `numba`). You'll get a URL like
`https://<your-app-slug>.streamlit.app`.

---

## Input format

**Exactly 10 miRNA expression values per patient.** Upload a CSV with
the 10 column headers shown in the app's "📁 CSV upload" tab (read live
from the loaded bundle). One row per patient.

**Column rules:**

- Order doesn't matter — the app aligns by column name.
- Extra columns (Patient_ID, Label, other miRNAs) are silently ignored.
- Missing panel miRNAs are imputed with training medians and explicitly
  flagged in the UI as unreliable.
- Values must be on the same scale as training: 3D-Gene Toray
  platform-processed, quantile-normalised log₂ expression.

In-app template download produces a CSV pre-filled with training-median
values for each panel miRNA — replace each row with the patient's
measured expression values.

---

## What's in the joblib bundle (`model_bundle.joblib`)

| Key | Type | Purpose |
|---|---|---|
| `feature_names` | list[str] (10) | 10 SHAP-shortlisted miRNA names |
| `stage2_model` | RandomForestClassifier | Trained Stage-2 predictor |
| `platt_A`, `platt_B` | float | Platt-scaling coefficients |
| `platt_scaler` | LogisticRegression | Full sklearn calibrator |
| `youden_threshold` | float | Operating threshold τ\* |
| `youden_J_max` | float | J = Sens + Spec − 1 at τ\* |
| `train_medians` | dict[str, float] | Per-feature median (imputation) |
| `train_quartiles` | dict[str, dict] | Per-feature q25/q50/q75/min/max/mean/std |
| `shap_importance` | dict[str, float] | Mean \|SHAP\| per feature (UI rank) |
| `internal_metrics` | dict | AUC / Sens / Spec / F1 / overfit gap |
| `external_metrics` | dict | Same on GSE113486 lung subset |
| `training_date` | ISO timestamp | Provenance |
| `run_id`, `random_seed` | str / int | Reproducibility identifiers |

---

## What was added beyond the original notebook

The notebook's main pipeline loop (Cell 21) produces **raw Random Forest
probabilities** at a default **0.5 threshold**. For deployment, Cell 23
adds two transformations specified in the thesis §3.10 but absent from
the notebook loop:

1. **Platt scaling** — 1-parameter logistic on the raw RF scores,
   fitted on the internal validation split.
2. **Youden-J threshold τ\*** — replaces the arbitrary 0.5 cutoff with
   the threshold maximising sensitivity + specificity − 1 on the
   calibrated validation curve.

Both are monotonic 1-D transforms, so internal AUC and pipeline ranking
are unchanged. The base CC+RF model is the exact one whose external AUC
of 98.65% is reported in the thesis — only the calibrated probability
scale and the binary verdict at τ\* differ from the notebook's raw
output.

---

## Common issues

**`model_bundle.joblib not found`**
→ Bundle wasn't committed to the repo root. Confirm it's at the same
level as `app.py`.

**Panel composition table shows all SHAP values as `0.0000`**
→ Your bundle was saved before the SHAP fix. Either re-run Cell 23 in a
fresh notebook session (after the fix is in place) or apply
`notebook_patch_cell.py` to update the existing bundle's
`shap_importance` field in place.

**`ModuleNotFoundError: No module named 'boruta'` on Streamlit Cloud**
→ You added `boruta` to `requirements.txt`. Remove it — Boruta is
training-time only. The deployed app does not need it.

**Predictions all return ≈0.5**
→ Input column names don't match training schema; everything is being
imputed to median. Check column-header spelling exactly (case-sensitive,
`hsa-miR-` prefix and hyphens preserved).

**Build fails on `shap` install**
→ `numba` (a SHAP dependency) sometimes times out on Streamlit Cloud.
Retry; if persistent, pin `numba==0.58.1` in `requirements.txt`.

---

## Citation

> [Author Name]. *Web-Based Explainable Machine Learning Detection of
> Lung Cancer Through Free Circulating Serum miRNAs.* MS Thesis,
> National University of Sciences and Technology (NUST), Islamabad,
> 2026.

---

## License

Specify your license (MIT, Apache-2.0, or institutional default).
