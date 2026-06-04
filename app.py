"""
app.py
=======

Streamlit frontend for the CC+RF deployable pipeline
(NUST MS Thesis 2026 — §6.5.2).

Loads model_bundle.joblib produced by Cell 23 of
G140_with_save_best_model.ipynb and exposes three interaction modes:
    1. CSV upload (one row per patient, columns are miRNA expression values)
    2. Manual entry (10 numeric fields for the SHAP-selected panel)
    3. Demo (training-median preset with sliders bounded by training quartiles)

For each prediction the app reports:
    • Calibrated probability (Platt scaling A, B from the bundle)
    • Binary verdict at the Youden-J operating threshold τ*
    • Per-feature SHAP attribution bar chart
    • Reliability warning if input falls outside training distribution
    • Disclaimer: research use only.

Compatibility notes:
    - Defensive SHAP unpacking handles both list-of-arrays (SHAP <0.44)
      and 3-D ndarray (SHAP ≥0.44) shap_values() return formats.
    - If the bundle's shap_importance is empty/zero (e.g., a known issue
      with the original notebook's save cell), this app recomputes it at
      startup using a synthetic background derived from train_quartiles.
"""

from pathlib import Path
import io
import time

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Lung Cancer miRNA Detector — CC+RF",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_PATH = Path(__file__).parent / "model_bundle.joblib"

DB_BLUE  = "#1F3A5F"
ACCENT   = "#D44B4B"
GREEN    = "#2E7D32"
AMBER    = "#FFB300"
GRAY     = "#5A5A5A"
LIGHT    = "#F2F4F7"


# ============================================================
# DEFENSIVE SHAP UNPACKING
# ============================================================
def unpack_shap_values(sv):
    """Coerce shap_values() output into a 2-D (n_samples, n_features) array
    of positive-class attributions, regardless of SHAP version.

    SHAP <0.44  → returns [neg_class_2d, pos_class_2d]   (list of 2-D arrays)
    SHAP ≥0.44  → returns (n_samples, n_features, n_classes)  (3-D array)
    Some single-output models may return a 2-D array directly.
    """
    if isinstance(sv, list):
        return np.asarray(sv[1])             # positive class
    arr = np.asarray(sv)
    if arr.ndim == 3:
        return arr[:, :, 1]                  # positive class
    return arr                                # already 2-D


# ============================================================
# LOAD BUNDLE
# ============================================================
@st.cache_resource(show_spinner="Loading model bundle...")
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


@st.cache_resource(show_spinner="Initialising SHAP explainer...")
def get_explainer(_model, _background_df):
    """Cached TreeExplainer keyed on the model identity (underscored args are not hashed)."""
    import shap
    return shap.TreeExplainer(_model, data=_background_df,
                                feature_perturbation="interventional")


@st.cache_data(show_spinner="Computing panel SHAP ranking...")
def get_panel_ranking(_bundle, _explainer):
    """Return dict {feature: mean|SHAP|}.

    Use the bundle's stored values if they're non-trivial; otherwise
    recompute on a synthetic background sampled from training quartiles.
    """
    stored = _bundle.get("shap_importance") or {}
    if stored and any(abs(v) > 1e-9 for v in stored.values()):
        return dict(stored)
    # Fallback: synthesize a small background from training quartiles
    feats = _bundle["feature_names"]
    quartiles = _bundle.get("train_quartiles", {})
    if not quartiles:
        return {f: 0.0 for f in feats}
    rows = []
    for tag in ("min", "q25", "q50", "q75", "max"):
        rows.append([quartiles[f][tag] for f in feats])
    synth_df = pd.DataFrame(rows, columns=feats)
    sv = unpack_shap_values(_explainer.shap_values(synth_df, check_additivity=False))
    mean_abs = np.abs(sv).mean(axis=0)
    return {f: float(v) for f, v in zip(feats, mean_abs)}


# ============================================================
# CORE INFERENCE
# ============================================================
def coerce_to_panel(input_df: pd.DataFrame, panel_features, train_medians):
    """Align an arbitrary input DataFrame to the panel.

    Missing columns are imputed with training medians.
    Returns (aligned_df, list of (feature, imputed_flag, n_nan_filled)).
    """
    out = pd.DataFrame(index=input_df.index, columns=panel_features, dtype=float)
    flags = []
    for f in panel_features:
        if f in input_df.columns:
            col = pd.to_numeric(input_df[f], errors="coerce")
            n_nan = int(col.isna().sum())
            col = col.fillna(train_medians.get(f, 0.0))
            out[f] = col
            flags.append((f, False, n_nan))
        else:
            out[f] = train_medians.get(f, 0.0)
            flags.append((f, True, len(input_df)))
    return out, flags


def predict_one(row_values, bundle, explainer):
    """Run a single sample (1-row DataFrame) through Stage-2 → Platt → Youden."""
    model    = bundle["stage2_model"]
    platt    = bundle["platt_scaler"]
    tau_star = bundle["youden_threshold"]
    feats    = bundle["feature_names"]

    raw_prob = float(model.predict_proba(row_values)[:, 1][0])
    cal_prob = float(platt.predict_proba(np.array([[raw_prob]]))[:, 1][0])
    verdict  = "Cancer-positive" if cal_prob >= tau_star else "Non-cancer"

    sv = unpack_shap_values(explainer.shap_values(row_values, check_additivity=False))
    sv = np.asarray(sv).flatten()[:len(feats)]

    return {
        "raw_prob"  : raw_prob,
        "cal_prob"  : cal_prob,
        "tau_star"  : tau_star,
        "verdict"   : verdict,
        "shap_vals" : sv,
    }


def reliability_band(value, quartile_dict):
    """Categorise a feature value against its training-distribution quartiles."""
    if value < quartile_dict["min"] - 0.5 * quartile_dict["std"]:
        return ("BELOW range", ACCENT)
    if value > quartile_dict["max"] + 0.5 * quartile_dict["std"]:
        return ("ABOVE range", ACCENT)
    if value < quartile_dict["q25"]:
        return ("Lower quartile", AMBER)
    if value > quartile_dict["q75"]:
        return ("Upper quartile", AMBER)
    return ("Interquartile (typical)", GREEN)


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar(bundle):
    st.sidebar.markdown(
        f"<h2 style='color:{DB_BLUE};margin-top:0'>Model Bundle</h2>",
        unsafe_allow_html=True,
    )
    if bundle is None:
        st.sidebar.error(
            "`model_bundle.joblib` not found at the repo root.\n\n"
            "Run Cell 23 of the notebook to produce it, then commit the "
            "file alongside `app.py`."
        )
        return
    st.sidebar.markdown(f"**Pipeline:** {bundle.get('pipeline_id', '—')}")
    st.sidebar.markdown(f"**Trained on:** {bundle.get('trained_on', '—')}")
    st.sidebar.markdown(f"**Panel size:** {bundle.get('n_features', len(bundle['feature_names']))} miRNAs")
    st.sidebar.markdown(f"**Trained:** {(bundle.get('training_date') or '—')[:19]}")
    st.sidebar.markdown(f"**Run ID:** `{bundle.get('run_id', '—')}`")

    im = bundle.get("internal_metrics", {}) or {}
    em = bundle.get("external_metrics") or {}
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<h4 style='color:{DB_BLUE}'>Performance (training run)</h4>",
        unsafe_allow_html=True,
    )
    if im.get("stage2_test_auc") is not None:
        ci = im.get("stage2_test_AUC_CI") or ""
        st.sidebar.markdown(
            f"**Internal test**\n\n"
            f"AUC {im['stage2_test_auc']}%{(' ['+ci+']') if ci else ''}  |  "
            f"Gap {im.get('overfit_gap', '—')}%"
        )
        mt = im.get("metrics_test", {}) or {}
        if mt:
            st.sidebar.markdown(
                f"Sens {mt.get('Sensitivity', '—')}%  |  "
                f"Spec {mt.get('Specificity', '—')}%  |  "
                f"F1 {mt.get('F1', '—')}%"
            )
    if em:
        st.sidebar.markdown(
            f"**External (GSE113486 lung)**\n\n"
            f"AUC {em.get('AUC', '—')}%{(' ['+em.get('AUC_CI', '')+']') if em.get('AUC_CI') else ''}  |  "
            f"Sens {em.get('Sensitivity', '—')}%  |  "
            f"Spec {em.get('Specificity', '—')}%"
        )
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Operating threshold τ\\*:** `{bundle['youden_threshold']:.4f}`")
    st.sidebar.markdown(f"**Platt A:** `{bundle['platt_A']:+.4f}`")
    st.sidebar.markdown(f"**Platt B:** `{bundle['platt_B']:+.4f}`")
    if bundle.get("youden_J_max") is not None:
        st.sidebar.markdown(f"**Youden J\\_max:** `{bundle['youden_J_max']:.4f}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<small style='color:{GRAY}'>"
        "Research use only. Not a clinical decision-making instrument. "
        f"Thesis NUST 2026, §6.5.2."
        "</small>",
        unsafe_allow_html=True,
    )


# ============================================================
# RENDERERS
# ============================================================
def render_header():
    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,{DB_BLUE},#3A5680);
                    padding:24px 28px;border-radius:8px;border-left:8px solid {ACCENT};
                    margin-bottom:12px'>
            <h1 style='color:white;margin:0;font-size:30px;font-weight:700'>
                🧬 Lung Cancer Detection from Serum miRNA Expression
            </h1>
            <p style='color:#cbd6e8;margin:6px 0 0;font-size:15px'>
                Upload expression values for a fixed panel of <b>10 SHAP-shortlisted miRNAs</b>.
                Returns calibrated probability (Platt) + binary verdict (Youden τ*) +
                per-feature SHAP attribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_panel(result, bundle, panel_inputs):
    cal = result["cal_prob"]
    tau = result["tau_star"]
    verdict = result["verdict"]
    feats = bundle["feature_names"]
    sv = result["shap_vals"]
    is_pos = (verdict == "Cancer-positive")

    # Row 1 — calibrated-probability gauge + verdict badge
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"<h4 style='color:{DB_BLUE}'>Calibrated probability of cancer</h4>",
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(7, 1.6))
        ax.barh([0], [1.0], color=LIGHT, edgecolor="#CCC", height=0.55)
        ax.barh([0], [cal], color=(ACCENT if is_pos else DB_BLUE),
                edgecolor="white", height=0.55)
        ax.axvline(tau, color=GRAY, linestyle="--", linewidth=1.5)
        ax.text(tau, 0.85, f"τ* = {tau:.3f}", color=GRAY,
                ha="center", fontsize=9, fontweight="bold")
        ax.text(cal, 0, f" {cal:.3f}", va="center", fontsize=18, fontweight="bold",
                color="white" if cal > 0.15 else GRAY)
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_ylim(-0.5, 1.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Probability", fontsize=10)
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.markdown(
            f"<h4 style='color:{DB_BLUE}'>Binary verdict at τ*</h4>",
            unsafe_allow_html=True,
        )
        bcol = ACCENT if is_pos else GREEN
        st.markdown(
            f"""
            <div style='background:{bcol};color:white;
                        padding:24px;border-radius:8px;text-align:center;
                        font-size:24px;font-weight:700;margin-top:6px'>
                {verdict}
            </div>
            <div style='color:{GRAY};font-size:12px;margin-top:8px;text-align:center'>
                cal. prob {cal:.3f} {'≥' if is_pos else '<'} τ* {tau:.3f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 2 — SHAP per-prediction attribution
    st.markdown(
        f"<h4 style='color:{DB_BLUE};margin-top:20px'>"
        "Per-feature contribution to this prediction (TreeSHAP)"
        "</h4>",
        unsafe_allow_html=True,
    )
    order = np.argsort(-np.abs(sv))
    sv_o = sv[order]
    fn_o = [feats[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [ACCENT if v > 0 else DB_BLUE for v in sv_o]
    bars = ax.barh(range(len(fn_o)), sv_o, color=colors,
                    edgecolor="white", linewidth=1.2)
    ax.set_yticks(range(len(fn_o)))
    ax.set_yticklabels(fn_o, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on log-odds)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for bar, v in zip(bars, sv_o):
        xt = v + (0.005 if v >= 0 else -0.005)
        ax.text(xt, bar.get_y() + bar.get_height()/2,
                f"{v:+.3f}", va="center", ha=("left" if v >= 0 else "right"),
                fontsize=9, fontweight="bold", color=GRAY)
    legend_handles = [
        mpatches.Patch(color=ACCENT, label="Pushes toward cancer-positive"),
        mpatches.Patch(color=DB_BLUE, label="Pushes toward non-cancer"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Row 3 — reliability table
    st.markdown(
        f"<h4 style='color:{DB_BLUE};margin-top:12px'>"
        "Input reliability vs training distribution"
        "</h4>",
        unsafe_allow_html=True,
    )
    rel_rows = []
    quartiles = bundle.get("train_quartiles", {})
    for f in feats:
        v = float(panel_inputs[f].iloc[0])
        if f in quartiles:
            band, _ = reliability_band(v, quartiles[f])
            rng = f"[{quartiles[f]['q25']:.3f}, {quartiles[f]['q75']:.3f}]"
            med = f"{quartiles[f]['q50']:.3f}"
        else:
            band, rng, med = "no reference", "—", "—"
        rel_rows.append({
            "miRNA"            : f,
            "Input value"      : f"{v:.3f}",
            "Median (train)"   : med,
            "Reference IQR"    : rng,
            "Band"             : band,
            "SHAP impact"      : f"{sv[feats.index(f)]:+.3f}",
        })
    st.dataframe(pd.DataFrame(rel_rows), hide_index=True, use_container_width=True)


def warn_out_of_range(panel_inputs, bundle):
    quartiles = bundle.get("train_quartiles", {})
    if not quartiles:
        return
    flagged = []
    for f in bundle["feature_names"]:
        v = float(panel_inputs[f].iloc[0])
        q = quartiles.get(f)
        if q is None:
            continue
        lo = q["min"] - 0.5 * q["std"]
        hi = q["max"] + 0.5 * q["std"]
        if v < lo or v > hi:
            flagged.append((f, v, q["min"], q["max"]))
    if flagged:
        st.warning(
            "**Input outside training distribution for the following features.** "
            "Model prediction may be unreliable:\n\n" +
            "\n".join(f"- `{f}` = {v:.3f}  (training min/max: {lo:.3f} / {hi:.3f})"
                       for f, v, lo, hi in flagged)
        )


# ============================================================
# MAIN
# ============================================================
def main():
    render_header()
    bundle = load_bundle()
    render_sidebar(bundle)

    if bundle is None:
        st.error(
            "Cannot run predictions: `model_bundle.joblib` is not present at the repo root.\n\n"
            "**To produce it:**\n\n"
            "1. Open `G140_with_save_best_model.ipynb` in Colab.\n"
            "2. Run all cells through Cell 23 (the save-best-model cell).\n"
            "3. Download `outputs/g140_v10_default/model_bundle.joblib`.\n"
            "4. Commit it to this repo at the same level as `app.py`."
        )
        st.stop()

    feats   = bundle["feature_names"]
    medians = bundle["train_medians"]

    # Background = the median row (single point; SHAP averages over it)
    bg = pd.DataFrame([medians], columns=feats)
    explainer = get_explainer(bundle["stage2_model"], bg)

    # Panel ranking: prefer bundle's stored shap_importance, else recompute
    shap_imp = get_panel_ranking(bundle, explainer)

    tabs = st.tabs(["📁 CSV upload", "✏️ Manual entry", "🎛️ Demo (sliders)"])

    # --------------------------------------------------------
    # TAB 1 — CSV UPLOAD
    # --------------------------------------------------------
    with tabs[0]:
        # Panel composition banner
        st.markdown(
            f"<div style='background:{LIGHT};padding:14px 18px;border-radius:6px;"
            f"border-left:5px solid {DB_BLUE};margin-bottom:14px'>"
            f"<b style='color:{DB_BLUE};font-size:15px'>Required input — 10 miRNA expression values</b><br>"
            f"<small style='color:{GRAY}'>"
            "Upload a CSV with one row per patient and the following 10 column "
            "headers (extra columns are ignored; column order doesn't matter)."
            "</small></div>",
            unsafe_allow_html=True,
        )

        quartiles = bundle.get("train_quartiles", {})
        ordered_feats = sorted(feats, key=lambda f: -shap_imp.get(f, 0.0))
        panel_rows = []
        for rank, f in enumerate(ordered_feats, 1):
            q = quartiles.get(f, {})
            panel_rows.append({
                "Rank"                       : rank,
                "miRNA (CSV column header)"  : f,
                "Mean |SHAP|"                : f"{shap_imp.get(f, 0.0):.4f}",
                "Training median"            : (f"{q.get('q50', float('nan')):.3f}" if q else "—"),
                "Training range [min, max]"  : (f"[{q['min']:.3f}, {q['max']:.3f}]" if q else "—"),
            })
        st.dataframe(pd.DataFrame(panel_rows), hide_index=True, use_container_width=True)

        # Template download — always reflects the bundle currently loaded
        template_df = pd.DataFrame(
            [{f: float(bundle["train_medians"].get(f, 0.0)) for f in feats}]
        )
        template_df.insert(0, "Patient_ID", "SAMPLE_001")
        buf = io.StringIO()
        template_df.to_csv(buf, index=False)

        ca, cb = st.columns([1, 1])
        with ca:
            st.download_button(
                "📄 Download CSV template (this trained model's exact schema)",
                buf.getvalue(),
                file_name="lung_mirna_panel_template.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with cb:
            st.markdown(
                f"<small style='color:{GRAY}'>"
                "The template contains the median value for each panel miRNA. "
                "Replace each row with measured expression values for one patient."
                "</small>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        uploaded = st.file_uploader(
            "Upload your CSV (one row per patient, 10 miRNA columns above)",
            type=["csv"], key="upload",
        )
        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
                st.stop()
            st.success(f"Loaded {len(input_df)} samples, {input_df.shape[1]} columns")
            aligned, flags = coerce_to_panel(input_df, feats, medians)
            missing = [f for f, was, _ in flags if was]
            if missing:
                st.warning(
                    f"⚠️ **{len(missing)} of {len(feats)} required miRNAs were missing** "
                    f"from your upload and were imputed with training medians:\n\n" +
                    "\n".join(f"- `{f}`" for f in missing) +
                    "\n\nPredictions for samples with imputed values may be unreliable. "
                    "Please verify your CSV column headers exactly match the panel above."
                )

            t0 = time.time()
            raws = bundle["stage2_model"].predict_proba(aligned)[:, 1]
            cals = bundle["platt_scaler"].predict_proba(raws.reshape(-1, 1))[:, 1]
            preds = (cals >= bundle["youden_threshold"]).astype(int)
            elapsed_ms = (time.time() - t0) * 1000

            out = aligned.copy()
            out["raw_prob"] = raws
            out["cal_prob"] = cals
            out["verdict_at_tau"] = ["Cancer-positive" if p else "Non-cancer" for p in preds]
            st.markdown(
                f"<h4 style='color:{DB_BLUE}'>Predictions  "
                f"<small style='color:{GRAY};font-weight:400'>"
                f"({elapsed_ms:.0f} ms)</small></h4>",
                unsafe_allow_html=True,
            )
            st.dataframe(out, hide_index=False, use_container_width=True)

            dbuf = io.StringIO()
            out.to_csv(dbuf, index=False)
            st.download_button("Download predictions as CSV", dbuf.getvalue(),
                                file_name="predictions.csv", mime="text/csv")

            if len(out) == 1:
                row = aligned.iloc[[0]]
                result = predict_one(row, bundle, explainer)
                st.markdown("---")
                warn_out_of_range(row, bundle)
                render_prediction_panel(result, bundle, row)

    # --------------------------------------------------------
    # TAB 2 — MANUAL ENTRY
    # --------------------------------------------------------
    with tabs[1]:
        st.markdown(
            f"<small style='color:{GRAY}'>"
            "Enter expression values for the 10 SHAP-shortlisted miRNAs. "
            "Defaults are training medians; replace with measured values."
            "</small>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        manual_vals = {}
        for i, f in enumerate(ordered_feats):
            with cols[i % 2]:
                manual_vals[f] = st.number_input(
                    f, value=float(medians.get(f, 0.0)),
                    format="%.4f", key=f"manual_{i}",
                )
        if st.button("Predict from manual entry", type="primary"):
            row = pd.DataFrame([manual_vals], columns=feats)
            result = predict_one(row, bundle, explainer)
            st.markdown("---")
            warn_out_of_range(row, bundle)
            render_prediction_panel(result, bundle, row)

    # --------------------------------------------------------
    # TAB 3 — DEMO SLIDERS
    # --------------------------------------------------------
    with tabs[2]:
        st.markdown(
            "Adjust the sliders below to see how each miRNA value affects "
            "the prediction. Slider ranges follow each feature's training "
            "distribution (min − 0.5σ to max + 0.5σ)."
        )
        cols = st.columns(2)
        demo_vals = {}
        for i, f in enumerate(ordered_feats):
            q = quartiles.get(f, {"min": -2.0, "max": 2.0, "q50": 0.0, "std": 1.0})
            rng_lo = float(q["min"] - 0.5 * q["std"])
            rng_hi = float(q["max"] + 0.5 * q["std"])
            with cols[i % 2]:
                demo_vals[f] = st.slider(
                    f, min_value=rng_lo, max_value=rng_hi,
                    value=float(q["q50"]), step=(rng_hi - rng_lo) / 200,
                    format="%.3f", key=f"demo_{i}",
                )
        row = pd.DataFrame([demo_vals], columns=feats)
        result = predict_one(row, bundle, explainer)
        st.markdown("---")
        render_prediction_panel(result, bundle, row)


if __name__ == "__main__":
    main()
