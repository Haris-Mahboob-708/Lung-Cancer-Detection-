"""
app.py — Lung Cancer miRNA Detector (CC+RF Deployable Pipeline)
================================================================

Clinical-decision-support Streamlit frontend for the CC+RF pipeline
defended in the NUST MS Thesis (2026, §6.5.2).

Three input methods + a comparison mode, all flowing into a unified
patient-report layout with:
    • Plotly radial probability gauge with τ* threshold marker
    • Animated verdict badge with risk-category tile
    • SHAP waterfall plot (cumulative feature attribution)
    • SHAP force plot (positive/negative push diagram)
    • Per-feature density grid (training distribution + patient marker)
    • Reliability table with percentile ranks vs training distribution
    • Sample patient presets (Healthy / Cancer / Borderline / High-risk)
    • Side-by-side patient comparison
"""

from __future__ import annotations
from pathlib import Path
import io
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyBboxPatch
import matplotlib.patches as mpatches
import plotly.graph_objects as go

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     PAGE CONFIG & CONSTANTS                        ║
# ╚══════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="Lung-miRNA Clinical Decision Support",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_PATH = Path(__file__).parent / "model_bundle.joblib"

# Clinical-software palette
PRIMARY      = "#1E3A5F"   # deep clinical blue
PRIMARY_DARK = "#152D4A"
PRIMARY_2    = "#3B82F6"   # accent blue
SLATE        = "#475569"
SLATE_LIGHT  = "#94A3B8"
SLATE_BG     = "#F1F5F9"
SURFACE      = "#FFFFFF"
TEXT         = "#0F172A"
TEXT_MUTED   = "#64748B"

OK_GREEN     = "#059669"
OK_GREEN_BG  = "#D1FAE5"
WARN_AMBER   = "#D97706"
WARN_AMBER_BG = "#FEF3C7"
DANGER_RED   = "#DC2626"
DANGER_RED_BG = "#FECACA"
INFO_BLUE    = "#0284C7"


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          CUSTOM CSS                                ║
# ╚══════════════════════════════════════════════════════════════════╝
def inject_css():
    st.markdown(f"""
    <style>
        /* App background */
        .stApp {{
            background: linear-gradient(180deg, {SLATE_BG} 0%, #FFFFFF 100%);
        }}
        /* Hide Streamlit default header */
        header[data-testid="stHeader"] {{
            background: transparent;
        }}
        /* Sidebar background */
        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }}
        /* Card surface used for content blocks */
        .clinical-card {{
            background: {SURFACE};
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
            margin-bottom: 14px;
        }}
        .clinical-card h4 {{
            color: {PRIMARY};
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border-bottom: 2px solid {SLATE_BG};
            padding-bottom: 8px;
        }}
        /* Pillbox / badge */
        .pill {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {SLATE_BG};
            padding: 4px;
            border-radius: 10px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            color: {SLATE};
            font-weight: 500;
            padding: 8px 16px;
            border-radius: 8px;
            transition: all 0.15s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background: {SURFACE};
            color: {PRIMARY};
            box-shadow: 0 1px 3px rgba(15,23,42,0.1);
            font-weight: 600;
        }}
        /* Buttons */
        .stButton button {{
            background: {PRIMARY};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 8px 18px;
            transition: all 0.15s ease;
        }}
        .stButton button:hover {{
            background: {PRIMARY_DARK};
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(30,58,95,0.2);
        }}
        /* Download buttons */
        .stDownloadButton button {{
            background: {SURFACE};
            color: {PRIMARY};
            border: 1.5px solid {PRIMARY};
        }}
        .stDownloadButton button:hover {{
            background: {PRIMARY};
            color: white;
        }}
        /* Dataframe styling */
        [data-testid="stDataFrame"] {{
            border-radius: 8px;
            overflow: hidden;
        }}
        /* Metric tiles */
        [data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }}
        [data-testid="stMetricLabel"] {{
            color: {SLATE} !important;
            font-size: 11px !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        [data-testid="stMetricValue"] {{
            color: {PRIMARY} !important;
            font-size: 28px !important;
            font-weight: 700 !important;
        }}
        /* Hide default Streamlit footer */
        footer {{ visibility: hidden; }}
        /* Custom divider */
        .clinical-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
            margin: 20px 0;
        }}
        /* Section heading */
        .section-h {{
            color: {PRIMARY};
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 18px;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 2px solid {PRIMARY_2};
            display: inline-block;
        }}
    </style>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                         DNA-HELIX SVG                              ║
# ╚══════════════════════════════════════════════════════════════════╝
def dna_helix_svg(width=180, height=70) -> str:
    n_points = 60
    amp = 22
    period = 70
    margin = 8
    avail = width - 2 * margin
    p1, p2, rungs = [], [], []
    for i in range(n_points):
        x = margin + i * avail / (n_points - 1)
        phase = (x - margin) / period * 2 * math.pi
        y1 = height/2 + amp * math.sin(phase)
        y2 = height/2 + amp * math.sin(phase + math.pi)
        p1.append(f"{x:.1f},{y1:.1f}")
        p2.append(f"{x:.1f},{y2:.1f}")
        if i % 5 == 0:
            rungs.append((x, y1, y2))
    rungs_svg = "".join(
        f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" '
        f'stroke="#94A3B8" stroke-width="1.2" opacity="0.55"/>'
        for x, y1, y2 in rungs
    )
    return f'''
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="dh1" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#60A5FA"/>
                <stop offset="100%" stop-color="#3B82F6"/>
            </linearGradient>
            <linearGradient id="dh2" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#1E3A5F"/>
                <stop offset="100%" stop-color="#3B82F6"/>
            </linearGradient>
        </defs>
        {rungs_svg}
        <path d="M {' L '.join(p1)}" stroke="url(#dh1)" fill="none" stroke-width="2.5" stroke-linecap="round"/>
        <path d="M {' L '.join(p2)}" stroke="url(#dh2)" fill="none" stroke-width="2.5" stroke-linecap="round"/>
    </svg>
    '''


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    DEFENSIVE SHAP UNPACKING                        ║
# ╚══════════════════════════════════════════════════════════════════╝
def unpack_shap_values(sv):
    """Coerce shap_values() output to a 2-D (n_samples, n_features) array
    of positive-class attributions, regardless of SHAP version."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    arr = np.asarray(sv)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       BUNDLE LOADING                               ║
# ╚══════════════════════════════════════════════════════════════════╝
@st.cache_resource(show_spinner="Loading clinical model bundle...")
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


@st.cache_resource(show_spinner="Initialising SHAP explainer...")
def get_explainer(_model, _background_df):
    import shap
    return shap.TreeExplainer(_model, data=_background_df,
                                feature_perturbation="interventional")


@st.cache_data(show_spinner="Computing panel SHAP ranking...")
def get_panel_ranking(_bundle, _explainer):
    stored = _bundle.get("shap_importance") or {}
    if stored and any(abs(v) > 1e-9 for v in stored.values()):
        return dict(stored)
    feats = _bundle["feature_names"]
    quartiles = _bundle.get("train_quartiles", {})
    if not quartiles:
        return {f: 0.0 for f in feats}
    rows = []
    for tag in ("min", "q25", "q50", "q75", "max"):
        rows.append([quartiles[f][tag] for f in feats])
    sd = pd.DataFrame(rows, columns=feats)
    sv = unpack_shap_values(_explainer.shap_values(sd, check_additivity=False))
    mean_abs = np.abs(sv).mean(axis=0)
    return {f: float(v) for f, v in zip(feats, mean_abs)}


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       UTILITY FUNCTIONS                            ║
# ╚══════════════════════════════════════════════════════════════════╝
def normal_cdf(x, mean, std):
    """Standard-normal CDF via stdlib math.erf."""
    if std <= 0:
        return 0.5
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))


def coerce_to_panel(input_df, panel_features, train_medians):
    """Align input DataFrame to the 10-feature panel.
    Returns (aligned_df, list of (feature, imputed_flag, n_nan_filled))."""
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
    """Run a single sample through Stage-2 → Platt → Youden."""
    model    = bundle["stage2_model"]
    platt    = bundle["platt_scaler"]
    tau_star = bundle["youden_threshold"]
    feats    = bundle["feature_names"]
    raw = float(model.predict_proba(row_values)[:, 1][0])
    cal = float(platt.predict_proba(np.array([[raw]]))[:, 1][0])
    verdict = "Cancer-positive" if cal >= tau_star else "Non-cancer"
    sv = unpack_shap_values(explainer.shap_values(row_values, check_additivity=False))
    sv = np.asarray(sv).flatten()[:len(feats)]
    # Risk category
    if cal < tau_star * 0.5:
        category, cat_color = "Low risk", OK_GREEN
    elif cal < tau_star:
        category, cat_color = "Below threshold", OK_GREEN
    elif cal < (tau_star + 1) / 2:
        category, cat_color = "Elevated risk", WARN_AMBER
    else:
        category, cat_color = "High risk", DANGER_RED
    return {
        "raw_prob": raw, "cal_prob": cal, "tau_star": tau_star,
        "verdict": verdict, "shap_vals": sv,
        "category": category, "cat_color": cat_color,
    }


@st.cache_data(show_spinner=False)
def _predict_cached(_bundle, _explainer, row_key: tuple) -> dict:
    """Cached single-patient prediction keyed by a rounded float tuple.

    _bundle and _explainer have leading underscores so Streamlit skips
    hashing them (they are stable @st.cache_resource objects).
    Only row_key is hashed, so the result is reused whenever the user
    returns to slider values they've already visited — avoiding redundant
    SHAP tree-traversal and matplotlib figure regeneration.
    """
    feats = _bundle["feature_names"]
    row = pd.DataFrame([list(row_key)], columns=feats)
    return predict_one(row, _bundle, _explainer)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       PLOTLY GAUGE                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_radial_gauge(probability: float, tau_star: float, is_pos: bool) -> go.Figure:
    """Plotly radial gauge with τ* threshold marker and color zones."""
    bar_color = DANGER_RED if is_pos else OK_GREEN
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={
            "valueformat": ".3f",
            "font": {"size": 38, "color": PRIMARY, "family": "Inter, sans-serif"},
        },
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {
                "range": [0, 1],
                "tickwidth": 1,
                "tickcolor": SLATE_LIGHT,
                "tickfont": {"size": 11, "color": SLATE},
                "tickvals": [0, 0.25, 0.5, 0.75, 1.0],
            },
            "bar": {"color": bar_color, "thickness": 0.32},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, tau_star * 0.5], "color": OK_GREEN_BG},
                {"range": [tau_star * 0.5, tau_star], "color": "#FEF9C3"},
                {"range": [tau_star, (tau_star + 1) / 2], "color": WARN_AMBER_BG},
                {"range": [(tau_star + 1) / 2, 1], "color": DANGER_RED_BG},
            ],
            "threshold": {
                "line": {"color": PRIMARY, "width": 4},
                "thickness": 0.92,
                "value": tau_star,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            x=0.5, y=-0.08, xref="paper", yref="paper",
            text=f"τ* = {tau_star:.3f}",
            showarrow=False, font=dict(size=11, color=PRIMARY, family="Inter"),
        )]
    )
    return fig


# ╔══════════════════════════════════════════════════════════════════╗
# ║                  SHAP WATERFALL & FORCE PLOTS                      ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_shap_waterfall(shap_vals, feature_names, feature_values, base_value):
    """Waterfall: base → +/- contributions → final."""
    order = np.argsort(-np.abs(shap_vals))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cumulative = base_value
    bar_left = []
    bar_w = []
    bar_color = []
    labels = []
    for idx in order:
        v = shap_vals[idx]
        if v >= 0:
            bar_left.append(cumulative)
            bar_w.append(v)
            bar_color.append(DANGER_RED)
        else:
            bar_left.append(cumulative + v)
            bar_w.append(-v)
            bar_color.append(PRIMARY_2)
        cumulative += v
        labels.append(f"{feature_names[idx]}\n= {feature_values[idx]:.3f}")
    y = np.arange(len(order))
    ax.barh(y, bar_w, left=bar_left, color=bar_color,
            edgecolor="white", linewidth=1.5, height=0.65)
    # Connecting lines between bars
    for i in range(len(order) - 1):
        idx = order[i]
        v = shap_vals[idx]
        x_end = bar_left[i] + bar_w[i] if v >= 0 else bar_left[i]
        ax.plot([x_end, x_end], [i + 0.32, i + 0.68], color=SLATE_LIGHT,
                linewidth=1, linestyle=":", zorder=1)
    # SHAP value annotations
    for i, idx in enumerate(order):
        v = shap_vals[idx]
        center = bar_left[i] + bar_w[i] / 2
        sign = "+" if v >= 0 else "−"
        ax.text(center, i, f"{sign}{abs(v):.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    final = base_value + shap_vals.sum()
    ax.axvline(base_value, color=SLATE, linestyle="--", linewidth=1.4, alpha=0.7,
                label=f"E[f(x)] = {base_value:.3f}")
    ax.axvline(final, color=PRIMARY, linewidth=2.5,
                label=f"f(x) = {final:.3f}")
    ax.set_xlabel("Model output (raw RF probability)", fontsize=10, color=SLATE)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
                edgecolor="#E2E8F0")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(SLATE_LIGHT)
    ax.spines["bottom"].set_color(SLATE_LIGHT)
    ax.tick_params(colors=SLATE)
    ax.grid(axis="x", alpha=0.2, linestyle=":")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def render_shap_force(shap_vals, feature_names, base_value):
    """Horizontal force plot: positive (red) and negative (blue) stacks."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    pos = sorted([(feature_names[i], v) for i, v in enumerate(shap_vals) if v > 0],
                  key=lambda x: -x[1])
    neg = sorted([(feature_names[i], v) for i, v in enumerate(shap_vals) if v < 0],
                  key=lambda x: x[1])
    # Positive stack (right of base)
    x_cursor = base_value
    for name, v in pos:
        ax.barh(0, v, left=x_cursor, color=DANGER_RED, edgecolor="white",
                linewidth=1.2, height=0.55)
        if v > 0.015:
            ax.text(x_cursor + v / 2, 0,
                    name.replace("hsa-miR-", "miR-")[:14],
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
        x_cursor += v
    # Negative stack (left of base)
    x_cursor = base_value
    for name, v in neg:
        ax.barh(0, v, left=x_cursor, color=PRIMARY_2, edgecolor="white",
                linewidth=1.2, height=0.55)
        if abs(v) > 0.015:
            ax.text(x_cursor + v / 2, 0,
                    name.replace("hsa-miR-", "miR-")[:14],
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
        x_cursor += v
    final = base_value + sum(v for _, v in pos) + sum(v for _, v in neg)
    ax.axvline(base_value, color=SLATE, linewidth=2)
    ax.text(base_value, 0.45, f"base = {base_value:.3f}",
            ha="center", fontsize=9, color=SLATE)
    ax.axvline(final, color=PRIMARY, linewidth=2.5)
    ax.text(final, -0.5, f"f(x) = {final:.3f}",
            ha="center", fontsize=10, color=PRIMARY, fontweight="bold")
    ax.set_ylim(-0.75, 0.75)
    ax.set_xlim(min(final, base_value) - 0.05, max(final, base_value) + 0.05)
    ax.set_yticks([])
    ax.set_xlabel("Probability contribution", fontsize=9, color=SLATE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(SLATE_LIGHT)
    ax.tick_params(colors=SLATE)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    PER-FEATURE DENSITY GRID                        ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_density_grid(panel_inputs, bundle, ordered_feats):
    """Small-multiples density curves with patient value marker."""
    quartiles = bundle["train_quartiles"]
    n = len(ordered_feats)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2.4))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()
    for i, f in enumerate(ordered_feats):
        ax = axes[i]
        q = quartiles[f]
        pv = float(panel_inputs[f].iloc[0])
        # Normal-approx density on q-stats
        span = q["max"] - q["min"] + q["std"]
        xlo = q["min"] - 0.3 * q["std"]
        xhi = q["max"] + 0.3 * q["std"]
        x = np.linspace(xlo, xhi, 200)
        if q["std"] > 0:
            y = np.exp(-0.5 * ((x - q["mean"]) / q["std"]) ** 2) / (q["std"] * math.sqrt(2 * math.pi))
        else:
            y = np.zeros_like(x)
        ax.fill_between(x, y, color="#CBD5E1", alpha=0.55, linewidth=0)
        ax.plot(x, y, color=SLATE, linewidth=1.2)
        # Patient marker
        if q["std"] > 0:
            ymax_p = np.exp(-0.5 * ((pv - q["mean"]) / q["std"]) ** 2) / (q["std"] * math.sqrt(2 * math.pi))
        else:
            ymax_p = 0
        ax.axvline(pv, color=DANGER_RED, linewidth=2.2, alpha=0.9)
        ax.plot(pv, ymax_p, marker="o", color=DANGER_RED, markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        # Percentile label
        pct = normal_cdf(pv, q["mean"], q["std"]) * 100
        ax.set_title(f.replace("hsa-miR-", "miR-"), fontsize=10,
                      fontweight="bold", color=PRIMARY)
        ax.text(0.98, 0.95, f"{pct:.0f}th\npctl",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=DANGER_RED, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=2))
        ax.set_yticks([])
        ax.set_xticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(SLATE_LIGHT)
        ax.spines["bottom"].set_color(SLATE_LIGHT)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       SIDEBAR RENDER                               ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_sidebar(bundle):
    st.sidebar.markdown(
        f"""
        <div style="text-align:center;margin-bottom:18px">
            {dna_helix_svg(width=180, height=55)}
            <div style="color:{PRIMARY};font-size:13px;font-weight:700;
                        letter-spacing:0.08em;text-transform:uppercase;
                        margin-top:6px">
                Clinical Model Bundle
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if bundle is None:
        st.sidebar.error("`model_bundle.joblib` not found at repo root.")
        return

    st.sidebar.markdown(f"<div class='section-h'>System</div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<small style='color:{SLATE}'>"
        f"<b>Pipeline:</b> {bundle.get('pipeline_id', '—')}<br>"
        f"<b>Trained on:</b> {bundle.get('trained_on', '—')}<br>"
        f"<b>Panel size:</b> {bundle.get('n_features', '—')} miRNAs<br>"
        f"<b>Run ID:</b> <code>{bundle.get('run_id', '—')}</code><br>"
        f"<b>Trained:</b> {(bundle.get('training_date') or '—')[:10]}"
        f"</small>",
        unsafe_allow_html=True,
    )

    im = bundle.get("internal_metrics", {}) or {}
    em = bundle.get("external_metrics") or {}
    st.sidebar.markdown(f"<div class='section-h'>Performance</div>", unsafe_allow_html=True)
    if im.get("stage2_test_auc") is not None:
        ci = im.get("stage2_test_AUC_CI") or ""
        st.sidebar.markdown(
            f"<small><b style='color:{PRIMARY}'>Internal test</b><br>"
            f"AUC <b>{im['stage2_test_auc']}%</b>"
            f"{(' [' + ci + ']') if ci else ''} | "
            f"Gap {im.get('overfit_gap', '—')}%</small>",
            unsafe_allow_html=True,
        )
    if em:
        st.sidebar.markdown(
            f"<small><b style='color:{PRIMARY}'>External (GSE113486 lung)</b><br>"
            f"AUC <b>{em.get('AUC', '—')}%</b> | "
            f"Sens {em.get('Sensitivity', '—')}% | "
            f"Spec {em.get('Specificity', '—')}%</small>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(f"<div class='section-h'>Calibration</div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<small style='color:{SLATE}'>"
        f"<b>τ* (Youden):</b> <code>{bundle['youden_threshold']:.4f}</code><br>"
        f"<b>Platt A:</b> <code>{bundle['platt_A']:+.4f}</code><br>"
        f"<b>Platt B:</b> <code>{bundle['platt_B']:+.4f}</code>"
        + (f"<br><b>J_max:</b> <code>{bundle.get('youden_J_max'):.4f}</code>"
            if bundle.get("youden_J_max") is not None else "")
        + "</small>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='background:{WARN_AMBER_BG};color:{WARN_AMBER};"
        f"padding:10px 12px;border-radius:8px;font-size:11px;line-height:1.45'>"
        f"<b>⚠ RESEARCH USE ONLY</b><br>"
        "Not a clinical decision-making instrument. "
        "Thesis NUST 2026, §6.5.2."
        f"</div>",
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       HERO HEADER                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_hero():
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{PRIMARY_DARK} 0%,{PRIMARY} 50%,#2D5384 100%);
                    padding:28px 32px;border-radius:14px;
                    box-shadow:0 4px 12px rgba(15,23,42,0.15);
                    margin-bottom:18px;
                    border-left:5px solid {DANGER_RED}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div style="color:#93C5FD;font-size:11px;font-weight:600;
                                letter-spacing:0.18em;text-transform:uppercase;
                                margin-bottom:6px">
                        Clinical Decision Support · Research Prototype
                    </div>
                    <h1 style="color:white;margin:0;font-size:30px;font-weight:700;
                                line-height:1.2;font-family:Inter,system-ui,sans-serif">
                        Lung Cancer Detection from Serum miRNA Expression
                    </h1>
                    <p style="color:#CBD5E1;margin:8px 0 0;font-size:14px;line-height:1.4">
                        Explainable two-stage classifier · ClusterCentroids + Random Forest ·
                        10-miRNA panel · Platt-calibrated · Youden-J operating threshold
                    </p>
                </div>
                <div style="display:flex;flex-direction:column;align-items:flex-end;
                            gap:4px;margin-left:24px;flex-shrink:0">
                    {dna_helix_svg(width=180, height=60)}
                    <div style="color:#93C5FD;font-size:10px;letter-spacing:0.08em">
                        NUST · 2026 · §6.5.2
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║                  PATIENT REPORT CARD RENDERER                      ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_patient_report(result, bundle, panel_inputs, patient_id="Sample-001",
                          explainer=None, compact=False):
    """The full patient-report layout. Used by all input tabs."""
    cal = result["cal_prob"]
    tau = result["tau_star"]
    verdict = result["verdict"]
    is_pos = (verdict == "Cancer-positive")
    sv = result["shap_vals"]
    feats = bundle["feature_names"]

    # ── Row 1: gauge + verdict tile ───────────────────────────────
    cols = st.columns([1.05, 1.0])
    with cols[0]:
        st.markdown(
            f"<div class='clinical-card'>"
            f"<h4>Calibrated Risk Probability</h4>",
            unsafe_allow_html=True,
        )
        gauge = render_radial_gauge(cal, tau, is_pos)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[1]:
        verdict_bg = DANGER_RED_BG if is_pos else OK_GREEN_BG
        verdict_fg = DANGER_RED if is_pos else OK_GREEN
        verdict_icon = "⚠" if is_pos else "✓"
        st.markdown(
            f"""
            <div class='clinical-card'>
                <h4>Clinical Verdict</h4>
                <div style="display:flex;flex-direction:column;align-items:center;
                            padding:14px 0">
                    <div style="background:{verdict_bg};color:{verdict_fg};
                                padding:18px 32px;border-radius:12px;
                                font-size:24px;font-weight:700;
                                box-shadow:0 2px 8px rgba(0,0,0,0.08);
                                text-align:center">
                        {verdict_icon} {verdict}
                    </div>
                    <div style="margin-top:14px;display:flex;gap:14px;
                                width:100%;justify-content:center;flex-wrap:wrap">
                        <div style="text-align:center">
                            <div style="color:{SLATE};font-size:10px;
                                        text-transform:uppercase;
                                        letter-spacing:0.06em;font-weight:600">
                                Risk Category
                            </div>
                            <div style="color:{result['cat_color']};font-size:14px;
                                        font-weight:700;margin-top:2px">
                                {result['category']}
                            </div>
                        </div>
                        <div style="text-align:center">
                            <div style="color:{SLATE};font-size:10px;
                                        text-transform:uppercase;
                                        letter-spacing:0.06em;font-weight:600">
                                Decision Margin
                            </div>
                            <div style="color:{PRIMARY};font-size:14px;
                                        font-weight:700;margin-top:2px">
                                {abs(cal - tau):+.3f}
                            </div>
                        </div>
                        <div style="text-align:center">
                            <div style="color:{SLATE};font-size:10px;
                                        text-transform:uppercase;
                                        letter-spacing:0.06em;font-weight:600">
                                Raw RF Prob.
                            </div>
                            <div style="color:{PRIMARY};font-size:14px;
                                        font-weight:700;margin-top:2px">
                                {result['raw_prob']:.3f}
                            </div>
                        </div>
                    </div>
                    <div style="margin-top:14px;color:{TEXT_MUTED};font-size:11px;
                                text-align:center;line-height:1.4">
                        Patient ID: <b>{patient_id}</b> ·
                        cal. prob {cal:.3f} {'≥' if is_pos else '<'} τ* {tau:.3f}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if compact:
        return

    # ── Row 2: SHAP waterfall ──────────────────────────────────────
    st.markdown(
        "<div class='clinical-card'>"
        "<h4>SHAP Waterfall — Cumulative Feature Attribution</h4>",
        unsafe_allow_html=True,
    )
    base_val = 0.5  # neutral baseline for waterfall display
    fvals = [float(panel_inputs[f].iloc[0]) for f in feats]
    wf = render_shap_waterfall(sv, feats, fvals, base_val)
    st.pyplot(wf, use_container_width=True)
    st.markdown(
        f"<small style='color:{TEXT_MUTED};font-style:italic'>"
        "Read top-to-bottom: each miRNA's contribution accumulates "
        f"from a baseline of E[f(x)] = 0.500 to the final raw probability. "
        f"<span style='color:{DANGER_RED};font-weight:600'>Red bars</span> "
        f"push toward cancer; "
        f"<span style='color:{PRIMARY_2};font-weight:600'>blue bars</span> "
        f"push toward non-cancer."
        "</small>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: SHAP force plot ─────────────────────────────────────
    st.markdown(
        "<div class='clinical-card'>"
        "<h4>SHAP Force Diagram — Net Positive vs Negative Contributions</h4>",
        unsafe_allow_html=True,
    )
    force = render_shap_force(sv, feats, base_val)
    st.pyplot(force, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 4: density grid ────────────────────────────────────────
    st.markdown(
        "<div class='clinical-card'>"
        "<h4>Per-miRNA Position in Training Distribution</h4>",
        unsafe_allow_html=True,
    )
    quartiles = bundle.get("train_quartiles", {})
    shap_imp = bundle.get("shap_importance") or {}
    ordered = sorted(feats, key=lambda f: -shap_imp.get(f, 0.0))
    dgrid = render_density_grid(panel_inputs, bundle, ordered)
    st.pyplot(dgrid, use_container_width=True)
    st.markdown(
        f"<small style='color:{TEXT_MUTED};font-style:italic'>"
        "Each panel shows the training-distribution density for one miRNA. "
        f"The <span style='color:{DANGER_RED};font-weight:600'>red marker</span> "
        "indicates this patient's measured value and the corresponding percentile "
        "rank against training samples."
        "</small>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 5: full reliability + SHAP table ───────────────────────
    rel_rows = []
    for f in ordered:
        v = float(panel_inputs[f].iloc[0])
        q = quartiles.get(f, {})
        pct = normal_cdf(v, q.get("mean", 0), q.get("std", 1)) * 100
        band = ("Within IQR" if q.get("q25", -np.inf) <= v <= q.get("q75", np.inf)
                else "Outside IQR")
        rel_rows.append({
            "miRNA"            : f,
            "Patient value"    : f"{v:.3f}",
            "Training median"  : f"{q.get('q50', float('nan')):.3f}" if q else "—",
            "IQR"              : (f"[{q['q25']:.3f}, {q['q75']:.3f}]"
                                    if q else "—"),
            "Percentile"       : f"{pct:.0f}th",
            "Band"             : band,
            "SHAP contribution": f"{sv[feats.index(f)]:+.3f}",
            "|SHAP| rank"      : f"#{ordered.index(f) + 1}",
        })
    st.markdown(
        "<div class='clinical-card'>"
        "<h4>Diagnostic Detail Table</h4>",
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(rel_rows), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       SAMPLE PRESETS                               ║
# ╚══════════════════════════════════════════════════════════════════╝
def get_presets(bundle):
    """Build 4 patient presets from training quartiles."""
    feats = bundle["feature_names"]
    q = bundle.get("train_quartiles", {})
    shap_imp = bundle.get("shap_importance") or {}
    ordered = sorted(feats, key=lambda f: -shap_imp.get(f, 0.0))

    # Healthy reference: training medians (most non-cancer-like)
    healthy = {f: q[f]["q50"] if f in q else 0.0 for f in feats}

    # High-risk: push top-5 SHAP miRNAs to upper quartile
    high_risk = {f: q[f]["q50"] if f in q else 0.0 for f in feats}
    for f in ordered[:5]:
        if f in q:
            high_risk[f] = q[f]["q75"]
    # Push bottom-5 toward lower quartile
    for f in ordered[5:]:
        if f in q:
            high_risk[f] = q[f]["q25"]

    # Cancer-like (stronger signal): max-leaning
    cancer = {f: q[f]["q75"] if f in q else 0.0 for f in feats}
    for f in ordered[5:]:
        if f in q:
            cancer[f] = q[f]["q25"]

    # Borderline: blend median and Q75 on top SHAP markers
    borderline = {f: q[f]["q50"] if f in q else 0.0 for f in feats}
    for f in ordered[:3]:
        if f in q:
            borderline[f] = (q[f]["q50"] + q[f]["q75"]) / 2

    return {
        "🟢 Healthy control reference": healthy,
        "🟡 Borderline case": borderline,
        "🟠 Elevated risk profile": high_risk,
        "🔴 Strong cancer signal": cancer,
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       INPUT TABS                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_csv_tab(bundle, explainer, shap_imp):
    feats = bundle["feature_names"]
    medians = bundle["train_medians"]
    quartiles = bundle.get("train_quartiles", {})
    ordered = sorted(feats, key=lambda f: -shap_imp.get(f, 0.0))

    # Panel composition banner
    st.markdown(
        f"<div class='clinical-card'>"
        f"<h4>Required Input — 10 miRNA Expression Values</h4>"
        f"<small style='color:{SLATE}'>"
        "Upload a CSV with one row per patient and the following 10 column "
        "headers. Extra columns (Patient_ID, Label, etc.) are silently dropped. "
        "Column order does not matter. Missing miRNAs are imputed with training "
        "medians and flagged as unreliable."
        "</small></div>",
        unsafe_allow_html=True,
    )

    panel_rows = []
    for rank, f in enumerate(ordered, 1):
        q = quartiles.get(f, {})
        panel_rows.append({
            "Rank": rank,
            "miRNA column header": f,
            "Mean |SHAP|": f"{shap_imp.get(f, 0.0):.4f}",
            "Training median": f"{q.get('q50', float('nan')):.3f}" if q else "—",
            "Range [min, max]": (f"[{q['min']:.3f}, {q['max']:.3f}]"
                                    if q else "—"),
        })
    st.dataframe(pd.DataFrame(panel_rows), hide_index=True, use_container_width=True)

    template_df = pd.DataFrame(
        [{f: float(medians.get(f, 0.0)) for f in feats}]
    )
    template_df.insert(0, "Patient_ID", "SAMPLE_001")
    buf = io.StringIO()
    template_df.to_csv(buf, index=False)

    ca, cb = st.columns([1, 1])
    with ca:
        st.download_button(
            "📄 Download CSV template (current model's exact schema)",
            buf.getvalue(),
            file_name="lung_mirna_panel_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cb:
        st.markdown(
            f"<small style='color:{SLATE}'>"
            "Template ships with training-median pre-filled values. "
            "Replace each row with your patient's measured expression values."
            "</small>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='clinical-divider'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload patient CSV (one row per patient)",
        type=["csv"], key="csv_upload",
    )
    if uploaded is None:
        return

    try:
        input_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return
    st.success(f"Loaded {len(input_df)} samples, {input_df.shape[1]} columns")
    aligned, flags = coerce_to_panel(input_df, feats, medians)
    missing = [f for f, was, _ in flags if was]
    if missing:
        st.warning(
            f"⚠ **{len(missing)} of {len(feats)} required miRNAs missing** from upload, "
            "imputed with training medians (predictions may be unreliable):\n\n" +
            "\n".join(f"- `{f}`" for f in missing)
        )

    t0 = time.time()
    raws = bundle["stage2_model"].predict_proba(aligned)[:, 1]
    cals = bundle["platt_scaler"].predict_proba(raws.reshape(-1, 1))[:, 1]
    preds = (cals >= bundle["youden_threshold"]).astype(int)
    elapsed_ms = (time.time() - t0) * 1000

    out = aligned.copy()
    if "Patient_ID" in input_df.columns:
        out.insert(0, "Patient_ID", input_df["Patient_ID"].values)
    out["raw_prob"]       = raws
    out["cal_prob"]       = cals
    out["verdict"]        = ["Cancer-positive" if p else "Non-cancer" for p in preds]

    st.markdown(
        f"<div class='section-h'>Bulk Predictions ({elapsed_ms:.0f} ms)</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(out, hide_index=False, use_container_width=True)

    dl = io.StringIO()
    out.to_csv(dl, index=False)
    st.download_button("📥 Download predictions as CSV", dl.getvalue(),
                        file_name="predictions.csv", mime="text/csv")

    # Detailed report for each sample (expandable)
    st.markdown(
        f"<div class='section-h'>Detailed Patient Reports</div>",
        unsafe_allow_html=True,
    )
    for i in range(min(len(out), 25)):
        pid = (str(out["Patient_ID"].iloc[i])
               if "Patient_ID" in out.columns
               else f"Sample-{i+1:03d}")
        verdict_icon = "⚠" if preds[i] == 1 else "✓"
        verdict_text = "Cancer-positive" if preds[i] == 1 else "Non-cancer"
        verdict_color = DANGER_RED if preds[i] == 1 else OK_GREEN
        with st.expander(
            f"{verdict_icon} {pid} — {verdict_text} (cal. prob {cals[i]:.3f})"
        ):
            row = aligned.iloc[[i]].reset_index(drop=True)
            result = predict_one(row, bundle, explainer)
            render_patient_report(result, bundle, row, patient_id=pid,
                                    explainer=explainer)
    if len(out) > 25:
        st.info(f"Detailed reports shown for first 25 samples. "
                f"Full predictions for all {len(out)} are in the table above.")


def render_manual_tab(bundle, explainer, shap_imp):
    feats = bundle["feature_names"]
    medians = bundle["train_medians"]
    ordered = sorted(feats, key=lambda f: -shap_imp.get(f, 0.0))

    st.markdown(
        f"<div class='clinical-card'>"
        f"<h4>Manual Patient Entry</h4>"
        f"<small style='color:{SLATE}'>"
        "Enter expression values for the 10 SHAP-shortlisted miRNAs. "
        "Defaults are training medians. Fields are ranked by mean |SHAP| (most "
        "influential miRNAs first)."
        "</small></div>",
        unsafe_allow_html=True,
    )

    patient_id = st.text_input("Patient ID (optional)", value="PATIENT-001",
                                 max_chars=40)

    cols = st.columns(2)
    manual_vals = {}
    for i, f in enumerate(ordered):
        with cols[i % 2]:
            manual_vals[f] = st.number_input(
                f"#{i+1} — {f}",
                value=float(medians.get(f, 0.0)),
                format="%.4f",
                key=f"manual_v2_{i}",
                help=f"Mean |SHAP|: {shap_imp.get(f, 0.0):.4f}",
            )

    if st.button("🩺 Run Prediction", type="primary", key="manual_predict"):
        row = pd.DataFrame([manual_vals], columns=feats)
        result = predict_one(row, bundle, explainer)
        st.markdown("<div class='clinical-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-h'>Patient Report — {patient_id}</div>",
            unsafe_allow_html=True,
        )
        render_patient_report(result, bundle, row, patient_id=patient_id,
                                explainer=explainer)


def render_demo_tab(bundle, explainer, shap_imp):
    feats = bundle["feature_names"]
    medians = bundle["train_medians"]
    quartiles = bundle.get("train_quartiles", {})
    ordered = sorted(feats, key=lambda f: -shap_imp.get(f, 0.0))
    presets = get_presets(bundle)

    st.markdown(
        f"<div class='clinical-card'>"
        f"<h4>Demo Mode — Interactive Sliders + Patient Presets</h4>"
        f"<small style='color:{SLATE}'>"
        "Click a preset to load a representative patient profile, then "
        "adjust sliders to see how each miRNA value affects the prediction "
        "in real time. Slider ranges follow each feature's training "
        "distribution (min − 0.5σ to max + 0.5σ)."
        "</small></div>",
        unsafe_allow_html=True,
    )

    # Initialise session state for slider values
    if "demo_vals" not in st.session_state:
        st.session_state.demo_vals = {f: float(medians.get(f, 0.0)) for f in feats}

    # Preset buttons row
    st.markdown(f"<div class='section-h'>Patient Presets</div>",
                  unsafe_allow_html=True)
    pcols = st.columns(len(presets))
    for (label, vals), col in zip(presets.items(), pcols):
        with col:
            if st.button(label, key=f"preset_{label}", use_container_width=True):
                st.session_state.demo_vals = dict(vals)
                st.rerun()

    st.markdown(f"<div class='section-h'>miRNA Expression Sliders</div>",
                  unsafe_allow_html=True)
    cols = st.columns(2)
    demo_vals = {}
    for i, f in enumerate(ordered):
        q = quartiles.get(f, {"min": -2.0, "max": 2.0, "q50": 0.0, "std": 1.0})
        rng_lo = float(q["min"] - 0.5 * q["std"])
        rng_hi = float(q["max"] + 0.5 * q["std"])
        with cols[i % 2]:
            demo_vals[f] = st.slider(
                f"#{i+1} — {f}",
                min_value=rng_lo, max_value=rng_hi,
                value=float(st.session_state.demo_vals.get(f, q["q50"])),
                step=(rng_hi - rng_lo) / 200,
                format="%.3f", key=f"demo_v2_{i}",
                help=f"Mean |SHAP|: {shap_imp.get(f, 0.0):.4f}",
            )
    st.session_state.demo_vals = demo_vals

    # Round to 5 dp so tiny float drift from slider steps never causes a
    # cache miss for values the user has already visited.
    row_key = tuple(round(float(demo_vals[f]), 5) for f in feats)
    result  = _predict_cached(bundle, explainer, row_key)
    row     = pd.DataFrame([demo_vals], columns=feats)
    st.markdown("<div class='clinical-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-h'>Live Patient Report</div>",
        unsafe_allow_html=True,
    )
    render_patient_report(result, bundle, row, patient_id="DEMO-PATIENT",
                            explainer=explainer)


def render_comparison_tab(bundle, explainer, shap_imp):
    feats = bundle["feature_names"]
    medians = bundle["train_medians"]
    ordered = sorted(feats, key=lambda f: -shap_imp.get(f, 0.0))
    presets = get_presets(bundle)
    preset_keys = list(presets.keys())

    st.markdown(
        f"<div class='clinical-card'>"
        f"<h4>Side-by-Side Patient Comparison</h4>"
        f"<small style='color:{SLATE}'>"
        "Load two patient profiles and compare predictions, SHAP contributions, "
        "and decision margins side-by-side. Useful for diagnostic discussion "
        "and visualising the role of specific miRNAs in the decision."
        "</small></div>",
        unsafe_allow_html=True,
    )

    if "cmp_A" not in st.session_state:
        st.session_state.cmp_A = dict(presets[preset_keys[0]])
    if "cmp_B" not in st.session_state:
        st.session_state.cmp_B = dict(presets[preset_keys[2]])

    pcols = st.columns(2)
    for col, label, key in [
        (pcols[0], "Patient A", "cmp_A"),
        (pcols[1], "Patient B", "cmp_B"),
    ]:
        with col:
            st.markdown(
                f"<div style='background:{PRIMARY};color:white;padding:8px 12px;"
                f"border-radius:6px;font-weight:600;text-align:center'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )
            sel = st.selectbox(
                f"Preset for {label}",
                options=preset_keys,
                index=preset_keys.index(
                    next((k for k in preset_keys
                          if presets[k] == st.session_state[key]),
                         preset_keys[0])
                ) if any(presets[k] == st.session_state[key] for k in preset_keys) else 0,
                key=f"sel_{key}",
            )
            if st.button(f"Load preset → {label}", key=f"load_{key}",
                          use_container_width=True):
                st.session_state[key] = dict(presets[sel])
                st.rerun()

    # Run predictions — keyed by rounded float tuples so preset re-loads
    # return from cache instantly without re-traversing the SHAP tree.
    rowA  = pd.DataFrame([st.session_state.cmp_A], columns=feats)
    rowB  = pd.DataFrame([st.session_state.cmp_B], columns=feats)
    key_A = tuple(round(float(st.session_state.cmp_A[f]), 5) for f in feats)
    key_B = tuple(round(float(st.session_state.cmp_B[f]), 5) for f in feats)
    rA    = _predict_cached(bundle, explainer, key_A)
    rB    = _predict_cached(bundle, explainer, key_B)

    st.markdown("<div class='clinical-divider'></div>", unsafe_allow_html=True)

    # Side-by-side compact reports
    rc = st.columns(2)
    with rc[0]:
        st.markdown(
            f"<div class='section-h'>Patient A Report</div>",
            unsafe_allow_html=True,
        )
        render_patient_report(rA, bundle, rowA, patient_id="PATIENT-A",
                                explainer=explainer, compact=True)
    with rc[1]:
        st.markdown(
            f"<div class='section-h'>Patient B Report</div>",
            unsafe_allow_html=True,
        )
        render_patient_report(rB, bundle, rowB, patient_id="PATIENT-B",
                                explainer=explainer, compact=True)

    # Comparison table
    st.markdown(
        f"<div class='section-h'>Direct Comparison</div>",
        unsafe_allow_html=True,
    )
    cmp_rows = []
    for f in ordered:
        vA = float(rowA[f].iloc[0])
        vB = float(rowB[f].iloc[0])
        svA = rA["shap_vals"][feats.index(f)]
        svB = rB["shap_vals"][feats.index(f)]
        delta = vB - vA
        cmp_rows.append({
            "miRNA"            : f,
            "Patient A value"  : f"{vA:.3f}",
            "Patient B value"  : f"{vB:.3f}",
            "Δ (B − A)"        : f"{delta:+.3f}",
            "SHAP A"           : f"{svA:+.3f}",
            "SHAP B"           : f"{svB:+.3f}",
            "Δ SHAP"           : f"{svB - svA:+.3f}",
        })
    cmp_df = pd.DataFrame(cmp_rows)
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)

    # Summary statistics
    summary = pd.DataFrame([
        {"Metric": "Calibrated probability",
         "Patient A": f"{rA['cal_prob']:.3f}",
         "Patient B": f"{rB['cal_prob']:.3f}",
         "Δ":         f"{rB['cal_prob'] - rA['cal_prob']:+.3f}"},
        {"Metric": "Verdict at τ*",
         "Patient A": rA["verdict"],
         "Patient B": rB["verdict"],
         "Δ": "—"},
        {"Metric": "Risk category",
         "Patient A": rA["category"],
         "Patient B": rB["category"],
         "Δ": "—"},
        {"Metric": "Decision margin |cal − τ*|",
         "Patient A": f"{abs(rA['cal_prob'] - rA['tau_star']):.3f}",
         "Patient B": f"{abs(rB['cal_prob'] - rB['tau_star']):.3f}",
         "Δ": "—"},
    ])
    st.dataframe(summary, hide_index=True, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                            MAIN                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
def main():
    inject_css()
    bundle = load_bundle()
    render_sidebar(bundle)
    render_hero()

    if bundle is None:
        st.error(
            "Cannot run predictions: `model_bundle.joblib` is not present at the repo root.\n\n"
            "**To produce it:**\n\n"
            "1. Open the modified notebook in Colab.\n"
            "2. Run all cells through Cell 23 (the save-best-model cell).\n"
            "3. Download `outputs/g140_v10_default/model_bundle.joblib`.\n"
            "4. Commit it to the repo root next to `app.py`."
        )
        st.stop()

    feats = bundle["feature_names"]
    medians = bundle["train_medians"]
    bg = pd.DataFrame([medians], columns=feats)
    explainer = get_explainer(bundle["stage2_model"], bg)
    shap_imp = get_panel_ranking(bundle, explainer)

    tabs = st.tabs([
        "📁  CSV Upload",
        "✏️  Manual Entry",
        "🎛️  Demo + Presets",
        "🔬  Side-by-Side Comparison",
    ])
    with tabs[0]:
        render_csv_tab(bundle, explainer, shap_imp)
    with tabs[1]:
        render_manual_tab(bundle, explainer, shap_imp)
    with tabs[2]:
        render_demo_tab(bundle, explainer, shap_imp)
    with tabs[3]:
        render_comparison_tab(bundle, explainer, shap_imp)

    # Footer
    st.markdown(
        f"""
        <div style="margin-top:32px;padding:16px;text-align:center;
                    background:{PRIMARY_DARK};border-radius:10px">
            <div style="color:#94A3B8;font-size:11px;line-height:1.6">
                <b style="color:#CBD5E1">Research Prototype</b> ·
                Not for clinical decision-making ·
                Model trained on GSE137140 ·
                Externally validated on GSE113486 lung subset
            </div>
            <div style="color:#64748B;font-size:10px;margin-top:6px">
                NUST MS Thesis 2026 · §6.5.2 ·
                Explainable Lung-miRNA Detection · CC+RF Deployable Pipeline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
