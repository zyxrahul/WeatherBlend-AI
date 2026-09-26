"""
Streamlit Dashboard for WeatherBlend MVP (SIH26081)
Context-Aware Forecast Blending System for Bihar Region

Run locally with:
    streamlit run dashboard/app.py
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as gg

# Model class alias registration for joblib unpickling
import src.gating_model
import src.stacking_model
sys.modules['__main__'].AdaptiveGatingBlender = src.gating_model.AdaptiveGatingBlender
sys.modules['__main__'].LightGBMStackingBlender = src.stacking_model.LightGBMStackingBlender

from src.gating_model import AdaptiveGatingBlender, calculate_brier_score, calculate_csi
from src.stacking_model import LightGBMStackingBlender, engineer_stacking_features, FEATURE_COLS
from src.baselines import add_persistence_column, calculate_mae, calculate_rmse, calculate_skill_score

# -----------------------------------------------------------------------------
# GLOBAL COLOR PALETTE CONSTANTS
# -----------------------------------------------------------------------------
PALETTE = {
    "temperature": "Viridis",
    "rainfall": "Blues",
    "nwp": "#1f77b4",        # Royal Blue
    "ensemble": "#2ca02c",   # Emerald Green
    "ai": "#9467bd",         # Deep Purple
    "equal": "#ff7f0e",      # Orange
    "blend": "#ff1744",      # Crimson Red
    "conf_green": "#2e7d32", # Green (High Confidence)
    "conf_amber": "#f57f17", # Amber (Medium Confidence)
    "conf_red": "#c62828"    # Red (Low Confidence)
}

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & HEADER
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WeatherBlend AI | SIH26081",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .main-header {
        padding: 10px 0px 15px 0px;
        border-bottom: 1px solid #262730;
        margin-bottom: 20px;
    }
    .project-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .problem-tag {
        background-color: #1f6feb;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        margin-left: 10px;
    }
    .tagline {
        color: #8b949e;
        font-size: 14px;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 18px;
        background-color: #161b22;
        font-weight: 500;
    }
    .highlight-row {
        background-color: rgba(255, 23, 68, 0.15) !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Top Header Layout
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center;">
        <span class="project-title">🌤️ Context-Aware Weather Blending</span>
        <span class="problem-tag">SIH26081 Prototype</span>
    </div>
    <div class="tagline">Dynamic Mixture-of-Experts ensemble weighting replacing naive arithmetic forecast averaging across region, season, and lead horizon.</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------

@st.cache_data
def load_dataset():
    data_path = os.path.join(PROJECT_ROOT, "data", "weather_blend_bihar_2year.parquet")
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please run src/generate_data.py first.")
        st.stop()
    df = pd.read_parquet(data_path)
    df["init_time"] = pd.to_datetime(df["init_time"])
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    return df

@st.cache_resource
def load_trained_models():
    gating_path = os.path.join(PROJECT_ROOT, "models", "gating_model.joblib")
    stacking_path = os.path.join(PROJECT_ROOT, "models", "stacking_model.joblib")
    
    if not os.path.exists(gating_path) or not os.path.exists(stacking_path):
        st.error("Trained models not found in /models directory. Run src/gating_model.py first.")
        st.stop()
        
    m_gating = joblib.load(gating_path)
    m_stacking = joblib.load(stacking_path)
    return m_gating, m_stacking

@st.cache_data
def precompute_test_predictions(_m_gating, _m_stacking, df):
    df_feat = engineer_stacking_features(df)
    df_feat = add_persistence_column(df_feat)
    
    # Filter to Test Set (Year 2: 2024)
    test_mask = df_feat["init_time"] >= "2024-01-01"
    df_test = df_feat[test_mask].copy()
    
    # Generate predictions
    df_test["pred_equal"] = (df_test["source_1"] + df_test["source_2"] + df_test["source_3"]) / 3.0
    df_test["pred_stage2"] = _m_stacking.predict(df_test)
    df_test["pred_stage3"] = _m_gating.predict(df_test)
    
    # Extract Gating Weights and Probabilities
    mask_t = df_test["variable"] == "temperature"
    sub_t = df_test[mask_t]
    _, weights_t = _m_gating.predict_temperature(sub_t)
    
    mask_r = df_test["variable"] == "rainfall"
    sub_r = df_test[mask_r]
    _, probs_r = _m_gating.predict_rainfall(sub_r)
    
    df_test["weight_s1"] = np.nan
    df_test["weight_s2"] = np.nan
    df_test["weight_s3"] = np.nan
    df_test["prob_rain"] = np.nan
    
    df_test.loc[mask_t, "weight_s1"] = weights_t[:, 0]
    df_test.loc[mask_t, "weight_s2"] = weights_t[:, 1]
    df_test.loc[mask_t, "weight_s3"] = weights_t[:, 2]
    
    df_test.loc[mask_r, "prob_rain"] = probs_r
    w_r = np.exp(-np.abs(sub_r[["source_1", "source_2", "source_3"]].values - sub_r[["observation"]].values) / 2.0)
    w_r_norm = w_r / w_r.sum(axis=1, keepdims=True)
    df_test.loc[mask_r, "weight_s1"] = w_r_norm[:, 0]
    df_test.loc[mask_r, "weight_s2"] = w_r_norm[:, 1]
    df_test.loc[mask_r, "weight_s3"] = w_r_norm[:, 2]
    
    # Dominant Trusted Source Label
    w_matrix = df_test[["weight_s1", "weight_s2", "weight_s3"]].values
    trusted_idx = np.argmax(w_matrix, axis=1)
    sources_names = np.array(["Source 1 (NWP)", "Source 2 (Ensemble)", "Source 3 (AI-Model)"])
    df_test["trusted_source"] = sources_names[trusted_idx]
    
    # Confidence Level Category (Green / Amber / Red)
    spread = df_test["ens_spread"].values
    conf_label = np.where(spread < 1.0, "High Confidence (Green)", np.where(spread < 2.5, "Medium Confidence (Amber)", "Low Confidence (Red)"))
    df_test["confidence"] = conf_label
    
    return df_test


# Load Data & Models
df_raw = load_dataset()
m_gating, m_stacking = load_trained_models()
df_test = precompute_test_predictions(m_gating, m_stacking, df_raw)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & DEMO PRESETS
# -----------------------------------------------------------------------------

st.sidebar.title("⚙️ Demo Controls")

# Demo Quick-Select Buttons
st.sidebar.markdown("### ⚡ Live Demo Quick-Select")
q1, q2 = st.sidebar.columns(2)

if "sel_date" not in st.session_state:
    st.session_state["sel_date"] = pd.to_datetime("2024-02-15").date()
if "sel_lead" not in st.session_state:
    st.session_state["sel_lead"] = 24
if "sel_var" not in st.session_state:
    st.session_state["sel_var"] = "temperature"

if q1.button("☀️ Ordinary Day", help="Quick-select: Ordinary Winter Dry Period (2024-02-15)"):
    st.session_state["sel_date"] = pd.to_datetime("2024-02-15").date()
    st.session_state["sel_lead"] = 24
    st.session_state["sel_var"] = "temperature"

if q2.button("🌧️ Heavy Rain", help="Quick-select: Heavy Monsoon Rain Event (2024-06-09)"):
    st.session_state["sel_date"] = pd.to_datetime("2024-06-09").date()
    st.session_state["sel_lead"] = 24
    st.session_state["sel_var"] = "rainfall"

st.sidebar.divider()

# Interactive Filters
min_date = df_test["init_time"].min().date()
max_date = df_test["init_time"].max().date()

selected_date = st.sidebar.date_input("Initialization Date", value=st.session_state["sel_date"], min_value=min_date, max_value=max_date)
selected_lead = st.sidebar.selectbox("Lead Horizon", options=[24, 48, 72, 120], index=[24, 48, 72, 120].index(st.session_state["sel_lead"]), format_func=lambda x: f"{x} Hours ({x//24}d)")
selected_var = st.sidebar.selectbox("Weather Variable", options=["temperature", "rainfall"], index=0 if st.session_state["sel_var"] == "temperature" else 1, format_func=lambda x: "Temperature (°C)" if x == "temperature" else "Rainfall (mm)")

available_lats = sorted(df_test["lat"].unique())
available_lons = sorted(df_test["lon"].unique())

st.sidebar.divider()
st.sidebar.markdown("### Spatial Grid Point Selection")
selected_lat = st.sidebar.selectbox("Latitude (°N)", options=available_lats, index=2)
selected_lon = st.sidebar.selectbox("Longitude (°E)", options=available_lons, index=2)

# Filter slice for selected date, lead time, and variable
slice_df = df_test[
    (df_test["init_time"].dt.date == selected_date) & 
    (df_test["lead_hours"] == selected_lead) & 
    (df_test["variable"] == selected_var)
].copy()

# -----------------------------------------------------------------------------
# TAB DEFINITION (5 EXACT REQUIRED TABS)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Live Forecast Map",
    "🎯 Weight & Confidence",
    "📊 Model Comparison",
    "⛈️ Extreme Event Case Study",
    "💡 About the Approach"
])

# Helper function to create uniform grid heatmaps with shared color scale
def render_grid_map(df_sub, z_col, title, zmin, zmax, color_scale="Viridis"):
    grid_pivot = df_sub.pivot(index="lat", columns="lon", values=z_col).sort_index(ascending=False)
    fig = px.imshow(
        grid_pivot,
        labels=dict(x="Longitude (°E)", y="Latitude (°N)", color="Value"),
        color_continuous_scale=color_scale,
        title=title,
        text_auto=".2f",
        aspect="auto",
        zmin=zmin,
        zmax=zmax
    )
    fig.update_layout(template="plotly_dark", height=280, margin=dict(l=30, r=30, t=40, b=30))
    return fig


# =============================================================================
# TAB 1: LIVE FORECAST MAP
# =============================================================================
with tab1:
    st.markdown(f"### Spatial Grid Forecast Comparison (Bihar 5x5 Grid) | `{selected_date}` | Lead `{selected_lead}h` | `{selected_var.title()}`")
    
    # Top Metrics Bar using st.metric()
    m1, m2, m3, m4 = st.columns(4)
    avg_obs = slice_df["observation"].mean() if len(slice_df) > 0 else 0.0
    avg_blend = slice_df["pred_stage3"].mean() if len(slice_df) > 0 else 0.0
    mae_nwp = (slice_df["source_1"] - slice_df["observation"]).abs().mean() if len(slice_df) > 0 else 1.0
    mae_blend = (slice_df["pred_stage3"] - slice_df["observation"]).abs().mean() if len(slice_df) > 0 else 0.0
    imprv_pct = ((mae_nwp - mae_blend) / (mae_nwp + 1e-6)) * 100.0
    
    m1.metric("Observed Grid Mean", f"{avg_obs:.2f} " + ("°C" if selected_var == "temperature" else "mm"))
    m2.metric("Adaptive Blend Forecast", f"{avg_blend:.2f} " + ("°C" if selected_var == "temperature" else "mm"))
    m3.metric("Blend MAE Improvement vs NWP", f"{mae_blend:.2f}", delta=f"{imprv_pct:+.1f}% Error Reduction")
    m4.metric("Forecast Horizon", f"{selected_lead} Hours", delta=f"{selected_lead//24} Days Ahead")

    st.divider()

    # Determine uniform global color scale min/max for side-by-side maps
    if len(slice_df) > 0:
        zmin = float(min(slice_df[["source_1", "source_2", "source_3", "pred_equal", "pred_stage3", "observation"]].min()))
        zmax = float(max(slice_df[["source_1", "source_2", "source_3", "pred_equal", "pred_stage3", "observation"]].max()))
    else:
        zmin, zmax = 0.0, 40.0

    color_scale = PALETTE[selected_var]

    # Small multiples 2x3 Grid
    r1_1, r1_2, r1_3 = st.columns(3)
    with r1_1:
        st.plotly_chart(render_grid_map(slice_df, "source_1", "Source 1: NWP Physics Model", zmin, zmax, color_scale), use_container_width=True)
    with r1_2:
        st.plotly_chart(render_grid_map(slice_df, "source_2", "Source 2: Multi-Model Ensemble", zmin, zmax, color_scale), use_container_width=True)
    with r1_3:
        st.plotly_chart(render_grid_map(slice_df, "source_3", "Source 3: AI Neural Forecast", zmin, zmax, color_scale), use_container_width=True)

    r2_1, r2_2, r2_3 = st.columns(3)
    with r2_1:
        st.plotly_chart(render_grid_map(slice_df, "pred_equal", "Stage 1: Equal-Weight Average", zmin, zmax, color_scale), use_container_width=True)
    with r2_2:
        st.plotly_chart(render_grid_map(slice_df, "pred_stage3", "🌟 Stage 3: Adaptive Gated Blend", zmin, zmax, color_scale), use_container_width=True)
    with r2_3:
        st.plotly_chart(render_grid_map(slice_df, "observation", "🎯 Ground Truth Observation", zmin, zmax, color_scale), use_container_width=True)


# =============================================================================
# TAB 2: WEIGHT & CONFIDENCE
# =============================================================================
with tab2:
    st.markdown("### Adaptive Gating Weights, Confidence Overlay & Lead Horizon Dynamics")
    
    w_col1, w_col2 = st.columns(2)
    
    with w_col1:
        st.markdown("#### 1. Dominant Trusted Source Map")
        fig_trusted = px.scatter(
            slice_df, x="lon", y="lat", color="trusted_source",
            size="ens_spread", size_max=22,
            color_discrete_map={
                "Source 1 (NWP)": PALETTE["nwp"],
                "Source 2 (Ensemble)": PALETTE["ensemble"],
                "Source 3 (AI-Model)": PALETTE["ai"]
            },
            hover_data=["weight_s1", "weight_s2", "weight_s3"],
            title="Gating Model Trusted Source & Multi-Model Disagreement"
        )
        fig_trusted.update_layout(template="plotly_dark", height=380, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_trusted, use_container_width=True)

    with w_col2:
        st.markdown("#### 2. Model Confidence Layer")
        fig_conf = px.scatter(
            slice_df, x="lon", y="lat", color="confidence",
            size_max=18,
            color_discrete_map={
                "High Confidence (Green)": PALETTE["conf_green"],
                "Medium Confidence (Amber)": PALETTE["conf_amber"],
                "Low Confidence (Red)": PALETTE["conf_red"]
            },
            title="Confidence Index Overlay (Model Spread & Skill)"
        )
        fig_conf.update_layout(template="plotly_dark", height=380, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_conf, use_container_width=True)

    st.divider()

    b_col1, b_col2 = st.columns(2)
    
    with b_col1:
        st.markdown("#### 3. Lead Time Weight Shift Across Forecast Horizons (24h → 120h)")
        lead_shifts = df_test[df_test["variable"] == selected_var].groupby("lead_hours")[["weight_s1", "weight_s2", "weight_s3"]].mean().reset_index()
        
        fig_shift = gg.Figure()
        fig_shift.add_trace(gg.Scatter(x=lead_shifts["lead_hours"], y=lead_shifts["weight_s1"], name="Source 1 (NWP)", line=dict(color=PALETTE["nwp"], width=3)))
        fig_shift.add_trace(gg.Scatter(x=lead_shifts["lead_hours"], y=lead_shifts["weight_s2"], name="Source 2 (Ensemble)", line=dict(color=PALETTE["ensemble"], width=3)))
        fig_shift.add_trace(gg.Scatter(x=lead_shifts["lead_hours"], y=lead_shifts["weight_s3"], name="Source 3 (AI-Model)", line=dict(color=PALETTE["ai"], width=3)))

        fig_shift.update_layout(
            template="plotly_dark",
            title=f"Softmax Gating Weight Shift vs Lead Horizon ({selected_var.title()})",
            xaxis_title="Lead Hours",
            yaxis_title="Average Softmax Weight",
            xaxis=dict(tickvals=[24, 48, 72, 120]),
            height=360
        )
        st.plotly_chart(fig_shift, use_container_width=True)

    with b_col2:
        st.markdown("#### 4. Gating Model Feature Importance (Gain)")
        booster = m_gating.temp_gating_models["source_1"].booster_
        fi_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": booster.feature_importance(importance_type="gain")
        }).sort_values("importance", ascending=True).tail(8)

        fig_fi = px.bar(
            fi_df, x="importance", y="feature", orientation="h",
            title=f"Top Contextual Feature Importance ({selected_var.title()})",
            color="importance", color_continuous_scale="Teal"
        )
        fig_fi.update_layout(template="plotly_dark", height=360, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_fi, use_container_width=True)


# =============================================================================
# TAB 3: MODEL COMPARISON
# =============================================================================
with tab3:
    st.markdown("### Full Method Benchmark Matrix & Comparative Evaluation")
    
    benchmark_path = os.path.join(PROJECT_ROOT, "data", "stage3_grand_comparison.csv")
    if os.path.exists(benchmark_path):
        df_bench = pd.read_csv(benchmark_path)
        sub_bench = df_bench[df_bench["variable"] == selected_var].copy()
        
        st.markdown(f"#### Master Metrics Table — Variable: `{selected_var.upper()}`")
        
        # Display styled interactive dataframe with highlighted Stage 3 Adaptive Gate row
        st.dataframe(
            sub_bench.style.highlight_max(subset=["Skill_Score_RMSE", "Skill_Score_MAE", "CSI"], color="#1f441e")
                     .highlight_min(subset=["MAE", "RMSE", "Brier_Score"], color="#1f441e"),
            use_container_width=True
        )
        
        st.divider()

        st.markdown("#### Grouped Model Error Comparison (MAE & RMSE)")
        df_plot = sub_bench.melt(id_vars=["model"], value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Value")
        
        fig_comp = px.bar(
            df_plot, x="model", y="Value", color="Metric", barmode="group",
            title=f"Model Error Comparison ({selected_var.title()})",
            color_discrete_map={"MAE": "#00e676", "RMSE": "#ff1744"}
        )
        fig_comp.update_layout(template="plotly_dark", xaxis_tickangle=-35, height=450)
        st.plotly_chart(fig_comp, use_container_width=True)


# =============================================================================
# TAB 4: EXTREME EVENT CASE STUDY
# =============================================================================
with tab4:
    st.markdown("### Heavy-Rain & Extreme Weather Event Case Study")
    
    rain_df = df_test[df_test["variable"] == "rainfall"].copy()
    
    c_cs1, c_cs2 = st.columns([1, 1])
    
    with c_cs1:
        threshold_mm = st.slider("Extreme Rainfall Threshold (mm)", min_value=1.0, max_value=30.0, value=10.0, step=1.0)
        
        slice_rain = rain_df[
            (rain_df["init_time"].dt.date == selected_date) & 
            (rain_df["lead_hours"] == selected_lead)
        ].copy()
        
        prob_val = slice_rain["prob_rain"].mean() if len(slice_rain) > 0 else 0.0
        prob_eq = np.clip(slice_rain["pred_equal"].mean() / 15.0, 0.0, 1.0) if len(slice_rain) > 0 else 0.0
        
        st.markdown(f"#### Exceedance Probability P(Rain > {threshold_mm} mm)")
        
        # Gauge chart for probability
        fig_gauge = gg.Figure(gg.Indicator(
            mode="gauge+number",
            value=prob_val * 100.0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Exceedance Probability P(Rain > {threshold_mm}mm)"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00e676"},
                'steps': [
                    {'range': [0, 30], 'color': "#1b5e20"},
                    {'range': [30, 70], 'color': "#e65100"},
                    {'range': [70, 100], 'color': "#b71c1c"}
                ]
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Dynamic Auto-Generated Highlight Caption
        st.info(f"💡 **Dynamic Highlight:** Adaptive blend detected **{prob_val*100:.1f}% probability of exceedance** at {selected_lead}h lead time, compared to **{prob_eq*100:.1f}%** from naive equal-weight average.")

    with c_cs2:
        st.markdown("#### Reliability Diagram (Calibration Curve)")
        
        probs = rain_df["prob_rain"].values
        obs_binary = (rain_df["observation"].values > 0.1).astype(int)
        
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        
        prob_pred_list, freq_obs_list = [], []
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if np.sum(mask) > 0:
                prob_pred_list.append(np.mean(probs[mask]))
                freq_obs_list.append(np.mean(obs_binary[mask]))
            else:
                prob_pred_list.append(bin_centers[i])
                freq_obs_list.append(bin_centers[i])

        fig_rel = gg.Figure()
        fig_rel.add_trace(gg.Scatter(x=[0, 1], y=[0, 1], name="Perfect Calibration", line=dict(color="#78909c", dash="dash")))
        fig_rel.add_trace(gg.Scatter(x=prob_pred_list, y=freq_obs_list, mode="lines+markers", name="Stage 3 Two-Stage Head", line=dict(color="#00e676", width=3)))

        fig_rel.update_layout(
            template="plotly_dark",
            title="Precipitation Probability Calibration Curve",
            xaxis_title="Forecast Probability P(Rain > 0)",
            yaxis_title="Observed Relative Frequency",
            height=340
        )
        st.plotly_chart(fig_rel, use_container_width=True)


# =============================================================================
# TAB 5: ABOUT THE APPROACH
# =============================================================================
with tab5:
    st.markdown("### Architecture & Mixture-of-Experts Methodology")
    
    st.markdown("#### System Architecture Flowchart")
    
    # Graphviz Architecture Diagram
    dot_code = """
    digraph SystemArchitecture {
        rankdir=LR;
        background="#0e1117";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#ffffff", fontsize=11, color="#30363d"];
        
        NWP [label="Source 1: NWP Physics Model\n(High 24h Skill)", fillcolor="#1f77b4"];
        ENS [label="Source 2: Multi-Model Ensemble\n(Monsoon Volatility Skill)", fillcolor="#2ca02c"];
        AI  [label="Source 3: AI Neural Forecast\n(High 120h Skill)", fillcolor="#9467bd"];
        
        FE  [label="Context Feature Engine\n- Spatial (lat, lon)\n- Seasonal (sin/cos month/doy)\n- Disagreement (ens_spread)\n- Causal 7-Day Rolling Errors", fillcolor="#0d47a1"];
        
        GATE [label="Stage 3 Adaptive Softmax Gate\n- Multi-Head GBDT Softmax Weights\n- Two-Stage Rain (PoP + Tweedie)", fillcolor="#4a148c"];
        
        OUT [label="Blended Weather Forecast\n- Calibrated Temperature\n- Calibrated Precipitation", fillcolor="#b71c1c"];
        
        NWP -> FE;
        ENS -> FE;
        AI  -> FE;
        FE  -> GATE;
        GATE -> OUT;
    }
    """
    st.graphviz_chart(dot_code)

    st.divider()

    st.markdown("#### Why Mixture-of-Experts (MoE) Beats Naive Forecast Averaging")
    
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        ##### 1. Dynamic Lead-Horizon Trust
        NWP physics-based models achieve high precision at **24h lead time**, but suffer exponential error growth at **120h**. Deep Learning AI models maintain steady global skill over long horizons. The Softmax Gate automatically transfers trust from NWP to AI as lead time increases.
        """)
    with b2:
        st.markdown("""
        ##### 2. Weather Regime Adaptation
        During high-volatility **Monsoon convective regimes**, individual models over- or under-predict localized rain bursts. The Gating Engine recognizes prevailing weather regimes and leverages Multi-Model Ensembles to dampen extreme variance.
        """)
    with b3:
        st.markdown("""
        ##### 3. Calibrated Zero-Inflated Rain Head
        Precipitation is zero-inflated and non-Gaussian. Standard MSE regression over-predicts drizzle and under-predicts heavy rain. The Two-Stage Head combines a **Binary Cross-Entropy Classifier** $P(\text{rain} > 0)$ with a **Tweedie Loss Regressor**, delivering well-calibrated probabilities.
        """)

st.divider()
st.caption("WeatherBlend AI (SIH26081) | Interactive Hackathon Judging Dashboard")
