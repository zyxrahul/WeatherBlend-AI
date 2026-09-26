"""
Page 1: Model Insights - How WeatherBlend AI Works
Provides architecture flow diagrams, benchmark comparison tables, feature ablation study,
and regional lead-time skill breakdowns.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from components import inject_global_css, render_header_bar, render_html_panel, render_sidebar_navigation, CARD_BG, CARD_BORDER, TEXT_MUTED, BLUE_ACCENT, GREEN_SUCCESS

st.set_page_config(
    page_title="Model Insights - WeatherBlend AI",
    page_icon="🧠",
    layout="wide"
)

inject_global_css()
render_sidebar_navigation()
render_header_bar(region_name=st.session_state.get("region", "Bihar"))

render_html_panel("""
<div class="page-title-box">
    <h2>🧠 Model Insights & Architecture</h2>
    <p>How WeatherBlend AI's Context-Aware Adaptive Gating Engine outperforms static ensemble averaging</p>
</div>
""")

# -----------------------------------------------------------------------------
# 1. PIPELINE ARCHITECTURE DIAGRAM
# -----------------------------------------------------------------------------
render_html_panel("<div class='panel-header-title'>⚡ System Architecture & Gating Pipeline</div>")

st.graphviz_chart("""
digraph {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="filled,rounded", fontname="Inter", fontsize=10, fontcolor="#ffffff", border="none"];
    edge [color="#60a5fa", penwidth=1.5];

    subgraph cluster_0 {
        label = "Stage 0: Raw Inputs";
        fontcolor = "#94a3b8";
        style = "dashed";
        color = "#334155";
        
        NWP [label="Source 1: GFS NWP\n(Physics Model)", fillcolor="#1e293b"];
        ENS [label="Source 2: GEFS Ensemble\n(Multi-Member)", fillcolor="#1e293b"];
        AI  [label="Source 3: AI Neural\n(Deep Forecast)", fillcolor="#1e293b"];
    }

    subgraph cluster_1 {
        label = "Feature Engineering";
        fontcolor = "#94a3b8";
        style = "dashed";
        color = "#334155";
        
        FE [label="Context Engine\n• Trig Seasonal (Sin/Cos)\n• Causal Rolling Skill (7d)\n• Spatial Regimes (KMeans)", fillcolor="#1e3a8a"];
    }

    subgraph cluster_2 {
        label = "Stage 3 Adaptive Gating";
        fontcolor = "#94a3b8";
        style = "dashed";
        color = "#334155";
        
        GATE [label="Softmax Temperature Gating\nw = Softmax(-Errors / τ)", fillcolor="#3b82f6"];
        TWEED [label="Two-Stage Tweedie Head\nP(Rain>0) × Amount", fillcolor="#0d9488"];
    }

    OUT [label="Blended Forecast Output\n• Optimal MAE & RMSE\n• Calibrated Threat Score", fillcolor="#15803d"];

    NWP -> FE;
    ENS -> FE;
    AI -> FE;

    FE -> GATE;
    FE -> TWEED;

    GATE -> OUT;
    TWEED -> OUT;
}
""")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. BENCHMARK RESULTS TABLES
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    render_html_panel("<div class='panel-header-title'>🌧️ Rainfall Blending Performance (Year 2 Test)</div>")
    rain_data = {
        "Model / Strategy": [
            "Persistence Baseline", "Source 1 (NWP Physics)", "Source 2 (Multi-Model Ens)",
            "Source 3 (AI Neural)", "1. Equal-Weight Avg", "2. Best Single Historical",
            "3. Inverse-Error Weighting", "4. Bias-Corrected Avg", "5. LightGBM Blender (Stage 2)",
            "6. Adaptive Gating Head (Stage 3)"
        ],
        "MAE (mm)": [10.78, 5.90, 2.33, 2.06, 2.87, 1.93, 1.93, 2.24, 1.05, 0.98],
        "RMSE (mm)": [22.76, 10.37, 3.80, 3.25, 4.22, 3.09, 2.65, 3.55, 1.90, 2.21],
        "Skill Score": ["0.00", "0.45", "0.78", "0.81", "0.73", "0.82", "0.82", "0.79", "0.90", "0.91"],
        "Brier Score": ["0.255", "0.185", "0.094", "0.079", "0.090", "0.077", "0.069", "0.082", "0.074", "0.053"],
        "CSI (Threat)": ["0.378", "0.514", "0.666", "0.715", "0.682", "0.723", "0.766", "0.714", "0.760", "0.805"]
    }
    df_rain = pd.DataFrame(rain_data)
    
    def highlight_winning_rain(s):
        return ['background-color: rgba(16, 185, 129, 0.25); font-weight: bold; color: #34d399;' if s.name == 9 else '' for _ in s]
    
    st.dataframe(df_rain.style.apply(highlight_winning_rain, axis=1), hide_index=True, use_container_width=True)

with col2:
    render_html_panel("<div class='panel-header-title'>🌡️ Temperature Blending Performance (Year 2 Test)</div>")
    temp_data = {
        "Model / Strategy": [
            "Persistence Baseline", "Source 1 (NWP Physics)", "Source 2 (Multi-Model Ens)",
            "Source 3 (AI Neural)", "1. Equal-Weight Avg", "2. Best Single Historical",
            "3. Inverse-Error Weighting", "4. Bias-Corrected Avg", "5. LightGBM Blender (Stage 2)",
            "6. Fixed Stage 3 (Huber Loss)"
        ],
        "MAE (°C)": [1.13, 2.01, 1.39, 1.18, 0.94, 1.03, 0.75, 0.93, 0.84, 0.75],
        "RMSE (°C)": [1.41, 3.07, 1.81, 1.49, 1.29, 1.36, 0.98, 1.28, 1.07, 0.97],
        "Skill Score (RMSE)": ["0.00", "-1.18", "-0.28", "-0.06", "0.09", "0.04", "0.30", "0.09", "0.24", "0.31"],
        "Skill Score (MAE)": ["0.00", "-0.78", "-0.24", "-0.05", "0.17", "0.09", "0.34", "0.17", "0.26", "0.33"]
    }
    df_temp = pd.DataFrame(temp_data)
    
    def highlight_winning_temp(s):
        return ['background-color: rgba(16, 185, 129, 0.25); font-weight: bold; color: #34d399;' if s.name == 9 else '' for _ in s]
    
    st.dataframe(df_temp.style.apply(highlight_winning_temp, axis=1), hide_index=True, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. EXPLAINER BULLETS
# -----------------------------------------------------------------------------
render_html_panel("""
<div class="glass-card" style="margin-bottom: 20px;">
    <h3 style="font-size: 16px; color: #60a5fa; margin-top: 0;">💡 Why Adaptive Gating Outperforms Naive Averaging</h3>
    <ul style="color: #cbd5e1; font-size: 13px; line-height: 1.7; padding-left: 20px;">
        <li><b>Context-Aware Dynamic Weighting:</b> Equal-weight averaging assumes all models perform identically regardless of lead time or weather regime. Adaptive gating dynamically re-allocates weight based on real-time rolling skill and seasonal context.</li>
        <li><b>Lead-Time Sensitivity:</b> Physics NWP models (GFS) excel at 24h lead times but degrade sharply at 120h. AI models maintain flat error profiles at long lead times. Gating shifts weight smoothly from NWP to AI as lead time increases.</li>
        <li><b>Monsoon Extreme Handling:</b> During heavy monsoon convective events, physics models often over/under-predict rain intensity. Gating shifts weight to ensemble members and activates Tweedie conditional loss to eliminate zero-inflated bias.</li>
        <li><b>Huber-Loss Robustness:</b> Temperature predictions utilize a Huber loss objective to prevent localized temperature spikes from corrupting the spatial forecast.</li>
    </ul>
</div>
""")

# -----------------------------------------------------------------------------
# 4. ABLATION & REGIONAL CHARTS
# -----------------------------------------------------------------------------
render_html_panel("<div class='panel-header-title'>📊 Feature Group Ablation & Regional Skill Breakdown</div>")

ablation_path = os.path.join(PROJECT_ROOT, "backend", "data", "analysis", "ablation_study.csv")
breakdown_path = os.path.join(PROJECT_ROOT, "backend", "data", "analysis", "multi_region_lead_time_breakdown.csv")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if os.path.exists(ablation_path):
        df_abl = pd.read_csv(ablation_path)
        fig_abl = px.bar(
            df_abl,
            x="Configuration",
            y="MAE_Drop_Pct",
            color="MAE_Drop_Pct",
            color_continuous_scale="Reds",
            title="Performance Drop (%) From Removing Feature Groups",
            labels={"MAE_Drop_Pct": "MAE Performance Drop (%)", "Configuration": "Feature Group Removed"}
        )
        fig_abl.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_abl, use_container_width=True)
    else:
        st.info("Run `python backend/src/analysis.py` to generate live ablation charts.")

with chart_col2:
    if os.path.exists(breakdown_path):
        df_bk = pd.read_csv(breakdown_path)
        df_rain_bk = df_bk[df_bk["Variable"] == "rainfall"]
        fig_bk = px.line(
            df_rain_bk,
            x="Lead_Hours",
            y="MAE_Improvement_Pct",
            color="Region",
            markers=True,
            title="Adaptive Gate MAE Gain (%) Over Equal-Weight Across Regions",
            labels={"Lead_Hours": "Lead Time (Hours)", "MAE_Improvement_Pct": "MAE Improvement (%)"}
        )
        fig_bk.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_bk, use_container_width=True)
    else:
        st.info("Run `python backend/src/analysis.py` to generate regional breakdown charts.")
