"""
Page 4: About - Problem Statement, Tech Stack, Honest Limitations & Future Roadmap
SIH26081 Submission Documentation for National-Level Judging.
"""

import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from components import inject_global_css, render_header_bar, render_html_panel, render_sidebar_navigation, CARD_BG, CARD_BORDER, TEXT_MUTED, BLUE_ACCENT

st.set_page_config(
    page_title="About & Roadmap - WeatherBlend AI",
    page_icon="ℹ️",
    layout="wide"
)

inject_global_css()
render_sidebar_navigation()
render_header_bar(region_name="SIH26081 Prototype")

render_html_panel("""
<div class="page-title-box">
    <h2>ℹ️ About WeatherBlend AI & SIH26081</h2>
    <p>Context-Aware Weather Forecast Blending & Gating Engine — Project Overview & Future Roadmap</p>
</div>
""")

# -----------------------------------------------------------------------------
# 1. PROBLEM STATEMENT & APPROACH
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    render_html_panel("""
    <div class="glass-card">
        <h3 style="font-size: 16px; color: #60a5fa; margin-top: 0;">📌 Problem Statement (SIH26081)</h3>
        <p style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
            Numerical Weather Prediction (NWP) models, ensemble systems, and deep learning weather models (e.g., FourCastNet, Pangu-Weather) frequently disagree in their regional forecasts. Traditional operational meteorology relies on simple ensemble averaging or fixed static weighting.
        </p>
        <p style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
            However, physics models degrade rapidly at long lead times (>72h), while AI models maintain low variance but miss extreme convective localized events. Static averaging fails to adapt to changing weather regimes and spatial contexts.
        </p>
    </div>
    """)

with col2:
    render_html_panel("""
    <div class="glass-card">
        <h3 style="font-size: 16px; color: #34d399; margin-top: 0;">⚡ The WeatherBlend AI Solution</h3>
        <p style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
            WeatherBlend AI introduces a <b>Context-Aware Machine Learning Blending & Adaptive Gating Engine</b>.
        </p>
        <p style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
            Instead of assigning fixed weights to forecast sources, WeatherBlend trains GBDT Softmax Gating networks that dynamically re-weigh forecast sources cell-by-cell based on:
        </p>
        <ul style="font-size: 12px; color: #94a3b8; padding-left: 18px; margin-bottom: 0;">
            <li>Forecast Horizon / Lead Time (24h to 120h)</li>
            <li>Causal Rolling Error Skill (7-day historical window)</li>
            <li>Trigonometric Seasonal & Cyclic Encodings</li>
            <li>Weather Regimes (KMeans Synoptic Clusters)</li>
        </ul>
    </div>
    """)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. TECHNOLOGY STACK
# -----------------------------------------------------------------------------
render_html_panel("<div class='panel-header-title'>🛠️ Technology Stack</div>")

render_html_panel("""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px;">
    <div class="glass-card" style="padding: 14px;">
        <div style="font-size: 14px; font-weight: 700; color: #60a5fa;">Backend Architecture</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">Python 3.10+<br/>FastAPI REST Server<br/>Pydantic & Joblib</div>
    </div>
    <div class="glass-card" style="padding: 14px;">
        <div style="font-size: 14px; font-weight: 700; color: #34d399;">Machine Learning</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">LightGBM Regressors<br/>Two-Stage Tweedie Head<br/>Huber & Softmax Loss</div>
    </div>
    <div class="glass-card" style="padding: 14px;">
        <div style="font-size: 14px; font-weight: 700; color: #fbbf24;">Geospatial & Visualization</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">PyDeck / Mapbox GL<br/>Scipy 2D Grid Interpolation<br/>Plotly Interactive Charts</div>
    </div>
    <div class="glass-card" style="padding: 14px;">
        <div style="font-size: 14px; font-weight: 700; color: #c084fc;">Frontend Dashboard</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">Streamlit Native Pages<br/>Glassmorphism UI Engine<br/>Custom HTML/CSS Tokens</div>
    </div>
</div>
""")

# -----------------------------------------------------------------------------
# 3. HONEST LIMITATIONS & FUTURE ROADMAP
# -----------------------------------------------------------------------------
c_lim, c_road = st.columns(2)

with c_lim:
    render_html_panel("""
    <div class="glass-card">
        <h3 style="font-size: 16px; color: #fbbf24; margin-top: 0;">⚠️ Current MVP Limitations (Honest Audit)</h3>
        <ul style="color: #cbd5e1; font-size: 13px; line-height: 1.6; padding-left: 20px;">
            <li><b>Synthetic Data Generator:</b> The current MVP utilizes realistic synthetic 2-year forecast streams simulated over Bihar, Jharkhand, and West Bengal grids.</li>
            <li><b>Spatial Grid Resolution:</b> Grids are evaluated at 0.05° resolution (~5.2 km). Full operational deployment will require 0.01° (~1 km) micro-grid scaling.</li>
            <li><b>Static Model Weights Artifacts:</b> Gating models are stored as pre-trained `.joblib` objects rather than streaming online continuous learning.</li>
        </ul>
    </div>
    """)

with c_road:
    render_html_panel("""
    <div class="glass-card">
        <h3 style="font-size: 16px; color: #60a5fa; margin-top: 0;">🚀 Next Steps & Production Roadmap</h3>
        <ul style="color: #cbd5e1; font-size: 13px; line-height: 1.6; padding-left: 20px;">
            <li><b>Real IMD & ERA5 Data Ingestion:</b> Connect real-time GFS, NCUM, GEFS, and ECMWF/GraphCast operational API streams.</li>
            <li><b>All-India 36 State Scaling:</b> Expand regional gating models to cover all 36 Indian states and Union Territories.</li>
            <li><b>Real-Time Extreme Weather Alerts:</b> Integrate Automated Weather Station (AWS) Doppler radar feeds for instant flash flood & heatwave Push Notifications.</li>
        </ul>
    </div>
    """)
