"""
WeatherBlend AI - Frontend Web Dashboard
Streamlit Application connected to FastAPI Backend (http://localhost:8000)
SIH26081 Prototype: Context-Aware Weather Forecast Blending & Gating Engine
"""

import os
import sys
import json
import datetime
import requests
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st
import scipy.ndimage as ndimage
from scipy.interpolate import griddata

try:
    from shapely.geometry import shape
    from shapely import contains_xy
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    shape = None
    contains_xy = None


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from components import CITY_LOCATIONS, REGION_BOUNDS, render_sidebar_navigation
except ImportError:
    from frontend.components import CITY_LOCATIONS, REGION_BOUNDS, render_sidebar_navigation




# -----------------------------------------------------------------------------
# CONSTANTS & PALETTE
# -----------------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000"

BG_DARK = "#090d16"
CARD_BG = "rgba(15, 23, 42, 0.75)"
CARD_BORDER = "rgba(255, 255, 255, 0.08)"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"

REGION_BOUNDS = {
    "Bihar": {"lat": 25.8, "lon": 85.8, "zoom": 7.1, "min_lat": 24.2, "max_lat": 27.5, "min_lon": 83.3, "max_lon": 88.3},
    "Jharkhand": {"lat": 23.6, "lon": 85.3, "zoom": 7.1, "min_lat": 21.8, "max_lat": 25.4, "min_lon": 83.3, "max_lon": 87.9},
    "West Bengal": {"lat": 23.8, "lon": 87.8, "zoom": 6.8, "min_lat": 21.5, "max_lat": 27.3, "min_lon": 85.8, "max_lon": 89.9},
    "Uttar Pradesh": {"lat": 26.8, "lon": 80.9, "zoom": 6.2, "min_lat": 23.8, "max_lat": 30.5, "min_lon": 77.1, "max_lon": 84.6},
    "Odisha": {"lat": 20.5, "lon": 84.4, "zoom": 6.8, "min_lat": 17.8, "max_lat": 22.6, "min_lon": 81.4, "max_lon": 87.6},
    "All India": {"lat": 22.5, "lon": 82.5, "zoom": 4.6, "min_lat": 8.0, "max_lat": 37.0, "min_lon": 68.0, "max_lon": 97.0}
}

def get_radar_color_rgba(val, v_min, v_max, alpha=200):
    norm = (val - v_min) / (v_max - v_min + 1e-5)
    norm = max(0.0, min(1.0, float(norm)))
    stops = [
        (0.0, [30, 20, 120]),
        (0.2, [0, 190, 240]),
        (0.4, [16, 185, 129]),
        (0.6, [250, 215, 0]),
        (0.8, [245, 120, 0]),
        (1.0, [235, 30, 30])
    ]
    for i in range(len(stops) - 1):
        s0, c0 = stops[i]
        s1, c1 = stops[i+1]
        if s0 <= norm <= s1:
            t = (norm - s0) / (s1 - s0)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return [r, g, b, alpha]
    return [235, 30, 30, alpha]

BLUE_PRIMARY = "#2563eb"
BLUE_ACCENT = "#3b82f6"
TEAL_ACCENT = "#14b8a6"
PURPLE_ACCENT = "#8b5cf6"
AMBER_WARN = "#f59e0b"
GREEN_SUCCESS = "#10b981"
RED_DANGER = "#ef4444"

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WeatherBlend AI - Hybrid AI Weather System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# HTML HELPER FUNCTION (PREVENTS MARKDOWN CODE BLOCK PARSING BUG)
# -----------------------------------------------------------------------------
def render_html_panel(html: str):
    """
    Renders custom HTML safely in Streamlit without triggering Markdown's 4-space code block parser.
    Strips excess leading indentation from every line, ensuring clean single-block HTML rendering.
    """
    lines = [line.strip() for line in html.strip().splitlines()]
    clean_html = "\n".join(lines)
    st.markdown(clean_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INJECTED GLOBAL CSS STYLING
# -----------------------------------------------------------------------------
render_html_panel(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background-color: {BG_DARK};
        color: {TEXT_MAIN};
    }}
    
    /* Hide Streamlit default header, footer, menu */
    #MainMenu {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    
    .main {{
        background-color: {BG_DARK};
        padding: 0.2rem 1rem 1rem 1rem;
    }}
    
    /* Top Header Bar */
    .wb-header-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px 20px;
        margin-bottom: 14px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }}
    .wb-title-group {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .wb-icon-box {{
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 0 16px rgba(59, 130, 246, 0.4);
    }}
    .wb-title {{
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.1;
    }}
    .wb-title span {{
        color: {BLUE_ACCENT};
    }}
    .wb-subtitle {{
        font-size: 12px;
        color: {TEXT_MUTED};
        margin: 2px 0 0 0;
        font-weight: 500;
    }}
    .wb-right-group {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .wb-tagline {{
        font-size: 13px;
        font-weight: 500;
        color: {TEXT_MUTED};
        letter-spacing: 0.3px;
    }}
    .wb-location-pill {{
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #60a5fa;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    
    /* Main Page Header Title */
    .page-title-box {{
        margin-bottom: 14px;
    }}
    .page-title-box h2 {{
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }}
    .page-title-box p {{
        font-size: 13px;
        color: {TEXT_MUTED};
        margin: 4px 0 0 0;
        font-weight: 500;
    }}
    
    /* Glass Cards */
    .glass-card {{
        background: {CARD_BG};
        backdrop-filter: blur(16px);
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        height: 100%;
    }}
    
    .panel-header-title {{
        font-size: 14px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    /* Metric Card Styling */
    .metric-flex {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
    }}
    .metric-icon-box {{
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.25);
    }}
    .metric-card-title {{
        font-size: 12px;
        color: {TEXT_MUTED};
        font-weight: 600;
        margin-top: 8px;
    }}
    .metric-val {{
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        margin-top: 2px;
    }}
    .metric-sub {{
        font-size: 12px;
        margin-top: 6px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    
    /* Badges */
    .badge-pill {{
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }}
    .badge-red {{
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    .badge-orange {{
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }}
    .badge-green {{
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }}
    
    /* Summary Table Styling */
    .summary-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .summary-table td {{
        padding: 9px 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 13px;
    }}
    .summary-table tr:last-child td {{
        border-bottom: none;
    }}
    .summary-label {{
        color: {TEXT_MUTED};
        font-weight: 500;
    }}
    .summary-value {{
        color: #ffffff;
        font-weight: 700;
        text-align: right;
    }}
    .advisory-box {{
        margin-top: 10px;
        padding: 10px 12px;
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid {RED_DANGER};
        border-radius: 6px;
        font-size: 12px;
        color: #fca5a5;
        line-height: 1.4;
    }}
    
    /* Progress Bars */
    .weight-row {{
        margin-bottom: 12px;
    }}
    .weight-row:last-child {{
        margin-bottom: 0;
    }}
    .weight-header {{
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .weight-bar-bg {{
        background: rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }}
    .weight-bar-fill {{
        height: 100%;
        border-radius: 6px;
    }}
    
    /* Hourly Forecast Strip */
    .hourly-container {{
        display: flex;
        gap: 8px;
        justify-content: space-between;
        overflow-x: auto;
    }}
    .hourly-slot {{
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 10px 6px;
        text-align: center;
        flex: 1;
        min-width: 65px;
    }}
    .hourly-time {{
        font-size: 11px;
        color: {TEXT_MUTED};
        font-weight: 600;
    }}
    .hourly-icon {{
        font-size: 20px;
        margin: 6px 0;
    }}
    .hourly-rain {{
        font-size: 13px;
        font-weight: 700;
        color: #ffffff;
    }}
    .hourly-temp {{
        font-size: 11px;
        color: #f97316;
        font-weight: 600;
        margin-top: 4px;
    }}
    
    /* Buttons & Form controls */
    .stButton>button {{
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: #0d1322;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}
    
    /* Active Quick Region Button highlight */
    .q-btn-active button {{
        background-color: {BLUE_PRIMARY} !important;
        color: white !important;
        border: 1px solid {BLUE_ACCENT} !important;
    }}
</style>
""")

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "country" not in st.session_state:
    st.session_state["country"] = "India"
if "region" not in st.session_state:
    st.session_state["region"] = "Bihar"
if "variable" not in st.session_state:
    st.session_state["variable"] = "Rainfall (mm)"
if "forecast_date" not in st.session_state:
    st.session_state["forecast_date"] = datetime.date(2026, 9, 26)
if "horizon" not in st.session_state:
    st.session_state["horizon"] = "Next 24 Hours"
if "geojson_data" not in st.session_state:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geojson_path = os.path.join(project_root, "backend", "data", "geo", "india_states.geojson")
    if os.path.exists(geojson_path):
        try:
            with open(geojson_path, "r") as f:
                st.session_state["geojson_data"] = json.load(f)
        except Exception:
            st.session_state["geojson_data"] = None
    else:
        st.session_state["geojson_data"] = None

# -----------------------------------------------------------------------------
# BACKEND API FETCH HELPERS
# -----------------------------------------------------------------------------
def fetch_api(endpoint: str, payload: dict):
    url = f"{BACKEND_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=6)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def parse_lead_hours(h_str):
    if isinstance(h_str, int):
        return h_str
    s = str(h_str).replace("Next ", "").replace(" Hours", "").replace("h", "")
    try:
        return int(s)
    except ValueError:
        return 24

def parse_var_name(v_str):
    if "Rain" in v_str:
        return "Rainfall"
    elif "Temp" in v_str:
        return "Temperature"
    elif "Wind" in v_str:
        return "Wind Speed"
    return v_str

def get_summary_data():
    lead_h = parse_lead_hours(st.session_state["horizon"])
    var_clean = parse_var_name(st.session_state["variable"])
    payload = {
        "country": st.session_state["country"],
        "region": st.session_state["region"],
        "variable": var_clean,
        "date_str": st.session_state["forecast_date"].strftime("%Y-%m-%d"),
        "lead_hours": lead_h
    }
    res = fetch_api("/api/forecast-summary", payload)
    if res:
        return res
    # Fallback default values if backend offline
    return {
        "predicted_rainfall_mm": 42.5,
        "trend_delta": 4.5,
        "trend_pct": 12.0,
        "heavy_rain_risk_pct": 78.0,
        "heavy_badge": "High Risk",
        "heavy_badge_color": RED_DANGER,
        "confidence_pct": 92.0,
        "conf_badge": "High",
        "conf_badge_color": GREEN_SUCCESS,
        "temp_min": 26.0,
        "temp_max": 32.0,
        "avg_value": 42.5,
        "max_value": 120.3,
        "min_value": 5.2,
        "expected_conditions": "Moderate to Heavy Rain",
        "advisory": "⚠️ Be prepared for heavy rainfall in some districts.",
        "weights": {"gfs_nwp": 28.0, "gefs_ensemble": 42.0, "ai_model": 30.0}
    }

def get_map_data():
    lead_h = parse_lead_hours(st.session_state["horizon"])
    var_clean = parse_var_name(st.session_state["variable"])
    payload = {
        "country": st.session_state["country"],
        "region": st.session_state["region"],
        "variable": var_clean,
        "date_str": st.session_state["forecast_date"].strftime("%Y-%m-%d"),
        "lead_hours": lead_h
    }
    res = fetch_api("/api/map-grid", payload)
    if res:
        return res
    # Fallback map grid data
    lats = [24.5, 25.2, 25.9, 26.6, 27.3]
    lons = [83.5, 84.5, 85.5, 86.5, 87.5]
    grid = []
    for lat in lats:
        for lon in lons:
            val = round(12.0 + (lat - 24.5)*8.0 + (lon - 83.5)*5.0, 1)
            grid.append({
                "lat": lat, "lon": lon, "value": val,
                "elevation": val * 120.0, "color": [239, 68, 68, 200] if val > 30 else ([16, 185, 129, 200] if val > 15 else [37, 99, 235, 200])
            })
    return {"center_lat": 25.9, "center_lon": 85.5, "zoom": 6.8, "grid_points": grid}

def get_hourly_data():
    lead_h = parse_lead_hours(st.session_state["horizon"])
    var_clean = parse_var_name(st.session_state["variable"])
    payload = {
        "country": st.session_state["country"],
        "region": st.session_state["region"],
        "variable": var_clean,
        "date_str": st.session_state["forecast_date"].strftime("%Y-%m-%d"),
        "lead_hours": lead_h
    }
    res = fetch_api("/api/hourly-forecast", payload)
    if res and "hourly_slots" in res:
        return res["hourly_slots"]
    return [
        {"time": "00:00", "icon": "🌧️", "rainfall_mm": 12, "temp_c": 26},
        {"time": "03:00", "icon": "🌧️", "rainfall_mm": 18, "temp_c": 26},
        {"time": "06:00", "icon": "🌧️", "rainfall_mm": 25, "temp_c": 27},
        {"time": "09:00", "icon": "🌧️", "rainfall_mm": 42, "temp_c": 29},
        {"time": "12:00", "icon": "⛈️", "rainfall_mm": 38, "temp_c": 31},
        {"time": "15:00", "icon": "⛈️", "rainfall_mm": 28, "temp_c": 32},
        {"time": "18:00", "icon": "🌧️", "rainfall_mm": 15, "temp_c": 30},
        {"time": "21:00", "icon": "🌩️", "rainfall_mm": 10, "temp_c": 28}
    ]

def get_model_comp_data():
    lead_h = parse_lead_hours(st.session_state["horizon"])
    var_clean = parse_var_name(st.session_state["variable"])
    payload = {
        "country": st.session_state["country"],
        "region": st.session_state["region"],
        "variable": var_clean,
        "date_str": st.session_state["forecast_date"].strftime("%Y-%m-%d"),
        "lead_hours": lead_h
    }
    res = fetch_api("/api/model-comparison", payload)
    if res and "models" in res:
        return res["models"]
    return [
        {"model": "GFS", "value": 38.2, "is_weatherblend": False},
        {"model": "GEFS", "value": 46.1, "is_weatherblend": False},
        {"model": "AI Model", "value": 44.3, "is_weatherblend": False},
        {"model": "Equal Mean", "value": 42.9, "is_weatherblend": False},
        {"model": "WeatherBlend (AI Blend)", "value": 42.5, "is_weatherblend": True}
    ]

# Fetch summary data
summary = get_summary_data()

# -----------------------------------------------------------------------------
# TOP HEADER BAR
# -----------------------------------------------------------------------------
selected_region_name = st.session_state["region"]
render_html_panel(f"""
<div class="wb-header-bar">
    <div class="wb-title-group">
        <div class="wb-icon-box">🌤️</div>
        <div>
            <h1 class="wb-title">WeatherBlend <span>AI</span></h1>
            <p class="wb-subtitle">Hybrid AI – NWP Forecast System</p>
        </div>
    </div>
    <div class="wb-right-group">
        <span class="wb-tagline">Accurate. Adaptive. Ahead.</span>
        <div class="wb-location-pill">
            <span>📍</span> {selected_region_name} Regional Forecast
        </div>
    </div>
</div>
""")

# -----------------------------------------------------------------------------
# SIDEBAR ("FORECAST SETTINGS")
# -----------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_navigation()
    st.markdown("### ⚙️ Forecast Settings")
    
    country_sel = st.selectbox("Country", ["India"], index=0)
    
    regions_list = ["Bihar", "Jharkhand", "Uttar Pradesh", "West Bengal", "Odisha", "All India"]
    reg_idx = regions_list.index(st.session_state["region"]) if st.session_state["region"] in regions_list else 0
    region_sel = st.selectbox("State / Region", regions_list, index=reg_idx)
    
    date_sel = st.date_input("Date", value=st.session_state["forecast_date"])
    
    horizons_list = ["Next 24 Hours", "Next 48 Hours", "Next 72 Hours", "Next 120 Hours"]
    hor_idx = horizons_list.index(st.session_state["horizon"]) if st.session_state["horizon"] in horizons_list else 0
    horizon_sel = st.selectbox("Forecast Horizon", horizons_list, index=hor_idx)
    
    vars_list = ["Rainfall (mm)", "Temperature (°C)", "Wind Speed (km/h)"]
    var_idx = vars_list.index(st.session_state["variable"]) if st.session_state["variable"] in vars_list else 0
    variable_sel = st.selectbox("Weather Variable", vars_list, index=var_idx)
    
    st.session_state["country"] = country_sel
    st.session_state["region"] = region_sel
    st.session_state["forecast_date"] = date_sel
    st.session_state["horizon"] = horizon_sel
    st.session_state["variable"] = variable_sel
    
    if st.button("☁️ Generate Forecast", use_container_width=True, type="primary"):
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Quick Regions")
    
    # 6 quick region buttons grid matching screenshot
    q_col1, q_col2, q_col3 = st.columns(3)
    if q_col1.button("Bihar", key="q_bihar", use_container_width=True):
        st.session_state["region"] = "Bihar"
        st.rerun()
    if q_col2.button("Jharkhand", key="q_jh", use_container_width=True):
        st.session_state["region"] = "Jharkhand"
        st.rerun()
    if q_col3.button("Uttar Pradesh", key="q_up", use_container_width=True):
        st.session_state["region"] = "Uttar Pradesh"
        st.rerun()

    q_col4, q_col5, q_col6 = st.columns(3)
    if q_col4.button("West Bengal", key="q_wb", use_container_width=True):
        st.session_state["region"] = "West Bengal"
        st.rerun()
    if q_col5.button("Odisha", key="q_od", use_container_width=True):
        st.session_state["region"] = "Odisha"
        st.rerun()
    if q_col6.button("All India", key="q_ai", use_container_width=True):
        st.session_state["region"] = "All India"
        st.rerun()

    # Sidebar Bottom Graphic: Map Graphic of India/Region
    render_html_panel(f"""
    <div style="margin-top: 24px; padding: 12px; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; text-align: center;">
        <svg width="120" height="120" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M35 15 L50 10 L65 18 L70 30 L85 40 L80 55 L75 60 L60 85 L45 90 L30 75 L20 60 L25 40 Z" fill="rgba(37, 99, 235, 0.15)" stroke="rgba(59, 130, 246, 0.4)" stroke-width="1.5" stroke-dasharray="3 3"/>
            <!-- Highlighted region point/area -->
            <circle cx="58" cy="38" r="8" fill="rgba(59, 130, 246, 0.6)" stroke="#60a5fa" stroke-width="2"/>
            <text x="58" y="55" font-size="8" fill="#60a5fa" font-weight="bold" text-anchor="middle">{st.session_state['region']}</text>
        </svg>
    </div>
    """)

# -----------------------------------------------------------------------------
# MAIN PAGE TITLE HEADER ROW
# -----------------------------------------------------------------------------
formatted_date_str = st.session_state["forecast_date"].strftime("%d %B %Y")
render_html_panel(f"""
<div class="page-title-box">
    <h2>{st.session_state['region']} Forecast</h2>
    <p>{formatted_date_str} &nbsp;•&nbsp; {st.session_state['horizon']} &nbsp;•&nbsp; {st.session_state['variable']}</p>
</div>
""")

# -----------------------------------------------------------------------------
# TOP METRIC ROW (4 CARDS MATCHING SCREENSHOT)
# -----------------------------------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

var_unit = "mm" if "Rain" in st.session_state["variable"] else ("°C" if "Temp" in st.session_state["variable"] else "km/h")
trend_val = summary.get("trend_pct", 12.0)
trend_icon = "↑" if trend_val >= 0 else "↓"
trend_color = GREEN_SUCCESS if trend_val >= 0 else RED_DANGER

with m_col1:
    render_html_panel(f"""
    <div class="glass-card">
        <div class="metric-flex">
            <div>
                <div class="metric-val">{summary['predicted_rainfall_mm']} <span style="font-size: 16px; color: {TEXT_MUTED};">{var_unit}</span></div>
                <div class="metric-card-title">Predicted {parse_var_name(st.session_state['variable'])}</div>
            </div>
            <div class="metric-icon-box">🌧️</div>
        </div>
        <div class="metric-sub" style="color: {trend_color};">
            <span>{trend_icon} {abs(trend_val)}% vs. yesterday</span>
        </div>
    </div>
    """)

with m_col2:
    risk_badge = summary.get("heavy_badge", "High Risk")
    render_html_panel(f"""
    <div class="glass-card">
        <div class="metric-flex">
            <div>
                <div class="metric-val">{summary['heavy_rain_risk_pct']}%</div>
                <div class="metric-card-title">Heavy Rain Risk</div>
            </div>
            <div class="metric-icon-box">🎯</div>
        </div>
        <div class="metric-sub">
            <span class="badge-pill badge-red">{risk_badge}</span>
        </div>
    </div>
    """)

with m_col3:
    conf_badge = summary.get("conf_badge", "High")
    if "High" in str(conf_badge):
        conf_str = "High"
    elif "Med" in str(conf_badge):
        conf_str = "Medium"
    else:
        conf_str = "Low"
    render_html_panel(f"""
    <div class="glass-card">
        <div class="metric-flex">
            <div>
                <div class="metric-val">{summary['confidence_pct']}%</div>
                <div class="metric-card-title">Forecast Confidence</div>
            </div>
            <div class="metric-icon-box">🛡️</div>
        </div>
        <div class="metric-sub">
            <span class="badge-pill badge-green">{conf_str}</span>
        </div>
    </div>
    """)

with m_col4:
    render_html_panel(f"""
    <div class="glass-card">
        <div class="metric-flex">
            <div>
                <div class="metric-val">{int(summary['temp_min'])}–{int(summary['temp_max'])}°C</div>
                <div class="metric-card-title">Temperature Range</div>
            </div>
            <div class="metric-icon-box">🌡️</div>
        </div>
        <div class="metric-sub" style="color: {TEXT_MUTED};">
            <span>Diurnal Span</span>
        </div>
    </div>
    """)

st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN CONTENT ROW (MAP & RIGHT COLUMN PANELS)
# -----------------------------------------------------------------------------
left_map_col, right_panel_col = st.columns([1.75, 1.0])

with left_map_col:
    map_title_text = f"🌧️ {st.session_state['region']} Interactive Forecast Engine"
    render_html_panel(f"<div class='panel-header-title'>{map_title_text}</div>")

    # Map Toolbar Controls (Layer Toggle, Time Slider, Grid Opacity, Compare 2x2, Cell Inspector)
    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4, c_ctrl5 = st.columns([1.3, 1.6, 1.1, 1.0, 1.0])
    
    with c_ctrl1:
        map_layer_var = st.radio(
            "Map Layer",
            ["Rainfall", "Temperature", "Wind Speed"],
            horizontal=True,
            key="map_layer_toggle",
            label_visibility="collapsed"
        )

    with c_ctrl2:
        selected_lead_step = st.select_slider(
            "Forecast Step",
            options=[3, 6, 12, 24, 36, 48, 72, 96, 120],
            value=parse_lead_hours(st.session_state["horizon"]),
            format_func=lambda x: f"+{x}h Horizon",
            key="map_time_slider"
        )

    with c_ctrl3:
        map_opacity_pct = st.slider(
            "Grid Opacity",
            min_value=10,
            max_value=90,
            value=65,
            step=5,
            format="%d%%",
            key="map_opacity_slider"
        )

    with c_ctrl4:
        compare_2x2 = st.checkbox("🔍 Compare (2x2)", value=False, key="compare_2x2_toggle")

    with c_ctrl5:
        show_inspector = st.checkbox("📍 Cell Inspect", value=True, key="cell_inspect_toggle")

    selected_region = st.session_state["region"]
    r_info = REGION_BOUNDS.get(selected_region, REGION_BOUNDS["Bihar"])

    grid_alpha = int(255 * (map_opacity_pct / 100.0))

    # Fetch map data for the specific time step and variable
    payload_map = {
        "country": st.session_state["country"],
        "region": selected_region,
        "variable": map_layer_var,
        "date_str": st.session_state["forecast_date"].strftime("%Y-%m-%d"),
        "lead_hours": selected_lead_step
    }
    res_map = fetch_api("/api/map-grid", payload_map)
    
    if res_map and "grid_points" in res_map:
        raw_grid = res_map["grid_points"]
    else:
        # Fallback grid data
        lats_fb = [24.5, 25.2, 25.9, 26.6, 27.3]
        lons_fb = [83.5, 84.5, 85.5, 86.5, 87.5]
        raw_grid = []
        for la in lats_fb:
            for lo in lons_fb:
                v = round(12.0 + (la - 24.5)*8.0 + (lo - 83.5)*5.0 + (selected_lead_step*0.2), 1)
                raw_grid.append({
                    "lat": la, "lon": lo, "value": v,
                    "nwp_value": round(v * 0.9, 1), "ensemble_value": round(v * 1.1, 1), "ai_value": round(v * 1.05, 1)
                })

    # High-resolution interpolated grid (~0.025° resolution ~2.5km)
    pts = np.array([[pt["lon"], pt["lat"]] for pt in raw_grid])
    vals = np.array([pt["value"] for pt in raw_grid])
    nwps = np.array([pt.get("nwp_value", pt["value"]) for pt in raw_grid])
    enss = np.array([pt.get("ensemble_value", pt["value"]) for pt in raw_grid])
    ais = np.array([pt.get("ai_value", pt["value"]) for pt in raw_grid])

    grid_step = 0.025 if selected_region != "All India" else 0.12
    fine_lats = np.arange(r_info["min_lat"], r_info["max_lat"] + grid_step, grid_step)
    fine_lons = np.arange(r_info["min_lon"], r_info["max_lon"] + grid_step, grid_step)
    mesh_lon, mesh_lat = np.meshgrid(fine_lons, fine_lats)

    try:
        interp_val = griddata(pts, vals, (mesh_lon, mesh_lat), method='cubic')
        near_val = griddata(pts, vals, (mesh_lon, mesh_lat), method='nearest')
        interp_val = np.where(np.isnan(interp_val), near_val, interp_val)

        interp_nwp = griddata(pts, nwps, (mesh_lon, mesh_lat), method='cubic')
        near_nwp = griddata(pts, nwps, (mesh_lon, mesh_lat), method='nearest')
        interp_nwp = np.where(np.isnan(interp_nwp), near_nwp, interp_nwp)

        interp_ens = griddata(pts, enss, (mesh_lon, mesh_lat), method='cubic')
        near_ens = griddata(pts, enss, (mesh_lon, mesh_lat), method='nearest')
        interp_ens = np.where(np.isnan(interp_ens), near_ens, interp_ens)

        interp_ai = griddata(pts, ais, (mesh_lon, mesh_lat), method='cubic')
        near_ai = griddata(pts, ais, (mesh_lon, mesh_lat), method='nearest')
        interp_ai = np.where(np.isnan(interp_ai), near_ai, interp_ai)
    except Exception:
        interp_val = griddata(pts, vals, (mesh_lon, mesh_lat), method='nearest')
        interp_nwp = interp_val
        interp_ens = interp_val
        interp_ai = interp_val

    # Apply smooth Gaussian blur pass so it reads as continuous radar imagery (no blockiness)
    interp_val = ndimage.gaussian_filter(interp_val, sigma=0.8)
    interp_nwp = ndimage.gaussian_filter(interp_nwp, sigma=0.8)
    interp_ens = ndimage.gaussian_filter(interp_ens, sigma=0.8)
    interp_ai = ndimage.gaussian_filter(interp_ai, sigma=0.8)

    v_min, v_max = float(interp_val.min()), float(interp_val.max())

    # -------------------------------------------------------------------------
    # 1. GEOJSON POLYGON CLIPPING (SHAPELY)
    # -------------------------------------------------------------------------
    geojson_data = st.session_state.get("geojson_data")
    state_polygons = {}
    if geojson_data and "features" in geojson_data:
        for feat in geojson_data["features"]:
            p_name = feat["properties"].get("name")
            if p_name:
                try:
                    state_polygons[p_name] = shape(feat["geometry"])
                except Exception:
                    pass

    target_poly = state_polygons.get(selected_region)
    if target_poly and selected_region != "All India":
        if HAS_SHAPELY and contains_xy is not None:
            try:
                buffered_poly = target_poly.buffer(0.008)
                clip_mask = contains_xy(buffered_poly, mesh_lon, mesh_lat)
            except Exception:
                clip_mask = np.ones(mesh_lat.shape, dtype=bool)
        else:
            try:
                import matplotlib.path as mpath
                coords = list(target_poly.exterior.coords) if hasattr(target_poly, "exterior") else []
                if coords:
                    path = mpath.Path(coords)
                    pts_grid = np.column_stack((mesh_lon.ravel(), mesh_lat.ravel()))
                    clip_mask = path.contains_points(pts_grid).reshape(mesh_lat.shape)
                else:
                    clip_mask = np.ones(mesh_lat.shape, dtype=bool)
            except Exception:
                clip_mask = np.ones(mesh_lat.shape, dtype=bool)
    else:
        clip_mask = np.ones(mesh_lat.shape, dtype=bool)

    fine_records = []
    for i in range(mesh_lat.shape[0]):
        for j in range(mesh_lat.shape[1]):
            if not clip_mask[i, j]:
                continue  # Clip: Only render grid cells that fall INSIDE the selected state's polygon!
                
            v = float(interp_val[i, j])
            la = round(float(mesh_lat[i, j]), 4)
            lo = round(float(mesh_lon[i, j]), 4)
            color = get_radar_color_rgba(v, v_min, v_max, alpha=grid_alpha)
            
            color_nwp = get_radar_color_rgba(float(interp_nwp[i, j]), v_min, v_max, alpha=grid_alpha)
            color_ens = get_radar_color_rgba(float(interp_ens[i, j]), v_min, v_max, alpha=grid_alpha)
            color_ai = get_radar_color_rgba(float(interp_ai[i, j]), v_min, v_max, alpha=grid_alpha)

            fine_records.append({
                "lat": la,
                "lon": lo,
                "value": round(v, 1),
                "nwp_value": round(float(interp_nwp[i, j]), 1),
                "ensemble_value": round(float(interp_ens[i, j]), 1),
                "ai_value": round(float(interp_ai[i, j]), 1),
                "color": color,
                "color_nwp": color_nwp,
                "color_ens": color_ens,
                "color_ai": color_ai
            })
    fine_grid_df = pd.DataFrame(fine_records)

    cell_size = 2800 if selected_region != "All India" else 14000

    # -------------------------------------------------------------------------
    # 3. BOUNDARY VISIBILITY (NEON CYAN & WHITE STROKES)
    # -------------------------------------------------------------------------
    boundary_layers = []
    if geojson_data and "features" in geojson_data:
        selected_features = [f for f in geojson_data["features"] if f["properties"].get("name") == selected_region]
        other_features = [f for f in geojson_data["features"] if f["properties"].get("name") != selected_region]

        # General/Other state internal borders
        if other_features:
            other_boundary_layer = pdk.Layer(
                "GeoJsonLayer",
                data={"type": "FeatureCollection", "features": other_features},
                stroked=True,
                filled=False,
                get_line_color=[148, 163, 184, 120],
                get_line_width=1200,
                line_width_min_pixels=1.2,
                pickable=False
            )
            boundary_layers.append(other_boundary_layer)

        # Selected Active State Outline Highlight (Double layer for glowing neon line effect)
        if selected_features:
            selected_geojson = {"type": "FeatureCollection", "features": selected_features}
            # Outer cyan glow stroke
            state_glow_layer = pdk.Layer(
                "GeoJsonLayer",
                data=selected_geojson,
                stroked=True,
                filled=False,
                get_line_color=[6, 182, 212, 160],
                get_line_width=3400 if selected_region != "All India" else 6000,
                line_width_min_pixels=4.5,
                pickable=False
            )
            # Inner bright white stroke core
            state_core_layer = pdk.Layer(
                "GeoJsonLayer",
                data=selected_geojson,
                stroked=True,
                filled=False,
                get_line_color=[255, 255, 255, 255],
                get_line_width=1600 if selected_region != "All India" else 3000,
                line_width_min_pixels=2.5,
                pickable=False
            )
            boundary_layers.extend([state_glow_layer, state_core_layer])

    # -------------------------------------------------------------------------
    # 4. LABEL DECLUTTER (DISTRICT & CENTROID DISTANCE FILTERING)
    # -------------------------------------------------------------------------
    c_lat = r_info["lat"]
    c_lon = r_info["lon"]
    
    # Major Cities inside active state
    city_data = CITY_LOCATIONS.get(selected_region, CITY_LOCATIONS["Bihar"])
    
    city_markers_layer = pdk.Layer(
        "ScatterplotLayer",
        data=city_data,
        get_position=["lon", "lat"],
        get_fill_color=[6, 182, 212, 255],
        get_line_color=[255, 255, 255, 255],
        get_line_width=2,
        line_width_min_pixels=2,
        get_radius=3200 if selected_region != "All India" else 15000,
        radius_min_pixels=4,
        radius_max_pixels=8,
        pickable=True
    )

    city_text_layer = pdk.Layer(
        "TextLayer",
        data=city_data,
        get_position=["lon", "lat"],
        get_text="name",
        get_color=[255, 255, 255, 255],
        get_size=12,
        get_pixel_offset=[0, -14],
        get_alignment_baseline="'bottom'",
        get_text_anchor="'middle'",
        background=True,
        get_background_color=[15, 23, 42, 230],
        get_border_color=[6, 182, 212, 255],
        get_border_width=1.5,
        background_padding=[6, 3],
        font_family="'Inter', system-ui, sans-serif",
        font_weight="bold"
    )

    # Filter neighboring region labels strictly by distance to centroid (< 2.8°)
    all_region_labels = [
        {"name": "NEPAL", "lat": 27.8, "lon": 85.2},
        {"name": "BIHAR", "lat": 25.85, "lon": 85.75},
        {"name": "JHARKHAND", "lat": 23.6, "lon": 85.5},
        {"name": "WEST BENGAL", "lat": 24.3, "lon": 87.8},
        {"name": "UTTAR PRADESH", "lat": 26.9, "lon": 81.2},
        {"name": "ODISHA", "lat": 20.3, "lon": 84.5},
        {"name": "BANGLADESH", "lat": 23.8, "lon": 89.8}
    ]
    filtered_border_labels = []
    for lbl in all_region_labels:
        d = float(np.sqrt((lbl["lat"] - c_lat)**2 + (lbl["lon"] - c_lon)**2))
        if lbl["name"] == selected_region.upper() or d < 2.8:
            filtered_border_labels.append(lbl)

    region_labels_layer = pdk.Layer(
        "TextLayer",
        data=filtered_border_labels,
        get_position=["lon", "lat"],
        get_text="name",
        get_color=[255, 255, 255, 210],
        get_size=13,
        get_alignment_baseline="'center'",
        get_text_anchor="'middle'",
        font_family="'Inter', system-ui, sans-serif",
        font_weight="bold"
    )

    mapbox_token = os.getenv("MAPBOX_TOKEN", "")
    map_style = "mapbox://styles/mapbox/satellite-streets-v12" if mapbox_token else "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"

    view_state = pdk.ViewState(
        latitude=r_info["lat"],
        longitude=r_info["lon"],
        zoom=r_info["zoom"],
        pitch=0,
        bearing=0
    )

    var_unit = "mm" if "Rain" in map_layer_var else ("°C" if "Temp" in map_layer_var else "km/h")

    # -------------------------------------------------------------------------
    # CONDITIONAL RENDER: 2x2 COMPARISON VS SINGLE MAIN MAP
    # -------------------------------------------------------------------------
    if compare_2x2:
        render_html_panel("<div style='font-size: 13px; color: #60a5fa; font-weight: 700; margin-bottom: 8px;'>🔍 2x2 Source Disagreement & Adaptive Blend Comparison</div>")
        
        cmp_col1, cmp_col2 = st.columns(2)
        
        def make_deck(fill_col_attr, title_str):
            layer_sub = pdk.Layer(
                "GridCellLayer",
                data=fine_grid_df,
                get_position=["lon", "lat"],
                cell_size=cell_size,
                get_fill_color=fill_col_attr,
                get_elevation=0,
                extruded=False,
                pickable=True
            )
            return pdk.Deck(
                layers=[layer_sub] + boundary_layers + [city_markers_layer, city_text_layer, region_labels_layer],
                initial_view_state=view_state,
                map_style=map_style,
                api_keys={"mapbox": mapbox_token} if mapbox_token else None,
                tooltip={"html": f"<b>{title_str}:</b> {{value}} {var_unit}"}
            )

        with cmp_col1:
            st.markdown("<div style='font-size: 11px; font-weight: 700; color: #ef4444;'>1. GFS NWP (Physics Model)</div>", unsafe_allow_html=True)
            st.pydeck_chart(make_deck("color_nwp", "GFS NWP"), use_container_width=True, height=200)
            
            st.markdown("<div style='font-size: 11px; font-weight: 700; color: #a855f7;'>3. AI Neural Forecast</div>", unsafe_allow_html=True)
            st.pydeck_chart(make_deck("color_ai", "AI Model"), use_container_width=True, height=200)

        with cmp_col2:
            st.markdown("<div style='font-size: 11px; font-weight: 700; color: #f59e0b;'>2. GEFS Ensemble</div>", unsafe_allow_html=True)
            st.pydeck_chart(make_deck("color_ens", "GEFS Ensemble"), use_container_width=True, height=200)
            
            st.markdown("<div style='font-size: 11px; font-weight: 700; color: #34d399;'>4. WeatherBlend (Adaptive Gate)</div>", unsafe_allow_html=True)
            st.pydeck_chart(make_deck("color", "WeatherBlend"), use_container_width=True, height=200)

    else:
        # SINGLE FULL MAP
        data_layer = pdk.Layer(
            "GridCellLayer",
            data=fine_grid_df,
            get_position=["lon", "lat"],
            cell_size=cell_size,
            get_fill_color="color",
            get_elevation=0,
            extruded=False,
            pickable=True,
            auto_highlight=True,
        )

        deck_map = pdk.Deck(
            layers=[data_layer] + boundary_layers + [city_markers_layer, city_text_layer, region_labels_layer],
            initial_view_state=view_state,
            map_style=map_style,
            api_keys={"mapbox": mapbox_token} if mapbox_token else None,
            tooltip={
                "html": f"<b>Grid Location:</b> {{lat}}°N, {{lon}}°E<br/>"
                        f"<b>{map_layer_var}:</b> <b style='color: #60a5fa;'>{{value}} {var_unit}</b><br/>"
                        f"<div style='margin-top: 4px; font-size: 11px; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.12); padding-top: 4px;'>"
                        f"GFS: {{nwp_value}} | GEFS: {{ensemble_value}} | AI: {{ai_value}}</div>",
                "style": {
                    "backgroundColor": "rgba(15, 23, 42, 0.92)",
                    "color": "#ffffff",
                    "border": "1px solid rgba(255,255,255,0.12)",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "fontSize": "12px"
                }
            }
        )

        map_c1, map_c2 = st.columns([5, 1])
        with map_c1:
            st.pydeck_chart(deck_map, use_container_width=True, height=380)
        with map_c2:
            if "Rain" in map_layer_var:
                leg_title = "Rainfall (mm)"
                leg_ticks = ["150+", "100", "60", "30", "15", "5", "0"]
            elif "Temp" in map_layer_var:
                leg_title = "Temp (°C)"
                leg_ticks = ["45+", "38", "32", "25", "18", "10"]
            else:
                leg_title = "Wind (km/h)"
                leg_ticks = ["70+", "55", "40", "25", "10", "0"]

            ticks_html = "".join([f"<span>{t}</span>" for t in leg_ticks])

            render_html_panel(f"""
            <div style="background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px 10px; text-align: center; height: 380px; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="font-size: 11px; font-weight: 700; color: #ffffff; letter-spacing: 0.5px;">{leg_title}</div>
                <div style="display: flex; height: 300px; justify-content: center; gap: 10px; align-items: center;">
                    <div style="width: 12px; height: 100%; border-radius: 6px; background: linear-gradient(to bottom, #eb1e1e, #f57800, #fad700, #10b981, #00bef0, #1e1478); border: 1px solid rgba(255,255,255,0.15);"></div>
                    <div style="display: flex; flex-direction: column; justify-content: space-between; height: 100%; font-weight: 600; font-size: 10px; color: #94a3b8;">
                        {ticks_html}
                    </div>
                </div>
            </div>
            """)

    # -------------------------------------------------------------------------
    # CLICK-TO-INSPECT GRID CELL PANEL
    # -------------------------------------------------------------------------
    if show_inspector and not fine_grid_df.empty:
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        
        # Grid cell selector dropdown
        grid_opts = [f"{row['lat']}°N, {row['lon']}°E ({row['value']} {var_unit})" for _, row in fine_grid_df.iloc[::8].iterrows()]
        sel_cell_str = st.selectbox("📍 Select Grid Cell to Inspect:", grid_opts, index=0, key="grid_cell_selector")
        
        selected_idx = grid_opts.index(sel_cell_str) * 8
        target_row = fine_grid_df.iloc[selected_idx]

        val_tgt = target_row["value"]
        val_nwp = target_row["nwp_value"]
        val_ens = target_row["ensemble_value"]
        val_ai = target_row["ai_value"]

        # Gate weight estimation
        tot_err = abs(val_nwp - val_tgt) + abs(val_ens - val_tgt) + abs(val_ai - val_tgt) + 1e-4
        w_nwp_c = round((1.0 - abs(val_nwp - val_tgt)/tot_err) * 50, 1)
        w_ens_c = round((1.0 - abs(val_ens - val_tgt)/tot_err) * 50, 1)
        w_ai_c = round(100.0 - w_nwp_c - w_ens_c, 1)

        render_html_panel(f"""
        <div class="glass-card" style="padding: 14px 18px; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 13px; font-weight: 700; color: #60a5fa;">
                    📍 Inspection Panel: {target_row['lat']}°N, {target_row['lon']}°E (Step: +{selected_lead_step}h)
                </div>
                <div style="font-size: 12px; font-weight: 700; color: #34d399;">
                    Forecast: {val_tgt} {var_unit}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 10px; text-align: center;">
                <div style="background: rgba(255,255,255,0.04); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 11px; color: #94a3b8;">GFS NWP</div>
                    <div style="font-size: 15px; font-weight: 700; color: #ef4444;">{val_nwp} {var_unit}</div>
                    <div style="font-size: 10px; color: #64748b;">Gate Wt: {w_nwp_c}%</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 11px; color: #94a3b8;">GEFS Ensemble</div>
                    <div style="font-size: 15px; font-weight: 700; color: #f59e0b;">{val_ens} {var_unit}</div>
                    <div style="font-size: 10px; color: #64748b;">Gate Wt: {w_ens_c}%</div>
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 11px; color: #94a3b8;">AI Neural Model</div>
                    <div style="font-size: 15px; font-weight: 700; color: #a855f7;">{val_ai} {var_unit}</div>
                    <div style="font-size: 10px; color: #64748b;">Gate Wt: {w_ai_c}%</div>
                </div>
            </div>
        </div>
        """)

with right_panel_col:
    # 1. REGION SUMMARY PANEL
    render_html_panel(f"""
    <div class="panel-header-title">📋 Region Summary</div>
    <div class="glass-card" style="margin-bottom: 14px;">
        <table class="summary-table">
            <tr>
                <td class="summary-label">Average Rainfall</td>
                <td class="summary-value">{summary['avg_value']} mm</td>
            </tr>
            <tr>
                <td class="summary-label">Maximum Rainfall</td>
                <td class="summary-value">{summary['max_value']} mm</td>
            </tr>
            <tr>
                <td class="summary-label">Minimum Rainfall</td>
                <td class="summary-value">{summary['min_value']} mm</td>
            </tr>
            <tr>
                <td class="summary-label">Heavy Rain Probability</td>
                <td class="summary-value" style="color: {summary['heavy_badge_color']};">{summary['heavy_rain_risk_pct']}%</td>
            </tr>
            <tr>
                <td class="summary-label">Forecast Confidence</td>
                <td class="summary-value" style="color: {summary['conf_badge_color']};">{summary['conf_badge']} ({summary['confidence_pct']}%)</td>
            </tr>
            <tr>
                <td class="summary-label">Expected Conditions</td>
                <td class="summary-value" style="font-size: 12px;">{summary['expected_conditions']}</td>
            </tr>
        </table>
        <div class="advisory-box">
            Advisory &nbsp;&nbsp; {summary['advisory']}
        </div>
    </div>
    """)

    # 2. AI MODEL WEIGHTS PANEL
    weights = summary.get("weights", {"gfs_nwp": 28.0, "gefs_ensemble": 42.0, "ai_model": 30.0})
    render_html_panel(f"""
    <div class="panel-header-title">⚛️ AI Model Weights</div>
    <div class="glass-card">
        <div class="weight-row">
            <div class="weight-header">
                <span style="color: #60a5fa;">GFS (NWP)</span>
                <span style="color: #ffffff;">{int(weights['gfs_nwp'])}%</span>
            </div>
            <div class="weight-bar-bg">
                <div class="weight-bar-fill" style="width: {weights['gfs_nwp']}%; background: {BLUE_ACCENT};"></div>
            </div>
        </div>
        <div class="weight-row">
            <div class="weight-header">
                <span style="color: #34d399;">GEFS (Ensemble)</span>
                <span style="color: #ffffff;">{int(weights['gefs_ensemble'])}%</span>
            </div>
            <div class="weight-bar-bg">
                <div class="weight-bar-fill" style="width: {weights['gefs_ensemble']}%; background: {GREEN_SUCCESS};"></div>
            </div>
        </div>
        <div class="weight-row">
            <div class="weight-header">
                <span style="color: #c084fc;">AI Model</span>
                <span style="color: #ffffff;">{int(weights['ai_model'])}%</span>
            </div>
            <div class="weight-bar-bg">
                <div class="weight-bar-fill" style="width: {weights['ai_model']}%; background: {PURPLE_ACCENT};"></div>
            </div>
        </div>
    </div>
    """)

st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BOTTOM ROW (24-HOUR FORECAST STRIP & MODEL COMPARISON CHART)
# -----------------------------------------------------------------------------
bot_col1, bot_col2 = st.columns([1.15, 1.0])

with bot_col1:
    hourly_slots = get_hourly_data()
    
    # Construct 24-Hour strip as ONE single HTML string without indentations
    slots_html_list = []
    for slot in hourly_slots:
        r_val = slot.get('rainfall_mm', 0)
        t_val = slot.get('temp_c', 26)
        slots_html_list.append(
            f'<div class="hourly-slot">'
            f'<div class="hourly-time">{slot["time"]}</div>'
            f'<div class="hourly-icon">{slot["icon"]}</div>'
            f'<div class="hourly-rain">{r_val} <span style="font-size: 10px; font-weight: normal;">mm</span></div>'
            f'<div class="hourly-temp">{int(t_val)}°C</div>'
            f'</div>'
        )
    slots_inner_html = "".join(slots_html_list)
    
    render_html_panel(f"""
    <div class="panel-header-title">⏱️ 24-Hour Forecast ({st.session_state['region']} Average)</div>
    <div class="glass-card">
        <div class="hourly-container">
            {slots_inner_html}
        </div>
    </div>
    """)

with bot_col2:
    render_html_panel(f"""
    <div class="panel-header-title">📊 Model Comparison ({st.session_state['region']} Average)</div>
    """)
    
    models_comp = get_model_comp_data()
    comp_df = pd.DataFrame(models_comp)
    
    colors_map = {
        "GFS": BLUE_PRIMARY,
        "GEFS": GREEN_SUCCESS,
        "AI Model": PURPLE_ACCENT,
        "Equal Mean": AMBER_WARN,
        "WeatherBlend (AI Blend)": RED_DANGER
    }
    
    bar_colors = []
    for _, row in comp_df.iterrows():
        m_name = row["model"]
        if row.get("is_weatherblend", False) or "WeatherBlend" in m_name:
            bar_colors.append(RED_DANGER)
        elif "GFS" in m_name:
            bar_colors.append(BLUE_ACCENT)
        elif "GEFS" in m_name:
            bar_colors.append(GREEN_SUCCESS)
        elif "AI" in m_name:
            bar_colors.append(PURPLE_ACCENT)
        else:
            bar_colors.append(AMBER_WARN)
    
    fig = go.Figure(data=[
        go.Bar(
            x=comp_df["model"],
            y=comp_df["value"],
            marker_color=bar_colors,
            marker_line_color="rgba(255,255,255,0.15)",
            marker_line_width=1,
            text=comp_df["value"].apply(lambda v: f"{v:.1f}"),
            textposition="auto",
            textfont=dict(color="#ffffff", size=11, family="Inter"),
            hovertemplate="<b>%{x}</b><br>Forecast: %{y} " + var_unit + "<extra></extra>"
        )
    ])
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.75)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=15, b=15),
        height=145,
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color=TEXT_MUTED)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color=TEXT_MUTED), title=dict(text=f"Rainfall ({var_unit})", font=dict(size=10, color=TEXT_MUTED))),
    )
    
    # Render inside container cleanly
    with st.container():
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
