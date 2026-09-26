"""
WeatherBlend AI - Shared UI Components & Helpers
Provides reusable headers, CSS styling, backend API callers, and color palettes across Streamlit pages.
"""

import os
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# CONSTANTS & PALETTE
# -----------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

BG_DARK = "#090d16"
CARD_BG = "rgba(15, 23, 42, 0.75)"
CARD_BORDER = "rgba(255, 255, 255, 0.08)"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"

BLUE_PRIMARY = "#2563eb"
BLUE_ACCENT = "#3b82f6"
TEAL_ACCENT = "#14b8a6"
PURPLE_ACCENT = "#8b5cf6"
AMBER_WARN = "#f59e0b"
GREEN_SUCCESS = "#10b981"
RED_DANGER = "#ef4444"

REGION_BOUNDS = {
    "Bihar": {"lat": 25.85, "lon": 85.75, "zoom": 7.6, "min_lat": 24.2, "max_lat": 27.5, "min_lon": 83.3, "max_lon": 88.3},
    "Jharkhand": {"lat": 23.6, "lon": 85.5, "zoom": 7.6, "min_lat": 21.8, "max_lat": 25.4, "min_lon": 83.3, "max_lon": 87.9},
    "West Bengal": {"lat": 24.3, "lon": 87.8, "zoom": 7.1, "min_lat": 21.5, "max_lat": 27.3, "min_lon": 85.8, "max_lon": 89.9},
    "Uttar Pradesh": {"lat": 26.9, "lon": 80.9, "zoom": 6.4, "min_lat": 23.8, "max_lat": 30.5, "min_lon": 77.1, "max_lon": 84.6},
    "Odisha": {"lat": 20.3, "lon": 84.5, "zoom": 7.1, "min_lat": 17.8, "max_lat": 22.6, "min_lon": 81.4, "max_lon": 87.6},
    "All India": {"lat": 22.5, "lon": 82.5, "zoom": 4.6, "min_lat": 8.0, "max_lat": 37.0, "min_lon": 68.0, "max_lon": 97.0}
}

CITY_LOCATIONS = {
    "Bihar": [
        {"name": "Patna (Capital)", "lat": 25.5941, "lon": 85.1376, "type": "Capital"},
        {"name": "Muzaffarpur", "lat": 26.1209, "lon": 85.3647, "type": "District HQ"},
        {"name": "Gaya", "lat": 24.7914, "lon": 85.0002, "type": "District HQ"},
        {"name": "Bhagalpur", "lat": 25.2425, "lon": 87.0135, "type": "District HQ"},
        {"name": "Purnia", "lat": 25.7771, "lon": 87.4753, "type": "District HQ"},
        {"name": "Darbhanga", "lat": 26.1542, "lon": 85.8918, "type": "District HQ"},
        {"name": "Begusarai", "lat": 25.4182, "lon": 86.1272, "type": "District HQ"},
        {"name": "Arrah", "lat": 25.5560, "lon": 84.6603, "type": "District HQ"},
        {"name": "Katihar", "lat": 25.5398, "lon": 87.5732, "type": "District HQ"},
        {"name": "Munger", "lat": 25.3748, "lon": 86.4735, "type": "District HQ"},
        {"name": "Motihari", "lat": 26.6469, "lon": 84.9089, "type": "District HQ"},
        {"name": "Chhapra", "lat": 25.7801, "lon": 84.7479, "type": "District HQ"},
        {"name": "Saharsa", "lat": 25.8833, "lon": 86.6000, "type": "District HQ"},
        {"name": "Kishanganj", "lat": 26.0827, "lon": 87.9492, "type": "District HQ"}
    ],
    "Jharkhand": [
        {"name": "Ranchi (Capital)", "lat": 23.3441, "lon": 85.3096, "type": "Capital"},
        {"name": "Jamshedpur", "lat": 22.8046, "lon": 86.2029, "type": "City"},
        {"name": "Dhanbad", "lat": 23.7957, "lon": 86.4304, "type": "City"},
        {"name": "Bokaro", "lat": 23.6693, "lon": 86.1511, "type": "City"},
        {"name": "Deoghar", "lat": 24.4826, "lon": 86.6997, "type": "City"},
        {"name": "Hazaribagh", "lat": 23.9961, "lon": 85.3637, "type": "City"},
        {"name": "Giridih", "lat": 24.1900, "lon": 86.3000, "type": "City"}
    ],
    "West Bengal": [
        {"name": "Kolkata (Capital)", "lat": 22.5726, "lon": 88.3639, "type": "Capital"},
        {"name": "Siliguri", "lat": 26.7271, "lon": 88.3953, "type": "City"},
        {"name": "Asansol", "lat": 23.6889, "lon": 86.9661, "type": "City"},
        {"name": "Durgapur", "lat": 23.5204, "lon": 87.3119, "type": "City"},
        {"name": "Malda", "lat": 25.0108, "lon": 88.1411, "type": "City"},
        {"name": "Kharagpur", "lat": 22.3460, "lon": 87.2320, "type": "City"},
        {"name": "Darjeeling", "lat": 27.0410, "lon": 88.2663, "type": "City"},
        {"name": "Berhampore", "lat": 24.0988, "lon": 88.2497, "type": "City"}
    ],
    "Uttar Pradesh": [
        {"name": "Lucknow (Capital)", "lat": 26.8467, "lon": 80.9462, "type": "Capital"},
        {"name": "Varanasi", "lat": 25.3176, "lon": 82.9739, "type": "City"},
        {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319, "type": "City"},
        {"name": "Prayagraj", "lat": 25.4358, "lon": 81.8463, "type": "City"},
        {"name": "Gorakhpur", "lat": 26.7606, "lon": 83.3732, "type": "City"},
        {"name": "Ayodhya", "lat": 26.7922, "lon": 82.1998, "type": "City"},
        {"name": "Agra", "lat": 27.1767, "lon": 78.0081, "type": "City"}
    ],
    "Odisha": [
        {"name": "Bhubaneswar (Capital)", "lat": 20.2961, "lon": 85.8245, "type": "Capital"},
        {"name": "Cuttack", "lat": 20.4625, "lon": 85.8828, "type": "City"},
        {"name": "Rourkela", "lat": 22.2604, "lon": 84.8536, "type": "City"},
        {"name": "Sambalpur", "lat": 21.4669, "lon": 83.9812, "type": "City"},
        {"name": "Puri", "lat": 19.8135, "lon": 85.8312, "type": "City"},
        {"name": "Balasore", "lat": 21.4942, "lon": 86.9317, "type": "City"}
    ],
    "All India": [
        {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090, "type": "Capital"},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "type": "City"},
        {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "type": "City"},
        {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "type": "City"},
        {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "type": "City"},
        {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "type": "City"},
        {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "type": "City"},
        {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "type": "City"}
    ]
}



def render_html_panel(html: str):
    """Renders custom HTML safely without markdown 4-space code parsing issues."""
    lines = [line.strip() for line in html.strip().splitlines()]
    clean_html = "\n".join(lines)
    st.markdown(clean_html, unsafe_allow_html=True)


def render_sidebar_navigation():
    """Renders multi-page navigation links safely in Streamlit sidebar."""
    if not hasattr(st, "page_link"):
        return
    try:
        main_page = "app.py"
        p1 = "pages/1_Model_Insights.py"
        p2 = "pages/2_Extreme_Event_Case_Study.py"
        p3 = "pages/3_Regional_Comparison.py"
        p4 = "pages/4_About.py"

        st.sidebar.markdown("<div style='font-size: 13px; font-weight: 700; color: #60a5fa; margin-bottom: 8px;'>🗺️ Multi-Page Navigation</div>", unsafe_allow_html=True)
        try:
            st.sidebar.page_link(main_page, label="Forecast Dashboard", icon="🌤️")
        except Exception:
            pass
        try:
            st.sidebar.page_link(p1, label="Model Insights", icon="🧠")
        except Exception:
            pass
        try:
            st.sidebar.page_link(p2, label="Extreme Event Case Study", icon="⚡")
        except Exception:
            pass
        try:
            st.sidebar.page_link(p3, label="Regional Comparison", icon="🗺️")
        except Exception:
            pass
        try:
            st.sidebar.page_link(p4, label="About & System Spec", icon="ℹ️")
        except Exception:
            pass
        st.sidebar.markdown("<div style='margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)
    except Exception:
        pass


def inject_global_css():
    """Injects core design tokens, glassmorphism, fonts, and custom navigation styling."""
    render_html_panel(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: {BG_DARK};
            color: {TEXT_MAIN};
        }}
        
        #MainMenu {{ visibility: hidden; }}
        header {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        
        .main {{
            background-color: {BG_DARK};
            padding: 0.2rem 1rem 1rem 1rem;
        }}
        
        /* --------------------------------------------------------------------- */
        /* HIGH-CONTRAST SIDEBAR NAVIGATION & LABELS                             */
        /* --------------------------------------------------------------------- */
        section[data-testid="stSidebar"] {{
            background-color: #0d1322 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
        }}

        /* Sidebar Navigation Links & Page Links */
        [data-testid="stSidebarNav"] {{
            padding-top: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 16px;
        }}
        [data-testid="stSidebarNav"] ul {{
            gap: 6px;
        }}
        [data-testid="stSidebarNav"] a, div[data-testid="stPageLink"] a {{
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            padding: 8px 14px !important;
            margin-bottom: 4px !important;
            transition: all 0.2s ease;
            text-decoration: none !important;
        }}
        [data-testid="stSidebarNav"] a:hover, div[data-testid="stPageLink"] a:hover {{
            background: rgba(37, 99, 235, 0.3) !important;
            border-color: rgba(59, 130, 246, 0.6) !important;
        }}
        [data-testid="stSidebarNav"] a span, div[data-testid="stPageLink"] a p, div[data-testid="stPageLink"] a span {{
            color: #ffffff !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            letter-spacing: 0.2px !important;
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.4), rgba(59, 130, 246, 0.2)) !important;
            border: 1px solid #3b82f6 !important;
            box-shadow: 0 0 12px rgba(59, 130, 246, 0.3) !important;
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] span {{
            color: #60a5fa !important;
            font-weight: 700 !important;
        }}

        /* High contrast for all Form Labels & Controls */
        label, .stRadio label, .stCheckbox label, .stSelectbox label, .stDateInput label, .stSlider label {{
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 13px !important;
        }}
        
        div[data-testid="stMarkdownContainer"] p {{
            color: #e2e8f0;
        }}
        
        .stRadio p, div[role="radiogroup"] label p {{
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 13px !important;
        }}
        
        div[data-testid="stCheckbox"] label p {{
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 13px !important;
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
        
        .page-title-box {{
            margin-bottom: 14px;
        }}
        .page-title-box h2 {{
            font-size: 24px;
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
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 13px;
        }}
        .summary-table tr:last-child td {{
            border-bottom: none;
        }}
        .summary-label {{
            color: #cbd5e1 !important;
            font-weight: 500;
        }}
        .summary-value {{
            color: #ffffff !important;
            font-weight: 700;
            text-align: right;
        }}
    </style>
    """)


def render_header_bar(region_name: str = "Bihar"):
    """Renders consistent top banner across all pages."""
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
                <span>📍</span> {region_name} Regional Focus
            </div>
        </div>
    </div>
    """)


def fetch_api(endpoint: str, payload: dict):
    """Helper to connect to FastAPI backend."""
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


def get_radar_color_rgba(val, v_min, v_max, alpha=200):
    """Linear 6-stop weather radar colormap."""
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
