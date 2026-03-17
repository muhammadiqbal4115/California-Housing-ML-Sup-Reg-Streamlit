import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏠",
    initial_sidebar_state="expanded",
    layout="wide",
)

# ─────────────────────────────────────────────
#  NEON GREEN THEME  +  HIDDEN EASTER EGG CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── GLOBAL ── */
:root {
    --neon:      #39ff14;
    --neon-dim:  #1aff1a44;
    --neon-glow: #39ff1466;
    --bg:        #050f05;
    --bg2:       #0a1a0a;
    --bg3:       #0f280f;
    --text:      #c8ffc8;
    --text-dim:  #6bbb6b;
    --border:    #1f4d1f;
    --accent:    #00ff88;
    --danger:    #ff4d4d;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: var(--bg) !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text) !important;
}

/* ── HIDE ALL STREAMLIT DEFAULT UI CHROME ── */
/* Hamburger menu / top-right toolbar */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
header[data-testid="stHeader"] { display: none !important; }

/* "Manage app" button + bottom-right badge */
[data-testid="manage-app-button"],
.stDeployButton,
#MainMenu,
footer,
footer a,
[data-testid="stBottom"],
[class*="viewerBadge"],
[class*="StatusWidget"],
iframe[title="streamlit_analytics"] { display: none !important; }

/* Remove top padding left by hidden header */
.block-container { padding-top: 1.5rem !important; }

/* Top-right toolbar (Deploy, Settings, Star, Github icons) */
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
.stToolbar,
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] [data-testid="baseButton-headerNoPadding"] {
    display: none !important;
    visibility: hidden !important;
}
/* Bottom-right "Manage app" button */
[data-testid="manage-app-button"],
.st-emotion-cache-1dp5vir,
.viewerBadge_container__r5tak,
.viewerBadge_link__qRIco,
#MainMenu,
footer,
footer *,
[data-testid="stFooter"],
/* Streamlit top-right action bar */
[data-testid="stActionButtonIcon"],
[class*="StatusWidget"],
[class*="ToolbarActions"],
[class*="viewerBadge"],
/* Running indicator dots */
[data-testid="stStatusWidget"],
.stStatusWidget { display: none !important; visibility: hidden !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--neon); border-radius: 3px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── MAIN CONTAINER ── */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1200px;
}

/* ── HEADER ── */
.neon-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--neon);
    text-shadow: 0 0 10px var(--neon), 0 0 30px var(--neon-glow),
                 0 0 60px var(--neon-glow);
    letter-spacing: 0.06em;
    margin-bottom: 0.2rem;
    animation: flicker 5s infinite;
    cursor: default;
    user-select: none;
}
.neon-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
@keyframes flicker {
    0%,19%,21%,23%,25%,54%,56%,100% {
        text-shadow: 0 0 10px var(--neon), 0 0 30px var(--neon-glow), 0 0 60px var(--neon-glow);
        opacity: 1;
    }
    20%,24%,55% { opacity: 0.75; text-shadow: none; }
}

/* ── FORM / CARD ── */
[data-testid="stForm"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    box-shadow: 0 0 20px var(--neon-dim), inset 0 0 40px #00000055;
    position: relative;
}
[data-testid="stForm"]::before {
    content: "";
    position: absolute;
    top: -1px; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon), transparent);
}

/* ── NUMBER INPUTS ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--neon) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 0.4rem 0.7rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--neon) !important;
    box-shadow: 0 0 8px var(--neon-glow) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMarkdownContainer"] p {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--neon) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stSelectbox"] svg { fill: var(--neon) !important; }

/* ── SUBMIT BUTTON ── */
[data-testid="stFormSubmitButton"] > button {
    background: transparent !important;
    border: 1px solid var(--neon) !important;
    border-radius: 4px !important;
    color: var(--neon) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1rem !important;
    letter-spacing: 0.15em !important;
    padding: 0.6rem 2.5rem !important;
    box-shadow: 0 0 10px var(--neon-dim);
    transition: all 0.2s !important;
    text-transform: uppercase;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--neon) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 20px var(--neon), 0 0 40px var(--neon-glow) !important;
    transform: translateY(-1px);
}

/* ── RESULT BOX ── */
[data-testid="stAlert"] {
    background: var(--bg3) !important;
    border: 1px solid var(--neon) !important;
    border-radius: 6px !important;
    color: var(--neon) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.2rem !important;
    box-shadow: 0 0 20px var(--neon-dim);
    animation: pulseIn 0.4s ease;
}
@keyframes pulseIn {
    from { opacity: 0; transform: scale(0.97); }
    to   { opacity: 1; transform: scale(1); }
}

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; }
[data-testid="stMetricValue"] {
    color: var(--neon) !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-shadow: 0 0 8px var(--neon-glow);
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── SCAN LINE OVERLAY ── */
body::after {
    content: "";
    pointer-events: none;
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 3px,
        rgba(0,0,0,0.05) 3px,
        rgba(0,0,0,0.05) 4px
    );
    z-index: 9999;
}

/* ── EASTER EGG: hidden pixel trigger ── */
#egg-trigger {
    position: fixed;
    bottom: 12px;
    right: 16px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #1a2e1a;
    border: 1px solid #1f4d1f;
    cursor: pointer;
    z-index: 10000;
    transition: background 0.3s, box-shadow 0.3s;
}
#egg-trigger:hover {
    background: var(--neon);
    box-shadow: 0 0 12px var(--neon);
}
#egg-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(5,15,5,0.93);
    z-index: 20000;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    font-family: 'Share Tech Mono', monospace;
    animation: fadeIn 0.4s ease;
}
#egg-overlay.show { display: flex; }
@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
.egg-box {
    border: 1px solid var(--neon);
    border-radius: 8px;
    padding: 3rem 4rem;
    text-align: center;
    box-shadow: 0 0 60px var(--neon-glow);
    max-width: 480px;
}
.egg-title {
    font-size: 2.5rem;
    color: var(--neon);
    text-shadow: 0 0 20px var(--neon);
    margin-bottom: 0.5rem;
    letter-spacing: 0.1em;
}
.egg-msg {
    color: var(--text-dim);
    font-size: 1rem;
    line-height: 1.8;
    margin-bottom: 1.5rem;
}
.egg-close {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border: 1px solid var(--neon);
    border-radius: 4px;
    color: var(--neon);
    cursor: pointer;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    transition: all 0.2s;
}
.egg-close:hover {
    background: var(--neon);
    color: var(--bg);
}
.egg-matrix {
    font-size: 0.7rem;
    color: var(--neon);
    opacity: 0.15;
    letter-spacing: 0.4em;
    margin-bottom: 1.5rem;
    white-space: nowrap;
    overflow: hidden;
    animation: scroll-matrix 6s linear infinite;
}
@keyframes scroll-matrix {
    from { transform: translateX(120%); }
    to   { transform: translateX(-120%); }
}
</style>

<!-- EASTER EGG HTML / JS -->
<div id="egg-trigger" title="..."></div>

<div id="egg-overlay">
  <div class="egg-box">
    <div class="egg-matrix">01001000 01001111 01010101 01010011 01000101</div>
    <div class="egg-title">🏠 ACCESS GRANTED</div>
    <div class="egg-msg">
      You found the hidden node.<br><br>
      <strong style="color:var(--neon)">FUN FACT:</strong><br>
      The California Housing dataset was collected<br>
      from the 1990 US Census. The median house value<br>
      was capped at $500,000 — pocket change today! 🤑<br><br>
      <span style="color:var(--accent)">Keep predicting. Keep building.</span>
    </div>
    <div class="egg-close" onclick="closeEgg()">[ CLOSE TERMINAL ]</div>
  </div>
</div>

<script>
  document.getElementById('egg-trigger').addEventListener('click', function() {
    document.getElementById('egg-overlay').classList.add('show');
  });
  function closeEgg() {
    document.getElementById('egg-overlay').classList.remove('show');
  }
  document.getElementById('egg-overlay').addEventListener('click', function(e) {
    if (e.target === this) closeEgg();
  });
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="neon-header">⟨ California Housing Price Prediction ⟩</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-sub">// RandomForest · 1990 Census · Median House Value Estimator</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR  — tips
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 SYSTEM INFO")
    st.markdown("""
**Model:** Random Forest Regressor  
**Features:** 9 input variables  
**Training split:** 80% stratified  
**Scaler:** StandardScaler  
**Encoding:** OneHotEncoder  
    """)
    st.divider()
    st.markdown("### 🔬 INPUT GUIDE")
    st.markdown("""
- **Longitude / Latitude** — block location  
- **Housing Median Age** — median age of block houses  
- **Total Rooms / Bedrooms** — block totals (not per house)  
- **Population** — people in the block  
- **Households** — total households in block  
- **Median Income** — in tens of thousands USD  
- **Ocean Proximity** — categorical location tag  
    """)
    st.divider()
    st.caption("Psst… explore the UI — there might be a hidden surprise 👀")

# ─────────────────────────────────────────────
#  FILE PATHS
# ─────────────────────────────────────────────
MODEL_FILE    = os.path.join("model.pkl")
PIPELINE_FILE = os.path.join("pipeline.pkl")
HOUSING       = os.path.join("housing.csv")

# ─────────────────────────────────────────────
#  FORM
# ─────────────────────────────────────────────
with st.container():
    with st.form("house_prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            longitude  = st.number_input("Longitude",        value=-121.46, min_value=-125.00, max_value=-110.00)
            latitude   = st.number_input("Latitude",         value=38.52,  min_value=30.00,   max_value=45.00)
        with col2:
            hma = st.number_input("Housing Median Age",  value=29.0,    min_value=1.00,    max_value=55.00, step=1.0)
            tr  = st.number_input("Total Rooms",         value=3873.0,  min_value=1.00,    max_value=40000.00, step=1.0)
        with col3:
            tb         = st.number_input("Total Bedrooms",   value=797.0,   min_value=3.00,    max_value=6500.00, step=1.0)
            population = st.number_input("Population",       value=2237.0,  min_value=2.00,    max_value=36000.00, step=1.0)
        with col4:
            households = st.number_input("Households",       value=706.0,   min_value=1.00,    max_value=6500.00, step=1.0)
            mi         = st.number_input("Median Income",    value=2.1736,  min_value=0.3,     max_value=16.00, step=1.0)

        op     = st.selectbox("Ocean Proximity", ("NEAR BAY", "INLAND", "<1H OCEAN", "NEAR OCEAN"), index=1)
        submit = st.form_submit_button("▶  PREDICT HOUSE VALUE")

# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
    return full_pipeline

if submit:
    input_data = pd.DataFrame([{
        "longitude":           longitude,
        "latitude":            latitude,
        "housing_median_age":  hma,
        "total_rooms":         tr,
        "total_bedrooms":      tb,
        "population":          population,
        "households":          households,
        "median_income":       mi,
        "ocean_proximity":     op,
    }])

    with st.spinner("⚙ Running model inference…"):
        if not os.path.exists(MODEL_FILE):
            st.info("🔧 First run — training model on housing data…")
            housing = pd.read_csv(HOUSING)
            housing["income_cat"] = pd.cut(
                housing["median_income"],
                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                labels=[1, 2, 3, 4, 5],
            )
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in split.split(housing, housing["income_cat"]):
                housing = housing.loc[train_index].drop("income_cat", axis=1)

            housing_labels   = housing["median_house_value"].copy()
            housing_features = housing.drop("median_house_value", axis=1)
            num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
            cat_attribs = ["ocean_proximity"]

            pipeline         = build_pipeline(num_attribs, cat_attribs)
            housing_prepared = pipeline.fit_transform(housing_features)

            model = RandomForestRegressor(random_state=42)
            model.fit(housing_prepared, housing_labels)

            joblib.dump(model,    MODEL_FILE)
            joblib.dump(pipeline, PIPELINE_FILE)
        else:
            model    = joblib.load(MODEL_FILE)
            pipeline = joblib.load(PIPELINE_FILE)

        transformed_input = pipeline.transform(input_data)
        prediction        = model.predict(transformed_input)[0]

    # ── Result display ──
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("📍 Location", f"{latitude:.2f}°N  {abs(longitude):.2f}°W")
    with res_col2:
        st.metric("🏘 Ocean Proximity", op)
    with res_col3:
        st.metric("💰 Median Income Band", f"${mi * 10_000:,.0f}")

    st.success(f"🏠  Predicted Median House Value  →  **${prediction:,.2f}**")

    # ── Derived metrics ──
    m1, m2, m3 = st.columns(3)
    rooms_per_household = tr / max(households, 1)
    beds_per_room       = tb / max(tr, 1)
    people_per_household = population / max(households, 1)
    with m1:
        st.metric("Rooms / Household",    f"{rooms_per_household:.1f}")
    with m2:
        st.metric("Beds / Room",          f"{beds_per_room:.2f}")
    with m3:
        st.metric("People / Household",   f"{people_per_household:.1f}")
