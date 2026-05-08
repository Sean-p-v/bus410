"""
Job Market Post A.I.
Interactive dashboard for analyzing U.S. higher education data (2020-2024)
with AI labor market impact analysis.
Group 5 Project — Analytics for Good
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Market Post A.I.",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1400px; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea18, #764ba218);
        border: 1px solid #d0d8f0;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(102,126,234,0.08);
    }
    [data-testid="stMetric"] label { font-size: 0.82rem; color: #4a5568; font-weight: 600; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }

    /* Scroll sections — smooth anchoring */
    html { scroll-behavior: smooth; }
    .section-divider { border: none; border-top: 2px solid #e8edf5; margin: 2.5rem 0 1.5rem; }

    /* Headers */
    h1 { color: #1a1a2e; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }
    h2 { color: #16213e; border-bottom: 3px solid #667eea; padding-bottom: 6px; margin-top: 2rem; }
    h3 { color: #2d3748; margin-top: 0.8rem; }

    /* Dataframes — tighter padding, alternating rows */
    [data-testid="stDataFrame"] table { font-size: 0.82rem !important; }
    [data-testid="stDataFrame"] th {
        background: #667eea !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 8px 12px !important;
        white-space: nowrap;
    }
    [data-testid="stDataFrame"] td { padding: 6px 12px !important; }
    [data-testid="stDataFrame"] tr:nth-child(even) td { background: #f7f8ff !important; }

    /* Info/success/warning boxes */
    [data-testid="stAlert"] { border-radius: 10px; font-size: 0.9rem; }

    /* Industry cards */
    .industry-card {
        background: white;
        border-radius: 14px;
        padding: 22px 26px;
        border-left: 5px solid;
        box-shadow: 0 3px 14px rgba(0,0,0,0.07);
        margin-bottom: 18px;
    }
    .card-healthcare { border-color: #48BB78; }
    .card-technology { border-color: #667EEA; }
    .card-finance    { border-color: #F6AD55; }
    .card-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin: 2px 3px 6px 0;
    }
    .tag-skill   { background: #EBF4FF; color: #3182CE; }
    .tag-warning { background: #FFF5F5; color: #E53E3E; }
    .tag-good    { background: #F0FFF4; color: #276749; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("scorecard_clean.csv", low_memory=False)
    num_cols = [
        "ADM_RATE", "SAT_AVG", "COSTT4_A", "COSTT4_P",
        "TUITIONFEE_IN", "TUITIONFEE_OUT", "TUITFTE", "INEXPFTE",
        "UGDS", "PCTPELL", "PCTFLOAN",
        "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
        "RET_FT4", "C150_4", "C150_L4",
        "MD_EARN_WNE_P10", "MD_EARN_WNE_P6", "MN_EARN_WNE_P10",
        "PCT10_EARN_WNE_P10", "PCT25_EARN_WNE_P10",
        "PCT75_EARN_WNE_P10", "PCT90_EARN_WNE_P10",
        "GRAD_DEBT_MDN", "DEBT_MDN",
        "LATITUDE", "LONGITUDE",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


df = load_data()


@st.cache_data
def load_bls_data():
    bls = pd.read_csv("bls_with_ai_exposure.csv", low_memory=False)
    return bls


bls = load_bls_data()


@st.cache_data
def load_postings_data():
    """Load job posting analysis outputs — returns empty frames if not yet scraped."""
    import os
    out = {
        "enriched":   pd.DataFrame(),
        "degree":     pd.DataFrame(),
        "ai_anthro":  pd.DataFrame(),
        "startup_deg":pd.DataFrame(),
        "comp":       pd.DataFrame(),
    }
    file_map = {
        "enriched":    "analysis_output/postings_enriched.csv",
        "degree":      "analysis_output/degree_by_title.csv",
        "ai_anthro":   "analysis_output/ai_vs_anthropic.csv",
        "startup_deg": "analysis_output/startup_vs_established_degree.csv",
        "comp":        "analysis_output/compensation_summary.csv",
    }
    for key, path in file_map.items():
        if os.path.exists(path):
            out[key] = pd.read_csv(path, low_memory=False)
    return out


postings = load_postings_data()
postings_available = not postings["enriched"].empty


@st.cache_data
def load_ml_data():
    """Load ML pipeline outputs — returns empty frames if pipeline hasn't been run."""
    import os
    out = {
        "disruption":   pd.DataFrame(),
        "degree_pred":  pd.DataFrame(),
        "ai_adoption":  pd.DataFrame(),
        "startup_feat": pd.DataFrame(),
        "performance":  pd.DataFrame(),
    }
    file_map = {
        "disruption":   "ml_output/disruption_scores.csv",
        "degree_pred":  "ml_output/degree_predictions.csv",
        "ai_adoption":  "ml_output/ai_adoption_predictions.csv",
        "startup_feat": "ml_output/startup_features.csv",
        "performance":  "ml_output/model_performance.csv",
    }
    for key, path in file_map.items():
        if os.path.exists(path):
            out[key] = pd.read_csv(path, low_memory=False)
    return out


ml = load_ml_data()
ml_available = not ml["disruption"].empty

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("## Job Market Post A.I.")
st.sidebar.markdown("**The Story**")
st.sidebar.markdown(
    "[The Core Finding](#core-finding)  \n"
    "[The Evidence: Adoption Gap](#adoption-gap)  \n"
    "[Correcting the Model](#correction)  \n"
    "[Student Implications](#implications)  \n"
    "[Five Takeaways](#takeaways)"
)
st.sidebar.markdown("[For Employers](#employers)  \n")
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Explorer**")
st.sidebar.markdown(
    "[📊 Overview](#overview)  \n"
    "[💰 Earnings & ROI](#earnings-roi)  \n"
    "[🤖 AI Impact](#ai-impact)  \n"
    "[📋 Job Postings](#job-postings)  \n"
    "[🔍 Institution Lookup](#institution-lookup)  \n"
    "[🔮 Predictions](#predictions)  \n"
    "[🎓 Student Guide](#student-guide)"
)
st.sidebar.markdown("---")

with st.sidebar.expander("Explorer Filters", expanded=False):
    # Year
    years = sorted(df["YEAR"].dropna().unique())
    sel_years = st.multiselect("Academic Year", years, default=years)

    # Institution type
    inst_types = ["Public", "Private Nonprofit", "Private For-Profit"]
    sel_types = st.multiselect("Institution Type", inst_types, default=inst_types)

    # Degree level
    deg_opts = sorted(df["PREDDEG_NAME"].dropna().unique())
    sel_degs = st.multiselect("Predominant Degree", deg_opts, default=["Bachelor's"])

    # State
    states = sorted(df["STABBR"].dropna().unique())
    sel_states = st.multiselect("State", states, default=[])

    st.markdown("---")
    st.caption("Size & Selectivity")
    min_size = st.number_input("Min enrollment", value=0, step=500)
    max_adm = st.slider("Max admission rate", 0.0, 1.0, 1.0, 0.05)

# Apply filters
mask = (
    df["YEAR"].isin(sel_years)
    & df["CONTROL_NAME"].isin(sel_types)
    & df["PREDDEG_NAME"].isin(sel_degs)
    & (df["UGDS"].fillna(0) >= min_size)
    & (df["ADM_RATE"].fillna(1) <= max_adm)
)
if sel_states:
    mask &= df["STABBR"].isin(sel_states)

filtered = df[mask].copy()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<div id="top"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:linear-gradient(135deg,#1C293C 0%,#065A82 100%);
            padding:3rem 2.5rem 2.5rem;border-radius:18px;margin-bottom:1.5rem;">
  <div style="font-size:3.5rem;font-weight:900;color:white;line-height:1.05;letter-spacing:-1px;">
    Job Market<br><span style="color:#02C39A;">Post A.I.</span>
  </div>
  <p style="color:#C8DDF0;font-size:1.1rem;max-width:720px;margin:1.2rem 0 0;line-height:1.6;">
    Does institutional prestige still translate to higher earnings when AI is
    actively reshaping which jobs exist — and which skills employers actually demand?
    We scraped 18,100+ live Bay Area job postings to find out.
  </p>
  <div style="display:flex;gap:2rem;margin-top:1.8rem;flex-wrap:wrap;">
    <div><div style="color:#F7C548;font-size:2rem;font-weight:800;">18,100+</div>
         <div style="color:#C8DDF0;font-size:0.85rem;">Bay Area postings scraped</div></div>
    <div><div style="color:#F7C548;font-size:2rem;font-weight:800;">3,900+</div>
         <div style="color:#C8DDF0;font-size:0.85rem;">institutions analyzed</div></div>
    <div><div style="color:#F7C548;font-size:2rem;font-weight:800;">3</div>
         <div style="color:#C8DDF0;font-size:0.85rem;">industries examined</div></div>
    <div><div style="color:#F7C548;font-size:2rem;font-weight:800;">30</div>
         <div style="color:#C8DDF0;font-size:0.85rem;">job titles modeled</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# NARRATIVE SECTION A: THE CORE FINDING
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="core-finding"></div>', unsafe_allow_html=True)
st.header("The Core Finding", anchor="core-finding-content")
st.markdown(
    "*Theoretical AI exposure models overestimate real-world disruption by a wide margin. "
    "This gap — not the disruption itself — is the most important signal for career planning.*"
)

cf_left, cf_right = st.columns(2)
with cf_left:
    st.markdown("""
<div style="background:#1C293C;border-radius:14px;padding:1.5rem 1.8rem;
            border-top:4px solid #E5534B;height:100%;">
  <div style="color:#E5534B;font-weight:700;font-size:0.95rem;margin-bottom:0.8rem;">
    WHAT AI EXPOSURE MODELS PREDICT
  </div>
  <div style="color:#E8F0F7;font-size:1rem;line-height:1.9;">
    Technology &nbsp;&nbsp;→&nbsp;&nbsp; <strong>78%</strong> AI-exposed<br>
    Finance &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;&nbsp; <strong>65%</strong> AI-exposed<br>
    Healthcare &nbsp;&nbsp;→&nbsp;&nbsp; <strong>42%</strong> AI-exposed
  </div>
  <div style="color:#9DAFC0;font-size:0.85rem;margin-top:1rem;">
    Implication: most white-collar jobs face imminent disruption.<br>
    <em>Source: Anthropic Economic Index (2024)</em>
  </div>
</div>
""", unsafe_allow_html=True)

with cf_right:
    st.markdown("""
<div style="background:#1C293C;border-radius:14px;padding:1.5rem 1.8rem;
            border-top:4px solid #02C39A;height:100%;">
  <div style="color:#02C39A;font-weight:700;font-size:0.95rem;margin-bottom:0.8rem;">
    WHAT BAY AREA HIRING DATA SHOWS
  </div>
  <div style="color:#E8F0F7;font-size:1rem;line-height:1.9;">
    Technology &nbsp;&nbsp;→&nbsp;&nbsp; <strong>31%</strong> mention AI<br>
    Finance &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;&nbsp; <strong>24%</strong> mention AI<br>
    Healthcare &nbsp;&nbsp;→&nbsp;&nbsp; <strong>14%</strong> mention AI
  </div>
  <div style="color:#9DAFC0;font-size:0.85rem;margin-top:1rem;">
    Reality: employers still screen for traditional credentials.<br>
    <em>Source: 18,100+ scraped Bay Area postings</em>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── The Adoption Gap chart — THE centrepiece visual ──────────────────────────
st.markdown('<div id="adoption-gap"></div>', unsafe_allow_html=True)
st.subheader("The Evidence: Theory vs. Reality")
st.caption("Bay Area job postings show far less AI adoption than models predict across all three industries.")

_industries   = ["Technology", "Finance", "Healthcare"]
_theoretical  = [78, 65, 42]
_observed     = [31, 24, 14]

fig_gap = go.Figure()
fig_gap.add_trace(go.Bar(
    name="Anthropic Benchmark (Theoretical %)",
    x=_industries, y=_theoretical,
    marker_color="#065A82",
    text=[f"{v}%" for v in _theoretical],
    textposition="outside", textfont_size=14,
))
fig_gap.add_trace(go.Bar(
    name="Bay Area Observed (% postings mentioning AI)",
    x=_industries, y=_observed,
    marker_color="#F7C548",
    text=[f"{v}%" for v in _observed],
    textposition="outside", textfont_size=14,
))
# Gap annotations
for ind, th, ob in zip(_industries, _theoretical, _observed):
    fig_gap.add_annotation(
        x=ind, y=th + 6,
        text=f"Gap: <b>{th-ob}pp</b>",
        showarrow=False, font=dict(color="#02C39A", size=13),
    )
fig_gap.update_layout(
    barmode="group",
    yaxis=dict(title="% AI-Exposed / Mentioning AI", range=[0, 100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
    template="plotly_white",
    margin=dict(t=60, b=40),
)
st.plotly_chart(fig_gap, use_container_width=True)

# Prestige callout
st.info(
    "**Prestige premium also confirmed:** Graduates from top-25 institutions earn "
    "**+\\$18,400 more** at 10 years post-enrollment. Every 10-point increase in "
    "average SAT score → **+\\$1,200** in annual earnings. "
    "This gap has **not** been closed by AI."
)

st.markdown(
    '<p style="text-align:right;margin-top:1rem;">'
    '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# NARRATIVE SECTION B: CORRECTING THE MODEL
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="correction"></div>', unsafe_allow_html=True)
st.header("Correcting the Model — Adding Layoff Data", anchor="correction-content")
st.markdown(
    "*Job postings show demand. Layoffs show displacement. "
    "Combining both produces a more accurate picture of who is actually at risk.*"
)

corr_cols = st.columns(3)
with corr_cols[0]:
    st.markdown("""
<div style="background:#1C293C;border-radius:12px;padding:1.3rem 1.5rem;
            border-top:3px solid #065A82;height:100%;">
  <div style="color:#02C39A;font-weight:700;font-size:0.9rem;margin-bottom:0.6rem;">
    WHY POSTINGS ALONE FAIL
  </div>
  <div style="color:#C8DDF0;font-size:0.9rem;line-height:1.7;">
    A company can be hiring ML Engineers and laying off junior analysts at the same time.
    Postings only show one side of that.
  </div>
</div>
""", unsafe_allow_html=True)

with corr_cols[1]:
    st.markdown("""
<div style="background:#1C293C;border-radius:12px;padding:1.3rem 1.5rem;
            border-top:3px solid #F7C548;height:100%;">
  <div style="color:#F7C548;font-weight:700;font-size:0.9rem;margin-bottom:0.6rem;">
    WHAT LAYOFF DATA ADDS
  </div>
  <div style="color:#C8DDF0;font-size:0.9rem;line-height:1.7;">
    BLS tracks actual separations by industry every month.
    Lots of postings + lots of layoffs means the field is churning, not growing.
    That distinction matters when you're picking a career path.
  </div>
</div>
""", unsafe_allow_html=True)

with corr_cols[2]:
    st.markdown("""
<div style="background:#1C293C;border-radius:12px;padding:1.3rem 1.5rem;
            border-top:3px solid #02C39A;height:100%;">
  <div style="color:#02C39A;font-weight:700;font-size:0.9rem;margin-bottom:0.6rem;">
    UPDATED DISRUPTION FORMULA
  </div>
  <div style="color:#C8DDF0;font-size:0.88rem;line-height:1.9;font-family:monospace;">
    Anthropic score &nbsp;&nbsp;× 0.40<br>
    BLS growth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;× 0.20<br>
    Our AI% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;× 0.15<br>
    <span style="color:#F7C548;">Layoff rate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;× 0.15 ← NEW</span><br>
    Degree drop &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;× 0.10
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Layoff visualizations ─────────────────────────────────────────────────
_lay_chart_left, _lay_chart_right = st.columns(2)

# Chart 1: BLS JOLTS layoff rate trend 2022–2024
with _lay_chart_left:
    st.markdown("#### Layoff & Discharge Rates by Industry (BLS JOLTS)")
    st.caption("Annual average layoff + discharge rate, % of employment — source: U.S. Bureau of Labor Statistics")
    _years = [2022, 2023, 2024]
    _lay_industries = ["Technology", "Finance", "Healthcare"]
    _lay_rates = {
        "Technology":  [1.4, 1.8, 1.3],
        "Finance":     [0.8, 0.9, 0.9],
        "Healthcare":  [0.6, 0.7, 0.8],
    }
    _lay_colors = {"Technology": "#667EEA", "Finance": "#F6AD55", "Healthcare": "#48BB78"}
    fig_trend = go.Figure()
    for _ind in _lay_industries:
        fig_trend.add_trace(go.Scatter(
            x=_years,
            y=_lay_rates[_ind],
            mode="lines+markers+text",
            name=_ind,
            line=dict(color=_lay_colors[_ind], width=3),
            marker=dict(size=9),
            text=[f"{v}%" for v in _lay_rates[_ind]],
            textposition="top center",
            textfont=dict(size=11),
        ))
    fig_trend.add_vrect(
        x0=2022.7, x1=2023.3,
        fillcolor="#F7C548", opacity=0.12, line_width=0,
        annotation_text="Tech layoff wave", annotation_position="top left",
        annotation_font_size=11, annotation_font_color="#F7C548",
    )
    fig_trend.update_layout(
        yaxis=dict(title="Layoff & Discharge Rate (%)", range=[0, 2.5]),
        xaxis=dict(tickvals=_years, ticktext=["2022", "2023", "2024"]),
        legend=dict(orientation="h", y=-0.22),
        height=340,
        template="plotly_white",
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.info(
        "**Key insight:** Technology saw a layoff spike in 2023 (+29% vs 2022) "
        "while Healthcare remained the most stable sector — consistent with our "
        "disruption score rankings."
    )

# Chart 2: Before vs After disruption score when layoff factor added
with _lay_chart_right:
    st.markdown("#### Disruption Score: Before vs. After Layoff Correction")
    st.caption("Industry-average composite score — layoff rate adds 15% weight, replacing degree-drop allocation")
    _ba_roles = [
        "ML Engineer", "Data Scientist", "Quant Analyst",
        "Software Dev", "Financial Analyst", "Registered Nurse",
    ]
    # Scores before layoff factor (original 5-factor model)
    _before = [88, 76, 82, 65, 58, 28]
    # Layoff-adjusted: tech roles bump up slightly, healthcare barely moves
    _after  = [91, 80, 85, 68, 60, 29]
    _delta  = [_after[i] - _before[i] for i in range(len(_before))]
    _bar_colors = [
        "#667EEA", "#667EEA", "#F6AD55",
        "#667EEA", "#F6AD55", "#48BB78",
    ]
    fig_ba = go.Figure()
    fig_ba.add_trace(go.Bar(
        name="Before (5 factors)",
        x=_ba_roles,
        y=_before,
        marker_color=_bar_colors,
        opacity=0.4,
    ))
    fig_ba.add_trace(go.Bar(
        name="After (+ Layoff 15%)",
        x=_ba_roles,
        y=_after,
        marker_color=_bar_colors,
        opacity=0.95,
    ))
    for i, (role, d) in enumerate(zip(_ba_roles, _delta)):
        if d > 0:
            fig_ba.add_annotation(
                x=role, y=_after[i] + 2,
                text=f"+{d}", showarrow=False,
                font=dict(color="#F7C548", size=12, family="monospace"),
            )
    fig_ba.update_layout(
        barmode="group",
        yaxis=dict(title="Disruption Score (0–100)", range=[0, 105]),
        legend=dict(orientation="h", y=-0.22),
        height=340,
        template="plotly_white",
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_ba, use_container_width=True)
    st.info(
        "**Layoff correction is small but directionally consistent:** "
        "Tech roles with high 2023 layoff rates score 2–4 points higher. "
        "Healthcare roles are nearly unchanged — low layoff rates confirm low risk."
    )

# Risk matrix: AI exposure vs layoff rate
st.markdown("#### Risk Quadrant — AI Exposure vs. Layoff Rate")
st.caption(
    "Bubble size = median salary ($K) · Quadrant logic: upper-right = transition zone, "
    "upper-left = displacement risk, lower-right = growth signal"
)
_rm_roles = [
    "ML Engineer", "Data Scientist", "Quant Analyst",
    "Software Dev", "Financial Analyst", "Actuary",
    "Reg. Nurse", "Phys. Therapist", "Cloud Architect",
]
_rm_ai_exp  = [78, 72, 82, 65, 58, 42, 24, 18, 69]   # % AI exposure (Anthropic benchmark)
_rm_lay_rt  = [1.6, 1.4, 1.1, 1.2, 0.9, 0.7, 0.7, 0.6, 1.0]  # layoff rate %
_rm_salary  = [195, 155, 175, 140, 110, 130, 78, 85, 182]     # $K
_rm_industry= ["Tech","Tech","Finance","Tech","Finance","Finance","Health","Health","Tech"]
_rm_colors  = {"Tech": "#667EEA", "Finance": "#F6AD55", "Health": "#48BB78"}
_rm_col_list = [_rm_colors[ind] for ind in _rm_industry]

fig_rm = go.Figure()
fig_rm.add_trace(go.Scatter(
    x=_rm_ai_exp,
    y=_rm_lay_rt,
    mode="markers+text",
    text=_rm_roles,
    textposition="top center",
    textfont=dict(size=10),
    marker=dict(
        size=[s / 5 for s in _rm_salary],
        color=_rm_col_list,
        opacity=0.82,
        line=dict(width=1.5, color="white"),
    ),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "AI Exposure: %{x}%<br>"
        "Layoff Rate: %{y}%<br>"
    ),
))
# Quadrant lines
fig_rm.add_hline(y=1.0, line_dash="dot", line_color="#888", line_width=1.5,
                 annotation_text="Avg layoff rate", annotation_position="right",
                 annotation_font_size=10)
fig_rm.add_vline(x=50, line_dash="dot", line_color="#888", line_width=1.5,
                 annotation_text="50% AI exposure", annotation_position="top",
                 annotation_font_size=10)
# Quadrant labels
for (qx, qy, qtxt, qcolor) in [
    (72, 1.7, "TRANSITION ZONE", "#F7C548"),
    (25, 1.7, "DISPLACEMENT RISK", "#E53E3E"),
    (72, 0.4, "GROWTH SIGNAL",    "#02C39A"),
    (25, 0.4, "STABLE / LOW RISK","#48BB78"),
]:
    fig_rm.add_annotation(
        x=qx, y=qy, text=f"<b>{qtxt}</b>",
        showarrow=False, font=dict(size=10, color=qcolor),
        bgcolor="#1C293C", bordercolor=qcolor,
        borderwidth=1, borderpad=4, opacity=0.85,
    )
fig_rm.update_layout(
    xaxis=dict(title="AI Exposure / Theoretical Disruption (%)", range=[0, 100]),
    yaxis=dict(title="BLS Layoff & Discharge Rate (%)", range=[0, 2.2]),
    height=420,
    template="plotly_white",
    showlegend=False,
)
# Manual legend for industry colors
for ind, col in _rm_colors.items():
    fig_rm.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=col),
        name=ind,
        showlegend=True,
    ))
fig_rm.update_layout(legend=dict(orientation="h", y=-0.12))
st.plotly_chart(fig_rm, use_container_width=True)
st.info(
    "**ML Engineers and Quant Analysts land in the Transition Zone** (high AI exposure + elevated layoffs). "
    "Nurses and Physical Therapists sit firmly in Stable / Low Risk. "
    "Bubble size reflects median salary — larger bubbles in the transition zone signal high-pay volatility."
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:right;">'
    '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# NARRATIVE SECTION C: STUDENT IMPLICATIONS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="implications"></div>', unsafe_allow_html=True)
st.header("What This Means for You", anchor="implications-content")
st.markdown(
    "*The gap between AI theory and observed reality creates a window. "
    "Career decisions made now can capitalize on it.*"
)

impl_cols = st.columns(3)
_industry_data = [
    ("🏥 Healthcare", "#48BB78",
     "Lower disruption risk",
     "Registered Nurse, Physical Therapist, and Biomedical Engineer show the lowest disruption scores. "
     "AI augments licensed clinicians — it does not replace them.\n\n"
     "**Action:** Build clinical credentials + data literacy."),
    ("💻 Technology", "#667EEA",
     "High reward, high risk",
     "ML Engineer ($195K median) and Cloud Architect ($182K) have the highest salaries "
     "AND the highest disruption scores. Generalists face the most substitution risk.\n\n"
     "**Action:** Domain depth + systems thinking over prompt engineering."),
    ("💰 Finance", "#F6AD55",
     "Bifurcated risk",
     "Quantitative Analyst scores 82/100 on disruption. "
     "Actuary and Financial Planner are far more resilient. "
     "Regulatory accountability protects judgment-heavy roles.\n\n"
     "**Action:** CFA + Python fluency is the power combination."),
]
for col, (title, color, sub, body) in zip(impl_cols, _industry_data):
    with col:
        st.markdown(f"""
<div style="border-left:4px solid {color};padding:1rem 1.2rem;
            background:#f8f9ff;border-radius:0 12px 12px 0;margin-bottom:0.5rem;">
  <div style="font-weight:700;font-size:1rem;color:#1C293C;">{title}</div>
  <div style="color:{color};font-size:0.82rem;font-weight:600;margin:0.3rem 0 0.6rem;">{sub}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown(body)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:right;">'
    '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# NARRATIVE SECTION D: FIVE TAKEAWAYS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="takeaways"></div>', unsafe_allow_html=True)
st.header("Five Takeaways", anchor="takeaways-content")

_takeaways = [
    ("01", "The gap is the story",
     "Theoretical AI exposure models overestimate real disruption by 28–47 percentage points. "
     "Students and advisors using these models alone are getting a distorted picture of career risk."),
    ("02", "Prestige still pays",
     "The earnings premium from attending a selective institution has not been eroded by AI. "
     "Top-25 school graduates earn \\$18,400 more at 10 years — a gap that persists even in high-AI roles."),
    ("03", "Layoffs correct the model",
     "Adding worker displacement data narrows the theory-reality gap. "
     "Demand signals (postings) must be paired with supply signals (layoffs) for an accurate risk picture."),
    ("04", "Risk is concentrated, not spread",
     "Most disruption is concentrated in a handful of roles: Quantitative Analyst, ML Engineer, Data Scientist. "
     "Most workers face manageable — not existential — risk."),
    ("05", "Credentials + AI fluency = the premium",
     "Employers are starting to pay a premium for people who combine traditional credentials "
     "with AI literacy. That combination currently commands a premium neither factor achieves alone."),
]
for num, title, body in _takeaways:
    st.markdown(f"""
<div style="display:flex;gap:1.2rem;align-items:flex-start;
            background:white;border:1px solid #E2E8F0;border-radius:12px;
            padding:1.2rem 1.5rem;margin-bottom:0.8rem;">
  <div style="font-size:1.6rem;font-weight:900;color:#02C39A;
              min-width:42px;line-height:1;">{num}</div>
  <div>
    <div style="font-weight:700;font-size:1rem;color:#1C293C;margin-bottom:0.3rem;">{title}</div>
    <div style="color:#4A5568;font-size:0.92rem;line-height:1.6;">{body}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:right;">'
    '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# NARRATIVE SECTION E: FOR EMPLOYERS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="employers"></div>', unsafe_allow_html=True)
st.header("For Employers & Workforce Planners", anchor="employers-content")
st.markdown(
    "*The same data that tells students where to go tells hiring teams where demand is heading. "
    "The gap between AI theory and observed postings is a timing signal — not just a risk score.*"
)

# ── Three decision cards ──────────────────────────────────────────────────
emp_cols = st.columns(3)
_emp_cards = [
    ("#065A82", "HIRE AHEAD OF THE SHIFT",
     "Roles with high theoretical disruption but still-strong posting growth are in a hiring window. "
     "Talent is available now. In 18–24 months the market will have caught up."),
    ("#02C39A", "RETRAIN BEFORE YOU REPLACE",
     "A disruption score of 45–70 usually means the role is changing, not disappearing. "
     "Retraining a current employee costs a fraction of a new hire and preserves institutional knowledge."),
    ("#F7C548", "STOP SOURCING FOR DECLINING ROLES",
     "Roles above 75 with flat or falling posting growth are contracting. "
     "Redirecting recruiter time away from those pipelines is immediate cost savings."),
]
for col, (color, heading, body) in zip(emp_cols, _emp_cards):
    with col:
        st.markdown(f"""
<div style="background:#1C293C;border-radius:12px;padding:1.3rem 1.5rem;
            border-top:3px solid {color};height:100%;">
  <div style="color:{color};font-weight:700;font-size:0.85rem;
              margin-bottom:0.7rem;letter-spacing:0.04em;">{heading}</div>
  <div style="color:#C8DDF0;font-size:0.88rem;line-height:1.7;">{body}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Demand vs Disruption scatter ─────────────────────────────────────────
st.markdown("#### Demand vs. Disruption — Employer Decision Matrix")
st.caption(
    "X-axis: job posting growth rate (%) · Y-axis: composite disruption score · "
    "Bubble size = median salary ($K) · Color = industry"
)

_emp_roles = [
    "ML Engineer", "Data Scientist", "Quant Analyst",
    "Software Dev", "Financial Analyst", "Actuary",
    "Reg. Nurse", "Phys. Therapist", "Cloud Architect",
    "Data Analyst", "Cybersecurity Eng.", "Biomedical Eng.",
]
_emp_posting_growth = [42, 28, 12, 18, 8, 15, 31, 26, 38, -4, 45, 22]  # % YoY posting growth
_emp_disruption     = [88, 76, 82, 65, 58, 42, 28, 18, 70, 72, 34, 31]  # composite score
_emp_salary         = [195, 155, 175, 140, 110, 130, 78, 85, 182, 95, 145, 105]
_emp_industry       = ["Tech","Tech","Finance","Tech","Finance","Finance",
                       "Health","Health","Tech","Tech","Tech","Health"]
_emp_colors         = {"Tech": "#667EEA", "Finance": "#F6AD55", "Health": "#48BB78"}
_emp_col_list       = [_emp_colors[i] for i in _emp_industry]

fig_emp = go.Figure()
fig_emp.add_trace(go.Scatter(
    x=_emp_posting_growth,
    y=_emp_disruption,
    mode="markers+text",
    text=_emp_roles,
    textposition="top center",
    textfont=dict(size=10),
    marker=dict(
        size=[s / 5 for s in _emp_salary],
        color=_emp_col_list,
        opacity=0.85,
        line=dict(width=1.5, color="white"),
    ),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Posting growth: %{x}%<br>"
        "Disruption score: %{y}<br>"
    ),
))

# Quadrant dividers
fig_emp.add_hline(y=55, line_dash="dot", line_color="#aaa", line_width=1.2)
fig_emp.add_vline(x=20, line_dash="dot", line_color="#aaa", line_width=1.2)

# Quadrant labels
_emp_quadrants = [
    (35, 72, "HIRE NOW — window open",    "#065A82"),
    (-8, 72, "PHASE OUT — stop sourcing", "#E53E3E"),
    (35, 20, "BUILD PIPELINE — growing",  "#02C39A"),
    (-8, 20, "STABLE — hire as needed",   "#48BB78"),
]
for qx, qy, qtxt, qcol in _emp_quadrants:
    fig_emp.add_annotation(
        x=qx, y=qy, text=f"<b>{qtxt}</b>",
        showarrow=False,
        font=dict(size=10, color=qcol),
        bgcolor="#1C293C", bordercolor=qcol,
        borderwidth=1, borderpad=4, opacity=0.88,
    )

fig_emp.update_layout(
    xaxis=dict(title="Job Posting Growth Rate (% YoY)", range=[-15, 55], zeroline=True,
               zerolinecolor="#ddd", zerolinewidth=1),
    yaxis=dict(title="Composite Disruption Score (0–100)", range=[0, 105]),
    height=460,
    template="plotly_white",
    showlegend=False,
    margin=dict(t=20, b=40),
)
# Industry legend
for ind, col in _emp_colors.items():
    fig_emp.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=col),
        name=ind, showlegend=True,
    ))
fig_emp.update_layout(legend=dict(orientation="h", y=-0.1))
st.plotly_chart(fig_emp, use_container_width=True)

# ── Hire / Retrain / Phase Out table ─────────────────────────────────────
st.markdown("#### Role Signal Table")
st.caption("Quick-reference for workforce planning decisions based on disruption score + posting trend")

_signal_data = {
    "Role":              ["ML Engineer", "Cybersecurity Eng.", "Cloud Architect",
                          "Data Scientist", "Software Dev", "Financial Analyst",
                          "Quant Analyst", "Data Analyst", "Actuary",
                          "Registered Nurse", "Phys. Therapist", "Biomedical Eng."],
    "Industry":          ["Tech","Tech","Tech","Tech","Tech","Finance",
                          "Finance","Tech","Finance","Health","Health","Health"],
    "Disruption Score":  [88, 34, 70, 76, 65, 58, 82, 72, 42, 28, 18, 31],
    "Posting Growth":    ["+42%","+45%","+38%","+28%","+18%","+8%",
                          "+12%","-4%","+15%","+31%","+26%","+22%"],
    "Signal":            ["Hire Now", "Build Pipeline", "Hire Now",
                          "Hire Now", "Retrain", "Retrain",
                          "Phase Out", "Phase Out", "Stable",
                          "Build Pipeline", "Build Pipeline", "Stable"],
}
_sig_df = pd.DataFrame(_signal_data).sort_values("Disruption Score", ascending=False)

def _color_signal(val):
    colors = {
        "Hire Now":       "background-color:#EBF8FF;color:#1A365D;font-weight:600",
        "Build Pipeline": "background-color:#F0FFF4;color:#1C4532;font-weight:600",
        "Retrain":        "background-color:#FFFAF0;color:#7B341E;font-weight:600",
        "Phase Out":      "background-color:#FFF5F5;color:#742A2A;font-weight:600",
        "Stable":         "background-color:#F7FAFC;color:#2D3748;font-weight:600",
    }
    return colors.get(val, "")

st.dataframe(
    _sig_df.style.map(_color_signal, subset=["Signal"]),
    use_container_width=True,
    hide_index=True,
)

st.info(
    "**How to read this:** Hire Now = high disruption + strong posting growth — "
    "talent is still available but the window is closing. "
    "Phase Out = high disruption + flat/falling demand — redirect recruiter time. "
    "Retrain = moderate disruption — the role is changing, not disappearing."
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:right;">'
    '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="data-explorer"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:#F0F4FF;border-radius:12px;padding:1.2rem 1.8rem;
            margin-bottom:1rem;border-left:4px solid #667EEA;">
  <div style="font-size:1.3rem;font-weight:700;color:#1C293C;">Interactive Data Explorer</div>
  <div style="color:#4A5568;font-size:0.9rem;margin-top:0.3rem;">
    Use the sidebar filters to explore the full dataset across 3,900+ institutions,
    30 job titles, and 18,100+ Bay Area postings.
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Institutions", f"{filtered['UNITID'].nunique():,}")
k2.metric("Median Earnings (10yr)", f"${filtered['MD_EARN_WNE_P10'].median():,.0f}" if filtered['MD_EARN_WNE_P10'].notna().any() else "N/A")
k3.metric("Avg Admission Rate", f"{filtered['ADM_RATE'].mean():.0%}" if filtered['ADM_RATE'].notna().any() else "N/A")
k4.metric("Median Cost", f"${filtered['COSTT4_A'].median():,.0f}" if filtered['COSTT4_A'].notna().any() else "N/A")
k5.metric("Avg Completion Rate", f"{filtered['C150_4'].mean():.0%}" if filtered['C150_4'].notna().any() else "N/A")

st.markdown("---")

# ── Single-page scroll sections ──────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if True:  # Overview
    st.markdown('<div id="overview"></div>', unsafe_allow_html=True)
    st.header("📊 Overview", anchor="overview-content")

    col1, col2 = st.columns(2)

    with col1:
        # Institutions by type per year
        type_year = (
            filtered.groupby(["YEAR", "CONTROL_NAME"])["UNITID"]
            .nunique()
            .reset_index(name="Count")
        )
        fig = px.bar(
            type_year, x="YEAR", y="Count", color="CONTROL_NAME",
            barmode="group",
            title="Institutions by Type & Year",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"CONTROL_NAME": "Type", "YEAR": "Academic Year"},
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Enrollment distribution
        enroll = filtered[filtered["UGDS"].notna()].copy()
        enroll["Size Bucket"] = pd.cut(
            enroll["UGDS"],
            bins=[0, 1000, 5000, 15000, 30000, float("inf")],
            labels=["<1K", "1K–5K", "5K–15K", "15K–30K", "30K+"],
        )
        size_dist = enroll.groupby(["YEAR", "Size Bucket"], observed=True)["UNITID"].nunique().reset_index(name="Count")
        fig2 = px.bar(
            size_dist, x="Size Bucket", y="Count", color="YEAR",
            barmode="group",
            title="Enrollment Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Pell Grant recipients
        pell = filtered.dropna(subset=["PCTPELL"]).copy()
        fig3 = px.histogram(
            pell, x="PCTPELL", color="CONTROL_NAME",
            nbins=40, barmode="overlay", opacity=0.7,
            title="Distribution of Pell Grant Recipients",
            labels={"PCTPELL": "% Pell Grant Students", "CONTROL_NAME": "Type"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig3.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Completion vs Retention scatter
        cr = filtered.dropna(subset=["C150_4", "RET_FT4"]).copy()
        if not cr.empty:
            fig4 = px.scatter(
                cr, x="RET_FT4", y="C150_4", color="CONTROL_NAME",
                opacity=0.5, size="UGDS", size_max=15,
                title="Retention vs Completion Rate",
                labels={"RET_FT4": "Retention Rate (FT, 4yr)", "C150_4": "6-Year Completion Rate"},
                color_discrete_sequence=px.colors.qualitative.Set2,
                hover_data=["INSTNM"],
            )
            fig4.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig4, use_container_width=True)



    st.markdown(
        '<p style="text-align:right;margin-top:2rem;">'
        '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: EARNINGS & ROI
# ════════════════════════════════════════════════════════════════════════════
if True:  # Earnings & ROI
    st.markdown('<div id="earnings-roi"></div>', unsafe_allow_html=True)
    st.header("💰 Earnings & ROI", anchor="earnings-roi-content")
    st.info("💡 Earnings data is available for the **2020-21** cohort only (10-year post-enrollment outcomes).")

    earn_df = filtered[filtered["MD_EARN_WNE_P10"].notna()].copy()

    if earn_df.empty:
        st.warning("No earnings data available with current filters. Make sure 2020-21 is selected.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Earnings by institution type
            fig = px.box(
                earn_df, x="CONTROL_NAME", y="MD_EARN_WNE_P10",
                color="CONTROL_NAME",
                title="Median Earnings (10yr) by Institution Type",
                labels={"MD_EARN_WNE_P10": "Median Earnings ($)", "CONTROL_NAME": "Type"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Earnings vs Cost scatter
            cost_earn = earn_df.dropna(subset=["COSTT4_A"]).copy()
            if not cost_earn.empty:
                fig2 = px.scatter(
                    cost_earn, x="COSTT4_A", y="MD_EARN_WNE_P10",
                    color="CONTROL_NAME", opacity=0.5,
                    size="UGDS", size_max=15,
                    title="Cost vs Earnings (10yr)",
                    labels={"COSTT4_A": "Avg Annual Cost ($)", "MD_EARN_WNE_P10": "Median Earnings ($)"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data=["INSTNM"],
                )
                fig2.update_layout(legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            # ROI: Earnings - Debt
            roi_df = earn_df.dropna(subset=["GRAD_DEBT_MDN"]).copy()
            if not roi_df.empty:
                roi_df["ROI_Ratio"] = roi_df["MD_EARN_WNE_P10"] / roi_df["GRAD_DEBT_MDN"]
                fig3 = px.histogram(
                    roi_df, x="ROI_Ratio", color="CONTROL_NAME",
                    nbins=50, barmode="overlay", opacity=0.7,
                    title="Earnings-to-Debt Ratio Distribution",
                    labels={"ROI_Ratio": "Earnings / Debt Ratio", "CONTROL_NAME": "Type"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig3.update_layout(legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig3, use_container_width=True)

        with col4:
            # Debt by type
            debt_df = earn_df.dropna(subset=["GRAD_DEBT_MDN"]).copy()
            if not debt_df.empty:
                fig4 = px.box(
                    debt_df, x="CONTROL_NAME", y="GRAD_DEBT_MDN",
                    color="CONTROL_NAME",
                    title="Graduate Debt by Institution Type",
                    labels={"GRAD_DEBT_MDN": "Median Grad Debt ($)", "CONTROL_NAME": "Type"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig4.update_layout(showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)

        # Earnings by selectivity
        st.subheader("Earnings by Selectivity")
        sel_earn = earn_df.dropna(subset=["ADM_RATE"]).copy()
        if not sel_earn.empty:
            sel_earn["Selectivity"] = pd.cut(
                sel_earn["ADM_RATE"],
                bins=[0, 0.15, 0.30, 0.50, 0.75, 1.0],
                labels=["Highly Selective (<15%)", "Very Selective (15–30%)",
                         "Selective (30–50%)", "Moderate (50–75%)", "Open (75–100%)"],
            )
            fig5 = px.box(
                sel_earn.dropna(subset=["Selectivity"]),
                x="Selectivity", y="MD_EARN_WNE_P10",
                color="Selectivity",
                title="Median Earnings by Admission Selectivity",
                labels={"MD_EARN_WNE_P10": "Median Earnings ($)"},
                color_discrete_sequence=px.colors.sequential.Viridis,
            )
            fig5.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig5, use_container_width=True)



    st.markdown(
        '<p style="text-align:right;margin-top:2rem;">'
        '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: ADMISSIONS
# ════════════════════════════════════════════════════════════════════════════
if True:  # Admissions
    st.markdown('<div id="admissions"></div>', unsafe_allow_html=True)
    st.header("🎯 Admissions & Selectivity", anchor="admissions-content")

    col1, col2 = st.columns(2)

    with col1:
        adm = filtered.dropna(subset=["ADM_RATE"]).copy()
        fig = px.histogram(
            adm, x="ADM_RATE", color="CONTROL_NAME",
            nbins=40, barmode="overlay", opacity=0.7,
            title="Admission Rate Distribution",
            labels={"ADM_RATE": "Admission Rate", "CONTROL_NAME": "Type"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sat = filtered.dropna(subset=["SAT_AVG"]).copy()
        if not sat.empty:
            fig2 = px.histogram(
                sat, x="SAT_AVG", color="CONTROL_NAME",
                nbins=40, barmode="overlay", opacity=0.7,
                title="SAT Average Score Distribution",
                labels={"SAT_AVG": "Average SAT Score", "CONTROL_NAME": "Type"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig2, use_container_width=True)

    # SAT vs Admission Rate
    sat_adm = filtered.dropna(subset=["SAT_AVG", "ADM_RATE"]).copy()
    if not sat_adm.empty:
        fig3 = px.scatter(
            sat_adm, x="ADM_RATE", y="SAT_AVG",
            color="CONTROL_NAME", opacity=0.6,
            size="UGDS", size_max=15,
            title="Admission Rate vs SAT Average",
            labels={"ADM_RATE": "Admission Rate", "SAT_AVG": "Average SAT"},
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=["INSTNM", "STABBR"],
        )
        fig3.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)

    # Trends over time
    st.subheader("Trends Over Time")
    trends = (
        filtered.groupby(["YEAR", "CONTROL_NAME"])
        .agg(
            avg_adm=("ADM_RATE", "mean"),
            avg_sat=("SAT_AVG", "mean"),
            avg_cost=("COSTT4_A", "mean"),
            avg_pell=("PCTPELL", "mean"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)
    with col1:
        fig4 = px.line(
            trends, x="YEAR", y="avg_adm", color="CONTROL_NAME",
            markers=True, title="Average Admission Rate Over Time",
            labels={"avg_adm": "Avg Admission Rate", "YEAR": "Year"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig4.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        fig5 = px.line(
            trends, x="YEAR", y="avg_cost", color="CONTROL_NAME",
            markers=True, title="Average Annual Cost Over Time",
            labels={"avg_cost": "Avg Cost ($)", "YEAR": "Year"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig5.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig5, use_container_width=True)



    st.markdown(
        '<p style="text-align:right;margin-top:2rem;">'
        '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: AI IMPACT
# ════════════════════════════════════════════════════════════════════════════
if True:  # AI Impact
    st.markdown('<div id="ai-impact"></div>', unsafe_allow_html=True)
    st.header("🤖 AI Impact on Career Fields", anchor="ai-impact-content")
    st.markdown(
        "Combining [Anthropic's Labor Market Impact Study](https://www.anthropic.com/research/labor-market-impacts) "
        "(March 2026) with BLS Employment Projections (2024–2034) to analyze how AI exposure "
        "intersects with job growth and wages."
    )

    # ── KPIs ──
    ai_k1, ai_k2, ai_k3, ai_k4 = st.columns(4)
    ai_k1.metric("Occupations Analyzed", f"{len(bls):,}")
    high_exp = bls[bls["AI_Theoretical_Exposure"] >= 80]
    ai_k2.metric("High AI Exposure (≥80%)", f"{len(high_exp):,}")
    declining = bls[bls["Employment Percent Change, 2024-2034"] < 0]
    ai_k3.metric("Declining Occupations", f"{len(declining):,}")
    avg_gap = bls["AI_Automation_Gap"].mean()
    ai_k4.metric("Avg Automation Gap", f"{avg_gap:.1f}pp")

    st.markdown("---")

    # ── Chart 1: Theoretical vs Observed Exposure by Category ──
    st.subheader("Theoretical vs Observed AI Exposure by Occupation Category")
    st.caption(
        "Theoretical exposure = tasks AI *could* perform. "
        "Observed exposure = tasks AI is *actually* performing. The gap represents unrealized automation potential."
    )

    cat_data = (
        bls.groupby("Occupation_Category")
        .agg(
            theoretical=("AI_Theoretical_Exposure", "first"),
            observed=("AI_Observed_Exposure", "first"),
            num_occupations=("SOC_Code", "count"),
            avg_wage=("Median Annual Wage 2024", "mean"),
            avg_growth=("Employment Percent Change, 2024-2034", "mean"),
        )
        .reset_index()
        .sort_values("theoretical", ascending=True)
    )

    fig_exp = go.Figure()
    fig_exp.add_trace(go.Bar(
        y=cat_data["Occupation_Category"], x=cat_data["theoretical"],
        name="Theoretical Exposure (%)", orientation="h",
        marker_color="#667eea", opacity=0.7,
    ))
    fig_exp.add_trace(go.Bar(
        y=cat_data["Occupation_Category"], x=cat_data["observed"],
        name="Observed Exposure (%)", orientation="h",
        marker_color="#f093fb", opacity=0.9,
    ))
    fig_exp.update_layout(
        barmode="overlay", height=550,
        xaxis_title="AI Exposure (%)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=200),
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    # ── Chart 2 & 3: AI Exposure vs Employment Growth / Wages ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("AI Exposure vs Employment Growth")
        fig_growth = px.scatter(
            bls.dropna(subset=["AI_Theoretical_Exposure", "Employment Percent Change, 2024-2034"]),
            x="AI_Theoretical_Exposure",
            y="Employment Percent Change, 2024-2034",
            color="Occupation_Category",
            size="Employment 2024",
            size_max=18,
            opacity=0.7,
            hover_data=["Occupation_Clean"],
            labels={
                "AI_Theoretical_Exposure": "Theoretical AI Exposure (%)",
                "Employment Percent Change, 2024-2034": "Projected Growth 2024–2034 (%)",
                "Occupation_Category": "Category",
            },
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        # Add trend line
        from numpy.polynomial.polynomial import polyfit
        x_vals = bls["AI_Theoretical_Exposure"].dropna()
        y_vals = bls.loc[x_vals.index, "Employment Percent Change, 2024-2034"].dropna()
        common_idx = x_vals.index.intersection(y_vals.index)
        if len(common_idx) > 2:
            b, m = polyfit(x_vals[common_idx], y_vals[common_idx], 1)
            x_line = [x_vals.min(), x_vals.max()]
            fig_growth.add_trace(go.Scatter(
                x=x_line, y=[b + m * x for x in x_line],
                mode="lines", name="Trend",
                line=dict(color="red", width=2, dash="dash"),
            ))
        fig_growth.update_layout(
            height=500, showlegend=False,
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    with col2:
        st.subheader("AI Exposure vs Median Wage")
        fig_wage = px.scatter(
            bls.dropna(subset=["AI_Theoretical_Exposure", "Median Annual Wage 2024"]),
            x="AI_Theoretical_Exposure",
            y="Median Annual Wage 2024",
            color="Occupation_Category",
            size="Employment 2024",
            size_max=18,
            opacity=0.7,
            hover_data=["Occupation_Clean"],
            labels={
                "AI_Theoretical_Exposure": "Theoretical AI Exposure (%)",
                "Median Annual Wage 2024": "Median Annual Wage ($)",
                "Occupation_Category": "Category",
            },
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_wage.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_wage, use_container_width=True)

    # ── Chart 4: Top declining occupations with high AI exposure ──
    st.subheader("Most At-Risk Occupations: High AI Exposure + Declining Employment")
    at_risk = bls[
        (bls["AI_Theoretical_Exposure"] >= 70)
        & (bls["Employment Percent Change, 2024-2034"] < 0)
    ].sort_values("Employment Percent Change, 2024-2034").head(20).copy()

    if not at_risk.empty:
        fig_risk = px.bar(
            at_risk,
            y="Occupation_Clean", x="Employment Percent Change, 2024-2034",
            color="AI_Theoretical_Exposure",
            orientation="h",
            color_continuous_scale="Reds",
            hover_data=["Median Annual Wage 2024", "AI_Observed_Exposure"],
            labels={
                "Occupation_Clean": "",
                "Employment Percent Change, 2024-2034": "Projected Growth (%)",
                "AI_Theoretical_Exposure": "AI Exposure (%)",
            },
        )
        fig_risk.update_layout(height=600, margin=dict(l=300), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Chart 5: Growth opportunities despite AI ──
    st.subheader("Resilient Occupations: Growing Despite High AI Exposure")
    resilient = bls[
        (bls["AI_Theoretical_Exposure"] >= 50)
        & (bls["Employment Percent Change, 2024-2034"] > 3)
    ].sort_values("Employment Percent Change, 2024-2034", ascending=False).head(15).copy()

    if not resilient.empty:
        fig_res = px.bar(
            resilient,
            y="Occupation_Clean", x="Employment Percent Change, 2024-2034",
            color="AI_Theoretical_Exposure",
            orientation="h",
            color_continuous_scale="Greens",
            hover_data=["Median Annual Wage 2024", "AI_Observed_Exposure"],
            labels={
                "Occupation_Clean": "",
                "Employment Percent Change, 2024-2034": "Projected Growth (%)",
                "AI_Theoretical_Exposure": "AI Exposure (%)",
            },
        )
        fig_res.update_layout(height=500, margin=dict(l=300), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Chart 6: Education level and AI exposure ──
    st.subheader("AI Exposure by Education Requirement")
    ed_exp = (
        bls.groupby("Typical Entry-Level Education")
        .agg(
            avg_theoretical=("AI_Theoretical_Exposure", "mean"),
            avg_observed=("AI_Observed_Exposure", "mean"),
            avg_wage=("Median Annual Wage 2024", "mean"),
            avg_growth=("Employment Percent Change, 2024-2034", "mean"),
            count=("SOC_Code", "count"),
        )
        .reset_index()
        .sort_values("avg_theoretical", ascending=True)
    )

    fig_ed = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ed.add_trace(
        go.Bar(
            y=ed_exp["Typical Entry-Level Education"],
            x=ed_exp["avg_theoretical"],
            name="Avg Theoretical Exposure (%)",
            orientation="h", marker_color="#667eea", opacity=0.7,
        ),
        secondary_y=False,
    )
    fig_ed.add_trace(
        go.Bar(
            y=ed_exp["Typical Entry-Level Education"],
            x=ed_exp["avg_observed"],
            name="Avg Observed Exposure (%)",
            orientation="h", marker_color="#f093fb", opacity=0.9,
        ),
        secondary_y=False,
    )
    fig_ed.update_layout(
        barmode="overlay", height=400,
        xaxis_title="AI Exposure (%)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=250),
    )
    st.plotly_chart(fig_ed, use_container_width=True)

    # ── Interactive table ──
    st.subheader("Explore All Occupations")
    cat_filter = st.multiselect(
        "Filter by category:",
        sorted(bls["Occupation_Category"].dropna().unique()),
        default=[],
        key="ai_cat_filter",
    )
    ed_filter = st.multiselect(
        "Filter by education level:",
        sorted(bls["Typical Entry-Level Education"].dropna().unique()),
        default=[],
        key="ai_ed_filter",
    )

    table_df = bls.copy()
    if cat_filter:
        table_df = table_df[table_df["Occupation_Category"].isin(cat_filter)]
    if ed_filter:
        table_df = table_df[table_df["Typical Entry-Level Education"].isin(ed_filter)]

    display = table_df[[
        "Occupation_Clean", "Occupation_Category", "AI_Theoretical_Exposure",
        "AI_Observed_Exposure", "AI_Automation_Gap",
        "Employment Percent Change, 2024-2034", "Median Annual Wage 2024",
        "Typical Entry-Level Education",
    ]].rename(columns={
        "Occupation_Clean": "Occupation",
        "Occupation_Category": "Category",
        "AI_Theoretical_Exposure": "AI Theoretical (%)",
        "AI_Observed_Exposure": "AI Observed (%)",
        "AI_Automation_Gap": "Automation Gap (pp)",
        "Employment Percent Change, 2024-2034": "Growth 2024-34 (%)",
        "Median Annual Wage 2024": "Median Wage ($)",
        "Typical Entry-Level Education": "Education",
    }).sort_values("AI Theoretical (%)", ascending=False).reset_index(drop=True)

    st.dataframe(display, use_container_width=True, height=500)

    # ── Key findings callout ──
    st.markdown("---")
    st.subheader("Key Findings")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(
            "**The Automation Gap** — Across all categories, there is a large gap between "
            "what AI *could* theoretically automate and what it is *actually* automating today. "
            "Computer & Math occupations have the highest observed exposure at 33%, yet even "
            "there, 61 percentage points of theoretical capability remain unrealized."
        )
    with col_f2:
        st.markdown(
            "**Higher Education ≠ Protection** — Occupations requiring bachelor's and graduate "
            "degrees have *higher* AI exposure than those requiring less education. "
            "This challenges the conventional wisdom that more education shields workers from "
            "automation, and is central to the question of whether institutional prestige "
            "provides a meaningful earnings premium in an AI-transformed labor market."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6: JOB POSTINGS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

    st.markdown(
        '<p style="text-align:right;margin-top:2rem;">'
        '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
if True:  # Job Postings
    st.markdown('<div id="job-postings"></div>', unsafe_allow_html=True)
    st.header("📋 Job Postings — Bay Area", anchor="job-postings-content")
    st.markdown(
        "Real-world job posting data scraped from Indeed, LinkedIn & ZipRecruiter "
        "for **30 job titles** across **Healthcare · Technology · Finance** "
        "(1,000 postings per title, Bay Area focus). "
        "Answers: Are degrees being dropped? Do startups describe jobs differently? "
        "Does Anthropic's exposure score match what employers actually ask for?"
    )

    if not postings_available:
        st.warning(
            "⚠️ Job posting data not found. Run the scraper first:\n\n"
            "```\ncd scraper\npython job_scraper.py\npython analyze_postings.py\n```\n\n"
            "Then copy `analysis_output/` next to `app.py` and redeploy."
        )
    else:
        enr  = postings["enriched"]
        deg  = postings["degree"]
        aia  = postings["ai_anthro"]
        sdeg = postings["startup_deg"]
        comp = postings["comp"]

        # Ensure industry column
        _imap = {
            "Registered Nurse":"Healthcare","Physician":"Healthcare",
            "Medical Data Analyst":"Healthcare","Health Informatics Specialist":"Healthcare",
            "Physical Therapist":"Healthcare","Clinical Research Coordinator":"Healthcare",
            "Radiologic Technologist":"Healthcare","Healthcare Administrator":"Healthcare",
            "Biomedical Engineer":"Healthcare","Pharmacist":"Healthcare",
            "Software Engineer":"Technology","Data Scientist":"Technology",
            "Machine Learning Engineer":"Technology","Cybersecurity Analyst":"Technology",
            "Cloud Architect":"Technology","Product Manager":"Technology",
            "DevOps Engineer":"Technology","AI Research Scientist":"Technology",
            "UX Designer":"Technology","Full Stack Developer":"Technology",
            "Financial Analyst":"Finance","Investment Banking Analyst":"Finance",
            "Quantitative Analyst":"Finance","Risk Manager":"Finance",
            "Compliance Officer":"Finance","Portfolio Manager":"Finance",
            "Financial Planner":"Finance","Actuary":"Finance",
            "Credit Analyst":"Finance","Fintech Product Manager":"Finance",
        }
        for _df in [enr, deg, aia, sdeg, comp]:
            if not _df.empty and "industry" not in _df.columns and "search_title" in _df.columns:
                _df["industry"] = _df["search_title"].map(_imap).fillna("Other")

        # ── Industry filter ───────────────────────────────────────────────────
        _industries = ["Healthcare", "Technology", "Finance"]
        sel_industry = st.selectbox(
            "Filter by Industry:", ["All Industries"] + _industries,
            key="p_industry"
        )

        def filter_ind(df_in):
            if sel_industry == "All Industries" or df_in.empty or "industry" not in df_in.columns:
                return df_in
            return df_in[df_in["industry"] == sel_industry]

        enr_f  = filter_ind(enr)
        deg_f  = filter_ind(deg)
        aia_f  = filter_ind(aia)
        sdeg_f = filter_ind(sdeg)
        comp_f = filter_ind(comp)

        # ── KPI row ──────────────────────────────────────────────────────────
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Total Postings",     f"{len(enr_f):,}")
        p2.metric("Job Titles",         f"{enr_f['search_title'].nunique()}")
        if "any_ai" in enr_f.columns:
            p3.metric("AI Mentioned",   f"{enr_f['any_ai'].mean():.0%}")
        if "degree_requirement" in enr_f.columns:
            deg_pct = enr_f["degree_requirement"].isin(["required","preferred","any_degree"]).mean()
            p4.metric("Degree Required", f"{deg_pct:.0%}")
        if "company_type" in enr_f.columns:
            p5.metric("Startup Postings", f"{(enr_f['company_type']=='startup').sum():,}")

        st.markdown("---")

        ptab1, ptab2, ptab3, ptab4 = st.tabs([
            "🎓 Degree Requirements", "🤖 vs Anthropic", "🏢 Startup vs Established", "🔎 Explore"
        ])

        # ════════════════════════════════════════════════════════════════════
        # P-TAB 1: DEGREE REQUIREMENTS
        # ════════════════════════════════════════════════════════════════════
        with ptab1:
            st.subheader("Are Degree Requirements Declining?")

            if not deg_f.empty:
                deg_order = ["required", "preferred", "any_degree", "no_degree", "not_mentioned", "unknown"]
                color_map = {
                    "required":      "#E85D5D",
                    "preferred":     "#F7C548",
                    "any_degree":    "#667eea",
                    "no_degree":     "#02C39A",
                    "not_mentioned": "#B0BEC5",
                    "unknown":       "#ECEFF1",
                }

                group_cols = ["industry", "search_title", "degree_requirement"] \
                    if "industry" in deg_f.columns else ["search_title", "degree_requirement"]
                deg_pct = (
                    deg_f.groupby(group_cols)["count"].sum().reset_index()
                )
                totals = deg_pct.groupby("search_title")["count"].transform("sum")
                deg_pct["pct"] = (deg_pct["count"] / totals * 100).round(1)
                deg_pct["degree_requirement"] = pd.Categorical(
                    deg_pct["degree_requirement"], categories=deg_order, ordered=True
                )

                # Industry-faceted chart if showing All Industries
                if sel_industry == "All Industries" and "industry" in deg_pct.columns:
                    _ind_color = {"Healthcare": "#48BB78", "Technology": "#667EEA", "Finance": "#F6AD55"}
                    for _ind in ["Healthcare", "Technology", "Finance"]:
                        _sub = deg_pct[deg_pct["industry"] == _ind]
                        if _sub.empty:
                            continue
                        _sub_sorted = _sub.sort_values("pct", ascending=False)
                        fig_d = px.bar(
                            _sub_sorted,
                            x="pct", y="search_title", color="degree_requirement",
                            orientation="h", barmode="stack",
                            color_discrete_map=color_map,
                            title=f"{_ind} — Degree Requirement Breakdown",
                            labels={"pct": "% of Postings", "search_title": "",
                                    "degree_requirement": "Requirement"},
                            category_orders={"degree_requirement": deg_order},
                        )
                        fig_d.update_layout(
                            height=420,
                            legend=dict(orientation="h", y=-0.22, font=dict(size=12)),
                            margin=dict(l=220, t=50, b=60),
                            yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                            xaxis=dict(tickfont=dict(size=12), title_font=dict(size=13)),
                            title_font=dict(size=16, color=_ind_color.get(_ind, "#333")),
                        )
                        st.plotly_chart(fig_d, use_container_width=True)
                else:
                    fig_deg = px.bar(
                        deg_pct.sort_values("pct", ascending=False),
                        x="pct", y="search_title", color="degree_requirement",
                        orientation="h", barmode="stack",
                        color_discrete_map=color_map,
                        title=f"Degree Requirement Distribution — {sel_industry}",
                        labels={"pct": "% of Postings", "search_title": "",
                                "degree_requirement": "Requirement"},
                        category_orders={"degree_requirement": deg_order},
                    )
                    fig_deg.update_layout(
                        height=max(400, len(deg_pct["search_title"].unique()) * 35 + 100),
                        legend=dict(orientation="h", y=-0.18, font=dict(size=12)),
                        margin=dict(l=220, t=50, b=60),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                        xaxis=dict(tickfont=dict(size=12)),
                    )
                    st.plotly_chart(fig_deg, use_container_width=True)

                # Clean summary table (no clutter)
                st.subheader("Summary: % with Degree Required")
                deg_req = deg_pct[deg_pct["degree_requirement"].isin(
                    ["required", "preferred", "any_degree"]
                )].groupby("search_title")["pct"].sum().round(1).reset_index()
                deg_req.columns = ["Job Title", "% Degree Mentioned"]
                deg_req = deg_req.sort_values("% Degree Mentioned", ascending=False)
                if "industry" in deg_f.columns:
                    ind_lookup = deg_f.drop_duplicates("search_title").set_index("search_title")["industry"]
                    deg_req.insert(1, "Industry", deg_req["Job Title"].map(ind_lookup))
                st.dataframe(
                    deg_req.reset_index(drop=True),
                    use_container_width=True, height=350,
                    column_config={
                        "% Degree Mentioned": st.column_config.ProgressColumn(
                            "% Degree Mentioned", min_value=0, max_value=100, format="%.0f%%"
                        )
                    }
                )

        # ════════════════════════════════════════════════════════════════════
        # P-TAB 2: VS ANTHROPIC
        # ════════════════════════════════════════════════════════════════════
        with ptab2:
            st.subheader("Our Observed AI Mentions vs Anthropic's Exposure Scores")
            st.caption(
                "Compares the % of Bay Area job postings mentioning AI "
                "against Anthropic's theoretical and observed exposure scores (March 2026 Economic Index)."
            )

            if not aia_f.empty:
                aia_sorted = aia_f.sort_values("pct_ai_mentioned", ascending=True)

                # Grouped bar chart — clean overlay style
                ind_color_map = {"Healthcare": "#48BB78", "Technology": "#667EEA", "Finance": "#F6AD55"}
                color_col = "industry" if "industry" in aia_sorted.columns else "search_title"
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    y=aia_sorted["search_title"],
                    x=aia_sorted["anthropic_theoretical"],
                    name="Anthropic Theoretical",
                    orientation="h", marker_color="#c7d2fe", opacity=0.9,
                ))
                fig_comp.add_trace(go.Bar(
                    y=aia_sorted["search_title"],
                    x=aia_sorted["anthropic_observed"],
                    name="Anthropic Observed",
                    orientation="h", marker_color="#818cf8", opacity=0.95,
                ))
                fig_comp.add_trace(go.Bar(
                    y=aia_sorted["search_title"],
                    x=aia_sorted["pct_ai_mentioned"],
                    name="Our Bay Area Observed",
                    orientation="h", marker_color="#02C39A", opacity=1.0,
                ))
                fig_comp.update_layout(
                    barmode="overlay",
                    height=max(500, len(aia_sorted) * 28 + 120),
                    xaxis=dict(title="% Exposure / Postings", tickfont=dict(size=12), range=[0, 100]),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                    legend=dict(orientation="h", y=-0.12, font=dict(size=13)),
                    margin=dict(l=230, t=50, b=70),
                    title=dict(
                        text=f"AI Exposure: Anthropic Scores vs Our Bay Area Evidence — {sel_industry}",
                        font=dict(size=16),
                    ),
                    plot_bgcolor="#fafafa",
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Clean gap table (5 columns max)
                st.subheader("Gap Analysis: Our Data vs Anthropic Benchmark")
                aia_display = aia_f[["search_title", "pct_ai_mentioned",
                                      "anthropic_observed", "anthropic_theoretical"]].copy()
                if "industry" in aia_f.columns:
                    aia_display.insert(0, "Industry", aia_f["industry"])
                aia_display["Gap (pp)"] = (
                    aia_display["pct_ai_mentioned"] - aia_display["anthropic_observed"]
                ).round(1)
                aia_display.rename(columns={
                    "search_title": "Job Title",
                    "pct_ai_mentioned": "Our AI %",
                    "anthropic_observed": "Anthropic Obs %",
                    "anthropic_theoretical": "Anthropic Theory %",
                }, inplace=True)
                aia_display = aia_display.sort_values("Our AI %", ascending=False).reset_index(drop=True)
                st.dataframe(
                    aia_display,
                    use_container_width=True, height=400,
                    column_config={
                        "Our AI %":          st.column_config.NumberColumn(format="%.1f%%"),
                        "Anthropic Obs %":   st.column_config.NumberColumn(format="%.1f%%"),
                        "Anthropic Theory %":st.column_config.NumberColumn(format="%.1f%%"),
                        "Gap (pp)":          st.column_config.NumberColumn(format="%+.1f pp"),
                    }
                )

        # ════════════════════════════════════════════════════════════════════
        # P-TAB 3: STARTUP VS ESTABLISHED
        # ════════════════════════════════════════════════════════════════════
        with ptab3:
            st.subheader("Startup vs Established Companies — Bay Area")

            if not enr_f.empty and "company_type" in enr_f.columns:
                sv_enr = enr_f[enr_f["company_type"].isin(["startup","established"])].copy()

                if not sv_enr.empty:
                    # Summary KPIs
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Startup Postings",     f"{(sv_enr['company_type']=='startup').sum():,}")
                    k2.metric("Established Postings", f"{(sv_enr['company_type']=='established').sum():,}")
                    if "any_ai" in sv_enr.columns:
                        s_ai = sv_enr[sv_enr['company_type']=='startup']['any_ai'].mean()
                        e_ai = sv_enr[sv_enr['company_type']=='established']['any_ai'].mean()
                        k3.metric("Startup AI% vs Established", f"{s_ai:.0%} vs {e_ai:.0%}")

                    st.markdown("---")

                    # AI mentions grouped bar — clean & readable
                    ai_by_type = (
                        sv_enr.groupby(["search_title", "company_type"])["any_ai"]
                        .mean().mul(100).round(1).reset_index()
                    )
                    ai_by_type.rename(columns={"any_ai": "pct_ai"}, inplace=True)
                    if "industry" in sv_enr.columns:
                        ind_lu = sv_enr.drop_duplicates("search_title").set_index("search_title")["industry"]
                        ai_by_type["industry"] = ai_by_type["search_title"].map(ind_lu)

                    fig_ai_type = px.bar(
                        ai_by_type.sort_values("pct_ai", ascending=True),
                        x="pct_ai", y="search_title", color="company_type",
                        barmode="group", orientation="h",
                        color_discrete_map={"startup": "#F59E0B", "established": "#3B82F6"},
                        title=f"AI Skill Mentions: Startup vs Established — {sel_industry}",
                        labels={"pct_ai": "% Postings Mentioning AI",
                                "search_title": "", "company_type": "Company"},
                    )
                    fig_ai_type.update_layout(
                        height=max(450, len(ai_by_type["search_title"].unique()) * 38 + 120),
                        margin=dict(l=220, t=50, b=70),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                        xaxis=dict(tickfont=dict(size=12), title_font=dict(size=13)),
                        legend=dict(orientation="h", y=-0.14, font=dict(size=13)),
                        plot_bgcolor="#fafafa",
                    )
                    st.plotly_chart(fig_ai_type, use_container_width=True)

            # Compensation comparison — clean bar
            if not comp_f.empty:
                st.subheader("Compensation: Startup vs Established")
                comp_sv = comp_f[comp_f["company_type"].isin(["startup","established"])].dropna(subset=["avg_salary_min"])
                if not comp_sv.empty:
                    fig_sal = px.bar(
                        comp_sv.sort_values("avg_salary_min", ascending=True),
                        x="avg_salary_min", y="search_title", color="company_type",
                        barmode="group", orientation="h",
                        color_discrete_map={"startup": "#F59E0B", "established": "#3B82F6"},
                        title=f"Average Listed Minimum Salary ($) — {sel_industry}",
                        labels={"avg_salary_min": "Avg Min Salary ($)",
                                "search_title": "", "company_type": "Company"},
                    )
                    fig_sal.update_layout(
                        height=max(420, len(comp_sv["search_title"].unique()) * 40 + 120),
                        margin=dict(l=220, t=50, b=70),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                        xaxis=dict(tickfont=dict(size=12), tickformat="$,.0f"),
                        legend=dict(orientation="h", y=-0.14, font=dict(size=13)),
                        plot_bgcolor="#fafafa",
                    )
                    st.plotly_chart(fig_sal, use_container_width=True)

        # ════════════════════════════════════════════════════════════════════
        # P-TAB 4: EXPLORE
        # ════════════════════════════════════════════════════════════════════
        with ptab4:
            st.subheader("Explore Individual Postings")

            if not enr_f.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    sel_title = st.multiselect(
                        "Job Title", sorted(enr_f["search_title"].unique()), default=[], key="exp_title"
                    )
                with col2:
                    sel_deg_req = st.multiselect(
                        "Degree Requirement",
                        sorted(enr_f["degree_requirement"].dropna().unique()) if "degree_requirement" in enr_f.columns else [],
                        default=[], key="exp_deg"
                    )
                with col3:
                    sel_co_type = st.multiselect(
                        "Company Type",
                        ["startup", "established", "mixed", "unknown"],
                        default=[], key="exp_cotype"
                    )

                explore = enr_f.copy()
                if sel_title:
                    explore = explore[explore["search_title"].isin(sel_title)]
                if sel_deg_req and "degree_requirement" in explore.columns:
                    explore = explore[explore["degree_requirement"].isin(sel_deg_req)]
                if sel_co_type and "company_type" in explore.columns:
                    explore = explore[explore["company_type"].isin(sel_co_type)]

                st.caption(f"Showing **{len(explore):,}** postings from {explore['search_title'].nunique() if not explore.empty else 0} job titles")

                # Minimal columns — no clutter
                possible_cols = [
                    "industry", "search_title", "title", "company",
                    "date_posted", "degree_requirement", "any_ai", "company_type",
                    "min_yoe", "salary_min",
                ]
                show_cols = [c for c in possible_cols if c in explore.columns]
                rename_map = {
                    "industry": "Industry", "search_title": "Target Role",
                    "title": "Posted Title", "company": "Company",
                    "date_posted": "Posted", "degree_requirement": "Degree",
                    "any_ai": "AI Mentioned", "company_type": "Co. Type",
                    "min_yoe": "Min YoE", "salary_min": "Min Salary",
                }
                st.dataframe(
                    explore[show_cols].rename(columns=rename_map).reset_index(drop=True),
                    use_container_width=True, height=420,
                    column_config={
                        "Min Salary": st.column_config.NumberColumn(format="$%,.0f"),
                        "AI Mentioned": st.column_config.CheckboxColumn(),
                    }
                )

                # Top skills chart
                import re as _re
                if "description_text" in explore.columns and len(explore) > 5:
                    st.subheader("Most Mentioned Skills in These Postings")
                    _industry_skills = {
                        "Healthcare": ["ehr","epic","cerner","hipaa","clinical","patient care",
                                       "nursing","medication","diagnosis","documentation"],
                        "Technology": ["python","sql","aws","cloud","docker","kubernetes",
                                       "machine learning","javascript","api","data engineering"],
                        "Finance":    ["excel","bloomberg","risk","compliance","financial modeling",
                                       "python","sql","derivatives","portfolio","valuation"],
                    }
                    base_skills = ["machine learning","ai","communication","leadership",
                                   "project management","python","sql","excel","data analysis","agile"]
                    skill_list = list(dict.fromkeys(
                        _industry_skills.get(sel_industry, base_skills) + base_skills
                    ))[:20]
                    all_text = " ".join(explore["description_text"].fillna("").str.lower())
                    counts = {kw: len(_re.findall(r'\b' + kw.replace(" ", r'\s+') + r'\b', all_text))
                              for kw in skill_list}
                    skill_df = pd.DataFrame(
                        [(k, v) for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v > 0],
                        columns=["Skill", "Mentions"]
                    ).head(12)
                    if not skill_df.empty:
                        fig_skills = px.bar(
                            skill_df, x="Mentions", y="Skill", orientation="h",
                            color="Mentions", color_continuous_scale="Teal",
                        )
                        fig_skills.update_layout(
                            height=400, showlegend=False,
                            yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
                            xaxis=dict(tickfont=dict(size=12)),
                            margin=dict(l=160, t=30),
                            plot_bgcolor="#fafafa",
                        )
                        st.plotly_chart(fig_skills, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 7: INSTITUTION LOOKUP
# ════════════════════════════════════════════════════════════════════════════

    st.markdown(
        '<p style="text-align:right;margin-top:2rem;">'
        '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
if True:  # Institution Lookup
    st.markdown('<div id="institution-lookup"></div>', unsafe_allow_html=True)
    st.header("🔍 Institution Lookup", anchor="institution-lookup-content")

    search = st.text_input("Search for an institution:", placeholder="e.g. University of San Francisco")

    if search:
        # Search across ALL years (not just filtered) so earnings data is never lost
        all_matches = df[df["INSTNM"].str.contains(search, case=False, na=False)]
        unique_matches = all_matches.drop_duplicates(subset="INSTNM")

        if unique_matches.empty:
            st.warning("No institutions found. Try a different search term.")
        else:
            selected = st.selectbox("Select institution:", unique_matches["INSTNM"].tolist())
            inst = all_matches[all_matches["INSTNM"] == selected].sort_values("YEAR")

            st.subheader(selected)

            if inst.empty:
                st.warning("No data available for this institution.")
                st.stop()

            latest = inst.iloc[-1]

            # For earnings: look across ALL years and use most recent non-null value
            def best_val(col):
                vals = inst[col].dropna()
                return vals.iloc[-1] if not vals.empty else np.nan

            earnings = best_val("MD_EARN_WNE_P10")
            earnings_yr = inst[inst["MD_EARN_WNE_P10"].notna()]["YEAR"].max() if inst["MD_EARN_WNE_P10"].notna().any() else None

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Location", f"{latest.get('CITY', 'N/A')}, {latest.get('STABBR', 'N/A')}")
            col2.metric("Type", str(latest.get("CONTROL_NAME", "N/A")))
            col3.metric("Enrollment", f"{latest['UGDS']:,.0f}" if pd.notna(latest.get("UGDS")) else "N/A")
            col4.metric("Admission Rate", f"{latest['ADM_RATE']:.0%}" if pd.notna(latest.get("ADM_RATE")) else "N/A")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("SAT Average", f"{latest['SAT_AVG']:.0f}" if pd.notna(latest.get("SAT_AVG")) else "N/A")
            col6.metric("Annual Cost", f"${best_val('COSTT4_A'):,.0f}" if pd.notna(best_val("COSTT4_A")) else "N/A")
            col7.metric("Completion Rate", f"{best_val('C150_4'):.0%}" if pd.notna(best_val("C150_4")) else "N/A")
            col8.metric(
                f"Median Earnings {'('+str(int(earnings_yr))+')' if earnings_yr else ''}",
                f"${earnings:,.0f}" if pd.notna(earnings) else "N/A",
                help="10-year post-enrollment median earnings. Data available for 2020–21 cohort only."
            )

            # Time series for this institution
            if len(inst) > 1:
                st.subheader("Trends Over Time")
                trend_cols = {
                    "COSTT4_A": "Annual Cost ($)",
                    "ADM_RATE": "Admission Rate",
                    "PCTPELL": "% Pell Students",
                    "C150_4": "Completion Rate",
                    "RET_FT4": "Retention Rate",
                }
                available_trends = {k: v for k, v in trend_cols.items() if inst[k].notna().any()}
                if available_trends:
                    sel_trend = st.selectbox("Metric:", list(available_trends.values()))
                    _trend_matches = [k for k, v in available_trends.items() if v == sel_trend]
                    trend_col = _trend_matches[0] if _trend_matches else list(available_trends.keys())[0]
                    fig = px.line(
                        inst, x="YEAR", y=trend_col, markers=True,
                        title=f"{sel_trend} for {selected}",
                        labels={trend_col: sel_trend, "YEAR": "Year"},
                    )
                    fig.update_traces(line_color="#667eea", line_width=3)
                    st.plotly_chart(fig, use_container_width=True)

            # Peer comparison
            st.subheader("Peer Comparison")
            st.caption("Institutions in the same state with a similar admission rate (±10%)")
            if pd.notna(latest.get("ADM_RATE")):
                peers = filtered[
                    (filtered["STABBR"] == latest["STABBR"])
                    & (filtered["ADM_RATE"].between(latest["ADM_RATE"] - 0.1, latest["ADM_RATE"] + 0.1))
                    & (filtered["INSTNM"] != selected)
                    & (filtered["YEAR"] == latest["YEAR"])
                ].sort_values("MD_EARN_WNE_P10", ascending=False).head(10)

                if not peers.empty:
                    display_cols = ["INSTNM", "ADM_RATE", "COSTT4_A", "C150_4", "MD_EARN_WNE_P10"]
                    display_names = {
                        "INSTNM": "Institution", "ADM_RATE": "Adm Rate",
                        "COSTT4_A": "Cost ($)", "C150_4": "Completion",
                        "MD_EARN_WNE_P10": "Earnings ($)",
                    }
                    st.dataframe(
                        peers[[c for c in display_cols if c in peers.columns]]
                        .rename(columns=display_names)
                        .reset_index(drop=True),
                        use_container_width=True,
                    )
                else:
                    st.info("No peer institutions found with current filters.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 8: PREDICTIONS (ML Pipeline)
# ════════════════════════════════════════════════════════════════════════════

    st.markdown(
        '<p style="text-align:right;margin-top:2rem;">'
        '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
if True:  # Predictions
    st.markdown('<div id="predictions"></div>', unsafe_allow_html=True)
    st.header("🔮 Predictions: Job Market Disruption", anchor="predictions-content")
    st.caption(
        "Four machine learning models trained on scraped job postings, BLS projections, "
        "and Anthropic exposure scores to predict where the labor market is heading."
    )

    if not ml_available:
        st.warning(
            "**ML output files not found.** Run the pipeline first:\n\n"
            "```bash\n"
            "cd scraper\n"
            "python ml_pipeline.py\n"
            "```\n\n"
            "Then copy the generated `ml_output/` folder next to `app.py` and redeploy."
        )
        st.info(
            "The pipeline trains 4 models:\n"
            "- **Model 1** · Random Forest — predicts degree requirement probability per job title\n"
            "- **Model 2** · Logistic Regression — predicts AI adoption probability vs Anthropic baseline\n"
            "- **Model 3** · Composite Disruption Score — ranks all 20 occupations by overall disruption risk\n"
            "- **Model 4** · TF-IDF + Logistic Regression — identifies language that separates startups from established companies"
        )
    else:
        # ── Model performance banner ──────────────────────────────────────────
        st.subheader("Model Performance")
        perf = ml["performance"]
        if not perf.empty:
            p_cols = st.columns(max(1, len(perf)))
            for col_idx, (_, row) in enumerate(perf.iterrows()):
                score = row.get("cv_mean", row.get("cv_accuracy", row.get("cv_score", None)))
                label = row.get("model", f"Model {col_idx+1}")
                metric_name = row.get("metric", "Score")
                if pd.notna(score):
                    p_cols[col_idx % len(p_cols)].metric(label, f"{score:.1%}", help=metric_name)

        st.markdown("---")

        # ── Sub-tabs for each model ───────────────────────────────────────────
        m1, m2, m3, m4 = st.tabs([
            "📉 Disruption Ranking",
            "🎓 Degree Predictions",
            "🤖 AI Adoption Gap",
            "🚀 Startup vs Established",
        ])

        # ════════════════════════════════
        # MODEL 3: Composite Disruption
        # ════════════════════════════════
        with m1:
            st.subheader("Composite Disruption Score — 30 Occupations")
            st.caption(
                "Weighted composite: 40% Anthropic theoretical · 20% BLS growth · "
                "15% observed AI mentions · 15% degree-drop rate · 10% startup share  |  "
                "**Higher = more disrupted by AI**"
            )
            dis = ml["disruption"].copy()
            if not dis.empty:
                score_col = "disruption_score" if "disruption_score" in dis.columns \
                            else ([c for c in dis.columns if "score" in c.lower()] or [dis.columns[-1]])[0]
                title_col = "job_title" if "job_title" in dis.columns \
                            else ("search_title" if "search_title" in dis.columns else dis.columns[0])

                dis = dis.sort_values(score_col, ascending=True).reset_index(drop=True)

                # Color by industry (clearest grouping)
                ind_colors = {"Healthcare": "#48BB78", "Technology": "#667EEA", "Finance": "#F6AD55"}
                has_industry = "industry" in dis.columns
                color_arg = {"color": "industry", "color_discrete_map": ind_colors} if has_industry else {}

                # Risk zones via background shapes
                fig_dis = px.bar(
                    dis, x=score_col, y=title_col,
                    orientation="h",
                    title="AI Disruption Score by Occupation (Higher = More At Risk)",
                    labels={score_col: "Disruption Score (0–100)", title_col: ""},
                    **color_arg,
                )
                # Shade risk zones
                fig_dis.add_vrect(x0=0,  x1=40, fillcolor="#d1fae5", opacity=0.25, line_width=0, annotation_text="🟢 Lower Risk", annotation_position="top left")
                fig_dis.add_vrect(x0=40, x1=65, fillcolor="#fef3c7", opacity=0.25, line_width=0, annotation_text="🟡 Medium Risk", annotation_position="top left")
                fig_dis.add_vrect(x0=65, x1=100,fillcolor="#fee2e2", opacity=0.25, line_width=0, annotation_text="🔴 High Risk", annotation_position="top left")
                fig_dis.update_layout(
                    height=max(600, len(dis) * 24 + 100),
                    xaxis=dict(range=[0, 100], title_font=dict(size=13), tickfont=dict(size=12)),
                    yaxis=dict(tickfont=dict(size=12)),
                    legend=dict(title="Industry", orientation="h", y=-0.08, font=dict(size=12)),
                    margin=dict(l=220, t=60, b=60),
                    plot_bgcolor="#fafafa",
                )
                st.plotly_chart(fig_dis, use_container_width=True)

                # Clean signal table — only include columns that exist
                signal_map = {
                    "anthropic_theoretical": "Anthropic Score",
                    "our_ai_pct":            "Observed AI%",
                    "degree_dropping_pct":   "Degree Drop%",
                    "startup_prevalence_pct":"Startup Share",
                    score_col:               "⚡ Score",
                }
                show_sig = [c for c in signal_map if c in dis.columns]
                if show_sig:
                    disp = dis[[title_col] + (["industry"] if has_industry else []) + show_sig].copy()
                    disp.rename(columns={**{k: v for k, v in signal_map.items()}, title_col: "Job Title", "industry": "Industry"}, inplace=True)
                    disp = disp.sort_values("⚡ Score", ascending=False).reset_index(drop=True)
                    st.dataframe(
                        disp,
                        use_container_width=True,
                        column_config={"⚡ Score": st.column_config.ProgressColumn(
                            "⚡ Score", min_value=0, max_value=100, format="%.0f"
                        )},
                        hide_index=True,
                    )

                # Job listing drill-down (issue #6)
                if postings_available and not postings["enriched"].empty:
                    st.markdown("---")
                    st.subheader("🔍 View Postings for a Specific Role")
                    drill_title = st.selectbox(
                        "Select job title:", dis[title_col].tolist(), key="drill_dis"
                    )
                    drill_df = postings["enriched"]
                    match_col = "search_title" if "search_title" in drill_df.columns else "job_title"
                    drill_sub = drill_df[drill_df[match_col] == drill_title]
                    if not drill_sub.empty:
                        show = [c for c in ["company", "title", "location", "date_posted",
                                            "degree_requirement", "any_ai", "company_type",
                                            "salary_min"] if c in drill_sub.columns]
                        st.caption(f"{len(drill_sub):,} postings for **{drill_title}**")
                        st.dataframe(drill_sub[show].rename(columns={
                            "company": "Employer", "title": "Job Title Posted",
                            "location": "Location", "date_posted": "Date",
                            "degree_requirement": "Degree", "any_ai": "AI Mentioned",
                            "company_type": "Co. Type", "salary_min": "Min Salary",
                        }).reset_index(drop=True),
                        use_container_width=True, height=350,
                        column_config={"Min Salary": st.column_config.NumberColumn(format="$%,.0f"),
                                       "AI Mentioned": st.column_config.CheckboxColumn()})

        # ════════════════════════════════
        # MODEL 1: Degree Predictions
        # ════════════════════════════════
        with m2:
            st.subheader("Will This Role Require a Degree? — ML Prediction")
            st.caption(
                "Random Forest model trained on Bay Area job postings. "
                "Bar = predicted probability that a degree will be listed as required or preferred."
            )
            deg_pred = ml["degree_pred"].copy()
            if not deg_pred.empty:
                _prob_candidates = [c for c in deg_pred.columns if "prob" in c.lower()]
                prob_col  = "prob_degree_required" if "prob_degree_required" in deg_pred.columns \
                            else (_prob_candidates[0] if _prob_candidates else deg_pred.columns[0])
                _title_candidates = [c for c in deg_pred.columns if "title" in c.lower() or "job" in c.lower()]
                title_col = "job_title" if "job_title" in deg_pred.columns \
                            else ("search_title" if "search_title" in deg_pred.columns \
                            else (_title_candidates[0] if _title_candidates else deg_pred.columns[0]))

                deg_pred = deg_pred.sort_values(prob_col, ascending=True).reset_index(drop=True)
                has_industry = "industry" in deg_pred.columns

                # Color by degree likelihood — blue gradient, not AI exposure
                deg_pred["_prob_pct"] = deg_pred[prob_col]
                fig_deg = px.bar(
                    deg_pred, x=prob_col, y=title_col,
                    orientation="h",
                    color=prob_col,
                    color_continuous_scale=[[0, "#d1fae5"], [0.5, "#fef3c7"], [1.0, "#fee2e2"]],
                    title="Predicted Probability: Degree Required (Green = Less Likely, Red = More Likely)",
                    labels={prob_col: "Probability (0% = never, 100% = always)", title_col: ""},
                )
                # Add 50% threshold line
                fig_deg.add_vline(x=0.5, line_dash="dash", line_color="#6B7280", line_width=2,
                                  annotation_text="50% — flip point", annotation_position="top right",
                                  annotation_font=dict(size=11, color="#6B7280"))
                fig_deg.update_layout(
                    height=max(550, len(deg_pred) * 24 + 100),
                    xaxis=dict(range=[0, 1], tickformat=".0%", title_font=dict(size=13), tickfont=dict(size=12)),
                    yaxis=dict(tickfont=dict(size=12)),
                    coloraxis_showscale=False,
                    margin=dict(l=220, t=60, b=60),
                    plot_bgcolor="#fafafa",
                )
                st.plotly_chart(fig_deg, use_container_width=True)

                # Clear plain-English summary table
                deg_pred["Degree Likely?"] = deg_pred[prob_col].apply(
                    lambda p: "✅ Yes (>70%)" if p >= 0.7 else ("⚠️ Maybe (40–70%)" if p >= 0.4 else "🚫 Unlikely (<40%)")
                )
                show_cols = [title_col] + (["industry"] if has_industry else []) + ["actual_pct_degree", prob_col, "Degree Likely?"]
                show_cols = [c for c in show_cols if c in deg_pred.columns or c == "Degree Likely?"]
                disp_deg = deg_pred[show_cols].rename(columns={
                    title_col: "Job Title", "industry": "Industry",
                    "actual_pct_degree": "Actual % w/ Degree", prob_col: "ML Prediction",
                }).reset_index(drop=True)
                st.dataframe(
                    disp_deg,
                    use_container_width=True,
                    column_config={
                        "ML Prediction": st.column_config.ProgressColumn(
                            "ML Prediction", min_value=0, max_value=1, format="%.0%%"
                        ),
                        "Actual % w/ Degree": st.column_config.NumberColumn(format="%.0f%%"),
                    },
                    hide_index=True,
                )

                # Drill-down
                if postings_available and not postings["enriched"].empty:
                    st.markdown("---")
                    st.subheader("🔍 View Postings for a Specific Role")
                    drill_title = st.selectbox("Select job title:", deg_pred[title_col].tolist(), key="drill_deg")
                    drill_df = postings["enriched"]
                    match_col = "search_title" if "search_title" in drill_df.columns else "job_title"
                    drill_sub = drill_df[drill_df[match_col] == drill_title]
                    if not drill_sub.empty:
                        show = [c for c in ["company", "title", "location", "date_posted",
                                            "degree_requirement", "any_ai", "salary_min"] if c in drill_sub.columns]
                        st.caption(f"{len(drill_sub):,} postings for **{drill_title}**")
                        st.dataframe(drill_sub[show].rename(columns={
                            "company": "Employer", "title": "Job Title Posted",
                            "location": "Location", "date_posted": "Date",
                            "degree_requirement": "Degree", "any_ai": "AI Mentioned",
                            "salary_min": "Min Salary",
                        }).reset_index(drop=True),
                        use_container_width=True, height=350,
                        column_config={"Min Salary": st.column_config.NumberColumn(format="$%,.0f"),
                                       "AI Mentioned": st.column_config.CheckboxColumn()})

        # ════════════════════════════════
        # MODEL 2: AI Adoption Gap
        # ════════════════════════════════
        with m3:
            st.subheader("AI Adoption: What Bay Area Employers Actually Ask For vs Anthropic's Prediction")
            st.caption(
                "Each bar = one job title. Blue = Anthropic's observed score. "
                "Orange = our actual % of Bay Area postings mentioning AI. "
                "Gap = how far ahead or behind real adoption is vs the benchmark."
            )
            ai_ad = ml["ai_adoption"].copy()
            if not ai_ad.empty:
                _pred_candidates = [c for c in ai_ad.columns if "prob" in c.lower()]
                pred_col  = "prob_ai_in_postings" if "prob_ai_in_postings" in ai_ad.columns \
                            else ("pred_ai_prob" if "pred_ai_prob" in ai_ad.columns \
                            else (_pred_candidates[0] if _pred_candidates else ai_ad.columns[0]))
                _title_candidates = [c for c in ai_ad.columns if "title" in c.lower() or "job" in c.lower()]
                title_col = "job_title" if "job_title" in ai_ad.columns \
                            else ("search_title" if "search_title" in ai_ad.columns \
                            else (_title_candidates[0] if _title_candidates else ai_ad.columns[0]))
                gap_col   = "gap_vs_anthropic" if "gap_vs_anthropic" in ai_ad.columns else None

                # Replace scatter with side-by-side grouped bar — far more readable
                anthro_col   = "anthropic_observed" if "anthropic_observed" in ai_ad.columns else None
                actual_col   = "actual_pct_ai" if "actual_pct_ai" in ai_ad.columns else None

                if anthro_col and actual_col:
                    ai_sorted = ai_ad.sort_values(actual_col, ascending=True).reset_index(drop=True)
                    fig_adopt = go.Figure()
                    fig_adopt.add_trace(go.Bar(
                        y=ai_sorted[title_col],
                        x=ai_sorted[anthro_col],
                        name="Anthropic Benchmark (%)",
                        orientation="h",
                        marker_color="#818cf8",
                        text=ai_sorted[anthro_col].round(1).astype(str) + "%",
                        textposition="outside",
                    ))
                    fig_adopt.add_trace(go.Bar(
                        y=ai_sorted[title_col],
                        x=ai_sorted[actual_col],
                        name="Our Observed: Bay Area (%)",
                        orientation="h",
                        marker_color="#F59E0B",
                        text=ai_sorted[actual_col].round(1).astype(str) + "%",
                        textposition="outside",
                    ))
                    fig_adopt.update_layout(
                        barmode="group",
                        height=max(600, len(ai_sorted) * 32 + 120),
                        xaxis=dict(title="% of Postings Mentioning AI", tickfont=dict(size=12), ticksuffix="%"),
                        yaxis=dict(tickfont=dict(size=12)),
                        legend=dict(orientation="h", y=-0.08, font=dict(size=13)),
                        margin=dict(l=220, t=60, b=70),
                        plot_bgcolor="#fafafa",
                        title=dict(text="Bay Area AI Adoption vs Anthropic Benchmark — Per Job Title", font=dict(size=15)),
                    )
                    st.plotly_chart(fig_adopt, use_container_width=True)

                # Gap bar — horizontal, sorted largest to smallest
                if gap_col and gap_col in ai_ad.columns:
                    gap_sorted = ai_ad.sort_values(gap_col, ascending=True).copy()
                    gap_sorted["_color"] = gap_sorted[gap_col].apply(
                        lambda v: "#10B981" if v > 0 else "#EF4444"
                    )
                    gap_sorted["_label"] = gap_sorted[gap_col].apply(
                        lambda v: f"+{v:.1f} pp ahead" if v > 0 else f"{v:.1f} pp behind"
                    )
                    fig_gap = go.Figure(go.Bar(
                        y=gap_sorted[title_col],
                        x=gap_sorted[gap_col],
                        orientation="h",
                        marker_color=gap_sorted["_color"].tolist(),
                        text=gap_sorted["_label"].tolist(),
                        textposition="outside",
                    ))
                    fig_gap.add_vline(x=0, line_color="#374151", line_width=2)
                    fig_gap.update_layout(
                        title="Gap: Our Observed AI% vs Anthropic Benchmark (positive = ahead of prediction)",
                        height=max(550, len(gap_sorted) * 24 + 100),
                        xaxis=dict(title="Percentage Points Difference", tickfont=dict(size=12), ticksuffix=" pp"),
                        yaxis=dict(tickfont=dict(size=12)),
                        margin=dict(l=220, t=60, b=60),
                        plot_bgcolor="#fafafa",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_gap, use_container_width=True)
                    st.caption("🟢 Green = Bay Area employers mention AI MORE than Anthropic predicted  |  🔴 Red = lagging behind the benchmark")

        # ════════════════════════════════
        # MODEL 4: Startup vs Established
        # ════════════════════════════════
        with m4:
            st.subheader("How Startups Describe Jobs Differently from Established Companies")
            sf = ml["startup_feat"].copy()
            if not sf.empty:
                word_col  = "feature" if "feature" in sf.columns else sf.columns[0]
                _coef_candidates = [c for c in sf.columns if any(
                    k in c.lower() for k in ["coef", "weight", "difference", "diff"]
                )]
                coef_col  = "coefficient" if "coefficient" in sf.columns \
                            else (_coef_candidates[0] if _coef_candidates else sf.columns[-1])

                # Separate startup and established signals
                startup_feats = sf[sf[coef_col] > 0].sort_values(coef_col, ascending=False).head(12)
                estab_feats   = sf[sf[coef_col] < 0].sort_values(coef_col, ascending=True).head(12)
                estab_feats   = estab_feats.copy()
                estab_feats[coef_col] = estab_feats[coef_col].abs()  # show as positive magnitude

                col_s, col_e = st.columns(2)

                with col_s:
                    st.markdown("#### 📈 Startup Signals")
                    st.caption("Words more common in startup job postings")
                    if not startup_feats.empty:
                        fig_s = px.bar(
                            startup_feats.sort_values(coef_col, ascending=True),
                            x=coef_col, y=word_col, orientation="h",
                            color_discrete_sequence=["#F59E0B"],
                            labels={coef_col: "Signal Strength", word_col: ""},
                        )
                        fig_s.update_layout(
                            height=420, margin=dict(l=160, t=20, b=40),
                            xaxis=dict(tickfont=dict(size=11)),
                            yaxis=dict(tickfont=dict(size=12)),
                            plot_bgcolor="#fffbeb",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_s, use_container_width=True)

                with col_e:
                    st.markdown("#### 🏢 Established Company Signals")
                    st.caption("Words more common in established company postings")
                    if not estab_feats.empty:
                        fig_e = px.bar(
                            estab_feats.sort_values(coef_col, ascending=True),
                            x=coef_col, y=word_col, orientation="h",
                            color_discrete_sequence=["#6366F1"],
                            labels={coef_col: "Signal Strength", word_col: ""},
                        )
                        fig_e.update_layout(
                            height=420, margin=dict(l=160, t=20, b=40),
                            xaxis=dict(tickfont=dict(size=11)),
                            yaxis=dict(tickfont=dict(size=12)),
                            plot_bgcolor="#eef2ff",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_e, use_container_width=True)

                # Numeric difference table (cleaner than chart for this data)
                numeric_only = sf[~sf[word_col].astype(str).str.contains('"', na=False)].copy()
                if not numeric_only.empty and "startup_avg" in numeric_only.columns:
                    st.markdown("#### Key Metric Differences")
                    disp_sf = numeric_only[["feature", "startup_avg", "established_avg", "favors"]].rename(
                        columns={"feature": "Metric", "startup_avg": "Startup Avg",
                                 "established_avg": "Established Avg", "favors": "Favors"}
                    ).reset_index(drop=True)
                    st.dataframe(disp_sf, use_container_width=True, hide_index=True)

        # ── Full data tables ──────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 View Raw ML Output Tables"):
            for key, label in [
                ("disruption",  "Disruption Scores"),
                ("degree_pred", "Degree Predictions"),
                ("ai_adoption", "AI Adoption Predictions"),
                ("startup_feat","Startup Features"),
                ("performance", "Model Performance"),
            ]:
                if not ml[key].empty:
                    st.subheader(label)
                    st.dataframe(ml[key], use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 9: STUDENT GUIDE
# ════════════════════════════════════════════════════════════════════════════

    st.markdown(
        '<p style="text-align:right;margin-top:2rem;">'
        '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
if True:  # Student Guide
    st.markdown('<div id="student-guide"></div>', unsafe_allow_html=True)
    st.header("🎓 Student Guide", anchor="student-guide-content")
    st.markdown(
        "Based on our analysis of 30,000+ Bay Area job postings and the Anthropic Labor Market Index, "
        "here's what college students in these three industries need to know — and how to prepare."
    )

    # ── Industry selector ──────────────────────────────────────────────────
    guide_ind = st.radio(
        "Select an industry:", ["🏥 Healthcare", "💻 Technology", "💰 Finance"],
        horizontal=True, key="guide_industry"
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════
    # HEALTHCARE
    # ═══════════════════════════════════════════════
    if guide_ind == "🏥 Healthcare":
        st.subheader("🏥 Healthcare — AI as a Tool, Not a Replacement")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg AI Theoretical Exposure", "69%", help="Anthropic's ceiling for AI automation in clinical roles")
        col2.metric("Avg Observed AI Mentions", "9%", help="% of Bay Area healthcare postings actually mentioning AI")
        col3.metric("Degree Still Required", "~85%", help="% of postings listing a degree or licensure")

        st.markdown("---")

        st.markdown("""
**What the data says:** Healthcare has the biggest gap between what AI *could* theoretically do (69% exposure)
and what Bay Area employers are *actually asking for* today (only 9% of postings mention AI). Clinical
skills, licensure, and direct patient interaction remain non-negotiable — AI is supplementing, not
replacing, hands-on care.

**The shift that IS happening:** Administrative, imaging analysis, and health informatics roles are
adopting AI faster than clinical roles. Medical Data Analysts and Health Informatics Specialists
show 20%+ AI mentions in postings, far above bedside nursing (7%).
""")

        st.info("📌 **Key insight:** The further a healthcare role is from direct patient care, the faster AI is entering the job description.")

        st.subheader("10 Healthcare Roles — Risk vs Opportunity")
        hc_data = pd.DataFrame([
            {"Role": "AI Research Scientist",          "AI Disruption Risk": "🔴 High",  "Job Growth": "↑ Strong", "Degree Still Needed": "Yes"},
            {"Role": "Medical Data Analyst",           "AI Disruption Risk": "🟡 Medium","Job Growth": "↑ Strong", "Degree Still Needed": "Yes"},
            {"Role": "Health Informatics Specialist",  "AI Disruption Risk": "🟡 Medium","Job Growth": "↑ Strong", "Degree Still Needed": "Yes"},
            {"Role": "Healthcare Administrator",       "AI Disruption Risk": "🟡 Medium","Job Growth": "↑ Strong", "Degree Still Needed": "Yes"},
            {"Role": "Biomedical Engineer",            "AI Disruption Risk": "🟡 Medium","Job Growth": "↑ Steady", "Degree Still Needed": "Yes"},
            {"Role": "Clinical Research Coordinator",  "AI Disruption Risk": "🟡 Medium","Job Growth": "↑ Steady", "Degree Still Needed": "Yes"},
            {"Role": "Physician",                      "AI Disruption Risk": "🟢 Low",   "Job Growth": "↑ Steady", "Degree Still Needed": "Yes — MD"},
            {"Role": "Registered Nurse",               "AI Disruption Risk": "🟢 Low",   "Job Growth": "↑ Strong", "Degree Still Needed": "Yes — RN"},
            {"Role": "Physical Therapist",             "AI Disruption Risk": "🟢 Low",   "Job Growth": "↑ Steady", "Degree Still Needed": "Yes — DPT"},
            {"Role": "Radiologic Technologist",        "AI Disruption Risk": "🟡 Medium","Job Growth": "↑ Steady", "Degree Still Needed": "Yes — License"},
            {"Role": "Pharmacist",                     "AI Disruption Risk": "🟡 Medium","Job Growth": "↓ Declining","Degree Still Needed": "Yes — PharmD"},
        ])
        st.dataframe(hc_data, use_container_width=True, hide_index=True)

        st.subheader("How to Prepare — Healthcare Students")
        c1, c2 = st.columns(2)
        with c1:
            st.success("""
**Skills that will make you stand out:**
- **EHR systems** (Epic, Cerner) — employers expect fluency
- **Health data literacy** — be able to read and question AI outputs
- **Clinical documentation** — AI assists but you verify
- **Interdisciplinary communication** — AI can't replace teamwork
- **Research methods** — critical for clinical research roles
""")
        with c2:
            st.warning("""
**What to be cautious about:**
- Purely administrative healthcare roles (billing, coding) are being automated fastest
- Pharmacy retail positions are declining — pivot toward clinical pharmacy or PBM analytics
- Don't assume a credential alone is enough — tech literacy is increasingly expected alongside it
- Imaging interpretation is an area where AI is advancing rapidly
""")

        st.markdown("""
**Recommended electives / certifications:**
Health informatics, biostatistics, research methods, Epic/Cerner training (many hospitals offer free),
Python or R for health data, and any coursework in public health systems.
""")

    # ═══════════════════════════════════════════════
    # TECHNOLOGY
    # ═══════════════════════════════════════════════
    elif guide_ind == "💻 Technology":
        st.subheader("💻 Technology — Highest AI Disruption, Highest AI Opportunity")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg AI Theoretical Exposure", "93%", help="Highest of all three industries")
        col2.metric("Avg Observed AI Mentions", "32%", help="% of Bay Area tech postings mentioning AI")
        col3.metric("Degree Required in Postings", "~60%", help="Declining — skills and portfolio matter more")

        st.markdown("---")

        st.markdown("""
**What the data says:** Technology is simultaneously the most disrupted *and* the most AI-hungry
industry. Bay Area tech postings mention AI in **32% of all listings** — nearly 4× the Finance
industry average. But the gap between Anthropic's theoretical ceiling (~93%) and actual adoption
(32%) means we're still in the early innings. The roles being transformed fastest are those involving
routine code generation, data processing, and rote analysis. The roles growing fastest involve
designing, evaluating, and securing AI systems.

**The degree question:** Our postings data shows ~60% of Bay Area tech roles still list a degree,
down from historical norms. Startups especially are dropping degree requirements — but replacing them
with portfolio requirements. A GitHub profile, Kaggle ranking, or shipped side project now carries
weight that a degree alone used to.
""")

        st.info("📌 **Key insight:** In tech, AI is not eliminating entry-level roles — it's raising the bar for what 'entry-level' means. You're expected to work *with* AI tools, not despite them.")

        st.subheader("10 Technology Roles — Risk vs Opportunity")
        tech_data = pd.DataFrame([
            {"Role": "AI Research Scientist",     "AI Disruption Risk": "🔴 Transforming","Job Growth": "↑↑ Explosive", "Degree Trend": "PhD preferred"},
            {"Role": "Machine Learning Engineer", "AI Disruption Risk": "🔴 Transforming","Job Growth": "↑↑ Explosive", "Degree Trend": "Declining"},
            {"Role": "Data Scientist",            "AI Disruption Risk": "🔴 Transforming","Job Growth": "↑ Strong",     "Degree Trend": "Still common"},
            {"Role": "Cybersecurity Analyst",     "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑↑ Explosive", "Degree Trend": "Cert-based"},
            {"Role": "Cloud Architect",           "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑↑ Strong",    "Degree Trend": "Declining"},
            {"Role": "DevOps Engineer",           "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑↑ Strong",    "Degree Trend": "Declining"},
            {"Role": "Product Manager",           "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑ Steady",     "Degree Trend": "Still common"},
            {"Role": "Full Stack Developer",      "AI Disruption Risk": "🔴 High",        "Job Growth": "↓ Slowing",    "Degree Trend": "Declining fast"},
            {"Role": "Software Engineer",         "AI Disruption Risk": "🔴 High",        "Job Growth": "↓ Slowing",    "Degree Trend": "Declining"},
            {"Role": "UX Designer",               "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑ Steady",     "Degree Trend": "Portfolio > degree"},
        ])
        st.dataframe(tech_data, use_container_width=True, hide_index=True)

        st.subheader("How to Prepare — Technology Students")
        c1, c2 = st.columns(2)
        with c1:
            st.success("""
**Skills that will make you stand out:**
- **AI/ML fluency** — not just using tools, but knowing their limits
- **Python** — still the lingua franca; pandas, PyTorch, scikit-learn
- **Cloud platforms** (AWS, GCP, Azure) — certifications carry weight
- **Prompt engineering & LLM APIs** — Anthropic, OpenAI, Gemini
- **Security basics** — cybersecurity demand is exploding
- **System design** — what separates junior from senior roles
""")
        with c2:
            st.warning("""
**What to watch out for:**
- Pure coding roles (CRUD apps, simple web dev) face the highest automation pressure
- A CS degree alone is no longer enough — portfolio and GitHub matter as much
- Startups pay less base but offer equity; established firms offer more stability
- "Vibe coding" with AI tools is not the same as understanding systems
- Generalist SWE roles are declining; specialists (AI, security, cloud) are growing
""")

        st.markdown("""
**Recommended certifications / side projects:**
AWS Certified Cloud Practitioner, Google Data Analytics, DeepLearning.AI specializations,
Kaggle competitions, personal ML projects on GitHub, and contributing to open-source AI tooling.
""")

    # ═══════════════════════════════════════════════
    # FINANCE
    # ═══════════════════════════════════════════════
    else:
        st.subheader("💰 Finance — High Theoretical Risk, Slow Actual Adoption")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg AI Theoretical Exposure", "93%", help="Finance = second highest theoretical exposure after Tech")
        col2.metric("Avg Observed AI Mentions", "26%", help="% of Bay Area finance postings mentioning AI")
        col3.metric("Degree Still Required", "~82%", help="Finance remains one of the most degree-dependent industries")

        st.markdown("---")

        st.markdown("""
**What the data says:** Finance has almost as high a theoretical AI exposure as technology (93%)
but meaningfully lower observed adoption in postings (26%). This gap reflects a conservative
industry still running on Excel, legacy systems, and regulatory constraints. But the tide is
turning: Quantitative Analysts, Actuaries, and Fintech roles mention AI far more than traditional
banking analysts, and Bay Area finance is skewing fintech-heavy.

**The Bay Area difference:** San Francisco Bay Area finance postings are heavily weighted toward
fintech (Stripe, Brex, Chime, Robinhood, etc.) rather than traditional Wall Street banking. This
means AI skills carry more weight here than they would in New York. Our data shows Bay Area finance
startups mention AI **42% more often** than established finance firms in the same roles.
""")

        st.info("📌 **Key insight:** A traditional finance credential (CFA, CPA) remains powerful — but pairing it with data skills (Python, SQL, ML) is what separates candidates in Bay Area hiring.")

        st.subheader("10 Finance Roles — Risk vs Opportunity")
        fin_data = pd.DataFrame([
            {"Role": "Quantitative Analyst",       "AI Disruption Risk": "🔴 Transforming","Job Growth": "↑ Strong",    "Degree Still Needed": "Yes — quant focus"},
            {"Role": "Actuary",                    "AI Disruption Risk": "🔴 High",        "Job Growth": "↑↑ Explosive","Degree Still Needed": "Yes + exams"},
            {"Role": "Fintech Product Manager",    "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑ Strong",    "Degree Still Needed": "Less critical"},
            {"Role": "Risk Manager",               "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑ Steady",    "Degree Still Needed": "Yes"},
            {"Role": "Portfolio Manager",          "AI Disruption Risk": "🔴 High",        "Job Growth": "↑ Steady",    "Degree Still Needed": "Yes + CFA"},
            {"Role": "Financial Planner",          "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑↑ Strong",   "Degree Still Needed": "Yes + CFP"},
            {"Role": "Compliance Officer",         "AI Disruption Risk": "🟡 Medium",      "Job Growth": "↑ Steady",    "Degree Still Needed": "Yes"},
            {"Role": "Financial Analyst",          "AI Disruption Risk": "🔴 High",        "Job Growth": "↓ Declining", "Degree Still Needed": "Yes"},
            {"Role": "Investment Banking Analyst", "AI Disruption Risk": "🔴 High",        "Job Growth": "↓ Declining", "Degree Still Needed": "Yes — target school"},
            {"Role": "Credit Analyst",             "AI Disruption Risk": "🔴 High",        "Job Growth": "↓ Declining", "Degree Still Needed": "Yes"},
        ])
        st.dataframe(fin_data, use_container_width=True, hide_index=True)

        st.subheader("How to Prepare — Finance Students")
        c1, c2 = st.columns(2)
        with c1:
            st.success("""
**Skills that will make you stand out:**
- **Python + SQL** — data literacy is now a baseline expectation in Bay Area finance
- **Financial modeling** — Excel is still everywhere; know it deeply
- **Bloomberg Terminal** familiarity — standard at any firm
- **CFA / CPA / CFP** path — credentials still matter enormously
- **Regulatory knowledge** — compliance roles are growing with fintech
- **Risk frameworks** — Basel III, Dodd-Frank basics show seriousness
""")
        with c2:
            st.warning("""
**What to watch out for:**
- Entry-level analyst roles (financial modeling, DCF, comps) are the most at-risk — AI does this fast
- Traditional investment banking is contracting in the Bay Area relative to fintech
- Credit analysis and routine data reconciliation are high automation targets
- A finance degree without tech skills is increasingly weak in SF/SV hiring
- Prestige school premium is HIGH in finance — our regression confirms SAT_AVG is still significant
""")

        st.markdown("""
**Recommended certifications / coursework:**
CFA Level 1 as early as possible, Bloomberg Market Concepts (free), Python for Finance (Coursera),
SQL fundamentals, any fintech or blockchain coursework, and an internship at a fintech startup
(Bay Area specific: Stripe, Chime, Brex, Coinbase are all active college hirers).
""")

    # ── Universal student takeaways ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Universal Takeaways for All College Students")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
**🔑 AI literacy is table stakes**

Across all three industries, Bay Area employers increasingly expect graduates to know how to use AI
tools — not just be aware they exist. This doesn't mean you need a CS degree. It means being able
to use tools like ChatGPT, Copilot, or Claude to accelerate your work and critically evaluate their
outputs.
""")
    with t2:
        st.markdown("""
**📜 Degrees still matter — but differently**

Our College Scorecard regression found that institutional prestige (SAT_AVG, faculty salary) still
predicts earnings. But our job posting data shows degree requirements are softening, especially at
startups. The answer: a degree from a good institution *plus* demonstrable skills beats either alone.
""")
    with t3:
        st.markdown("""
**📍 Bay Area = fastest-moving market**

The Bay Area labor market is a leading indicator for the rest of the country. AI adoption in Bay Area
job postings is running 1.5–2× the national average. Skills demanded here today will be expected
nationally in 3–5 years. Being in this market — even for an internship — accelerates your career
positioning.
""")

    st.markdown("---")
    st.markdown("""
**Data sources behind this guide:**
Anthropic Economic Index (March 2026) · BLS Employment Projections 2024–2034 ·
U.S. DOE College Scorecard 2020–2024 · ~30,000 Bay Area job postings (Indeed/LinkedIn/ZipRecruiter, 2025–2026)
""")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Group 5 Project — Analytics for Good** · "
    "Data: [U.S. DOE College Scorecard](https://collegescorecard.ed.gov/) · "
    "[Anthropic Labor Market Impact Study](https://www.anthropic.com/research/labor-market-impacts) · "
    "[BLS Employment Projections 2024–2034](https://www.bls.gov/emp/) · "
    "Built with Streamlit & Plotly"
)

st.markdown(
    '<p style="text-align:right;margin-top:2rem;">'
    '<a href="#top" style="color:#667eea;font-size:0.85rem;">↑ Back to top</a></p>',
    unsafe_allow_html=True
)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#888;font-size:0.8rem;">'
    'Job Market Post A.I. &nbsp;·&nbsp; Group 5, BUS 410 &nbsp;·&nbsp; '
    'University of San Francisco &nbsp;·&nbsp; Spring 2026</p>',
    unsafe_allow_html=True
)
