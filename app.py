"""
College Scorecard Explorer
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
    page_title="College Scorecard Explorer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetric"] label { font-size: 0.85rem; color: #555; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }
    h1 { color: #1a1a2e; }
    h2 { color: #16213e; border-bottom: 2px solid #667eea; padding-bottom: 4px; }
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

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.title("🎓 Filters")

# Year
years = sorted(df["YEAR"].dropna().unique())
sel_years = st.sidebar.multiselect("Academic Year", years, default=years)

# Institution type
inst_types = ["Public", "Private Nonprofit", "Private For-Profit"]
sel_types = st.sidebar.multiselect("Institution Type", inst_types, default=inst_types)

# Degree level
deg_opts = sorted(df["PREDDEG_NAME"].dropna().unique())
sel_degs = st.sidebar.multiselect("Predominant Degree", deg_opts, default=["Bachelor's"])

# State
states = sorted(df["STABBR"].dropna().unique())
sel_states = st.sidebar.multiselect("State", states, default=[])

# Size filter
st.sidebar.markdown("---")
st.sidebar.subheader("Size & Selectivity")
min_size = st.sidebar.number_input("Min enrollment", value=0, step=500)
max_adm = st.sidebar.slider("Max admission rate", 0.0, 1.0, 1.0, 0.05)

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

# ── Header ───────────────────────────────────────────────────────────────────
st.title("College Scorecard Explorer")
st.caption(
    "Analyzing U.S. higher education outcomes across 6,500+ institutions · "
    "Data: U.S. Department of Education College Scorecard (2020–2024)"
)

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Institutions", f"{filtered['UNITID'].nunique():,}")
k2.metric("Median Earnings (10yr)", f"${filtered['MD_EARN_WNE_P10'].median():,.0f}" if filtered['MD_EARN_WNE_P10'].notna().any() else "N/A")
k3.metric("Avg Admission Rate", f"{filtered['ADM_RATE'].mean():.0%}" if filtered['ADM_RATE'].notna().any() else "N/A")
k4.metric("Median Cost", f"${filtered['COSTT4_A'].median():,.0f}" if filtered['COSTT4_A'].notna().any() else "N/A")
k5.metric("Avg Completion Rate", f"{filtered['C150_4'].mean():.0%}" if filtered['C150_4'].notna().any() else "N/A")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "💰 Earnings & ROI", "🎯 Admissions", "🗺️ Geographic",
    "🤖 AI Impact", "📋 Job Postings", "🔍 Institution Lookup"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Institution Landscape")

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


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: EARNINGS & ROI
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Earnings & Return on Investment")
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


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: ADMISSIONS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Admissions & Selectivity")

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


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: GEOGRAPHIC
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Geographic Analysis")

    metric_choice = st.selectbox(
        "Color institutions by:",
        ["Median Earnings (10yr)", "Admission Rate", "Average Cost", "Completion Rate", "Pell Grant %"],
    )
    metric_map = {
        "Median Earnings (10yr)": "MD_EARN_WNE_P10",
        "Admission Rate": "ADM_RATE",
        "Average Cost": "COSTT4_A",
        "Completion Rate": "C150_4",
        "Pell Grant %": "PCTPELL",
    }
    metric_col = metric_map[metric_choice]

    geo = filtered.dropna(subset=["LATITUDE", "LONGITUDE", metric_col]).copy()
    if not geo.empty:
        fig = px.scatter_mapbox(
            geo, lat="LATITUDE", lon="LONGITUDE",
            color=metric_col, size="UGDS", size_max=12,
            hover_name="INSTNM",
            hover_data={"STABBR": True, "CONTROL_NAME": True, metric_col: ":.2f", "UGDS": ":,"},
            title=f"Institutions by {metric_choice}",
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            zoom=3, center={"lat": 39.5, "lon": -98.35},
        )
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No geographic data available with current filters.")

    # State-level aggregation
    st.subheader("State-Level Averages")
    state_agg = (
        filtered.groupby("STABBR")
        .agg(
            institutions=("UNITID", "nunique"),
            avg_earnings=("MD_EARN_WNE_P10", "mean"),
            avg_cost=("COSTT4_A", "mean"),
            avg_adm=("ADM_RATE", "mean"),
            avg_completion=("C150_4", "mean"),
        )
        .reset_index()
    )

    state_metric = st.selectbox(
        "State map metric:",
        ["avg_earnings", "avg_cost", "avg_adm", "avg_completion", "institutions"],
    )
    state_labels = {
        "avg_earnings": "Avg Earnings ($)",
        "avg_cost": "Avg Cost ($)",
        "avg_adm": "Avg Admission Rate",
        "avg_completion": "Avg Completion Rate",
        "institutions": "# Institutions",
    }

    fig_state = px.choropleth(
        state_agg, locations="STABBR", locationmode="USA-states",
        color=state_metric, scope="usa",
        color_continuous_scale="Viridis",
        title=f"States by {state_labels[state_metric]}",
        labels={"STABBR": "State", state_metric: state_labels[state_metric]},
    )
    fig_state.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_state, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5: AI IMPACT
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("AI Impact on Career Fields")
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
with tab6:
    st.header("Job Postings Analysis")
    st.markdown(
        "Real-world job posting data scraped from Indeed, LinkedIn & ZipRecruiter "
        "for **20 job titles** (10 high AI exposure, 10 low AI exposure). "
        "Answers: Are degrees being dropped? Do startups describe jobs differently? "
        "Does Anthropic's exposure score match what employers actually ask for?"
    )

    if not postings_available:
        st.warning(
            "⚠️ Job posting data not found. Make sure the `analysis_output/` folder "
            "is in the same directory as `app.py` and contains the CSV files from "
            "`analyze_postings.py`."
        )
    else:
        enr  = postings["enriched"]
        deg  = postings["degree"]
        aia  = postings["ai_anthro"]
        sdeg = postings["startup_deg"]
        comp = postings["comp"]

        # ── KPI row ──────────────────────────────────────────────────────────
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Total Postings",     f"{len(enr):,}")
        p2.metric("Job Titles",         f"{enr['search_title'].nunique()}")
        if "any_ai" in enr.columns:
            p3.metric("Mention AI Tools",   f"{enr['any_ai'].mean():.0%}")
        if "degree_requirement" in enr.columns:
            no_deg = (enr["degree_requirement"].isin(["no_degree","not_mentioned"])).mean()
            p4.metric("No Degree Mentioned", f"{no_deg:.0%}")
        if "company_type" in enr.columns:
            n_startup = (enr["company_type"] == "startup").sum()
            p5.metric("Startup Postings",   f"{n_startup:,}")

        st.markdown("---")

        ptab1, ptab2, ptab3, ptab4 = st.tabs([
            "🎓 Degree Requirements", "🤖 vs Anthropic", "🏢 Startup vs Established", "🔎 Explore"
        ])

        # ════════════════════════════════════════════════════════════════════
        # P-TAB 1: DEGREE REQUIREMENTS
        # ════════════════════════════════════════════════════════════════════
        with ptab1:
            st.subheader("Are Degree Requirements Declining?")

            if not deg.empty:
                # Order degree categories meaningfully
                deg_order = ["required", "preferred", "any_degree", "no_degree", "not_mentioned", "unknown"]
                deg["degree_requirement"] = pd.Categorical(
                    deg["degree_requirement"], categories=deg_order, ordered=True
                )
                deg_sorted = deg.sort_values(["ai_exposed", "search_title", "degree_requirement"])

                # Stacked bar: all titles, colored by degree requirement
                deg_pct = (
                    deg_sorted.groupby(["search_title", "ai_exposed", "degree_requirement"])["count"]
                    .sum().reset_index()
                )
                totals = deg_pct.groupby("search_title")["count"].transform("sum")
                deg_pct["pct"] = deg_pct["count"] / totals * 100

                color_map = {
                    "required":      "#E85D5D",
                    "preferred":     "#F7C548",
                    "any_degree":    "#667eea",
                    "no_degree":     "#02C39A",
                    "not_mentioned": "#B0BEC5",
                    "unknown":       "#ECEFF1",
                }

                fig_deg = px.bar(
                    deg_pct,
                    x="pct", y="search_title", color="degree_requirement",
                    orientation="h", barmode="stack",
                    color_discrete_map=color_map,
                    title="Degree Requirement Distribution by Job Title",
                    labels={"pct": "% of Postings", "search_title": "",
                            "degree_requirement": "Requirement"},
                    category_orders={"degree_requirement": deg_order},
                )
                fig_deg.update_layout(
                    height=600, legend=dict(orientation="h", y=-0.15),
                    margin=dict(l=200), yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_deg, use_container_width=True)

                # AI-exposed vs not — summary comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("High AI Exposure Jobs")
                    hi = deg_pct[deg_pct["ai_exposed"] == True]
                    hi_sum = hi.groupby("degree_requirement")["pct"].mean().reset_index()
                    fig_hi = px.pie(
                        hi_sum, names="degree_requirement", values="pct",
                        color="degree_requirement", color_discrete_map=color_map,
                        title="Avg Degree Requirement Mix (High AI)",
                    )
                    fig_hi.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig_hi, use_container_width=True)

                with col2:
                    st.subheader("Low AI Exposure Jobs")
                    lo = deg_pct[deg_pct["ai_exposed"] == False]
                    lo_sum = lo.groupby("degree_requirement")["pct"].mean().reset_index()
                    fig_lo = px.pie(
                        lo_sum, names="degree_requirement", values="pct",
                        color="degree_requirement", color_discrete_map=color_map,
                        title="Avg Degree Requirement Mix (Low AI)",
                    )
                    fig_lo.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig_lo, use_container_width=True)

                # Key stat callout
                if "any_degree" in deg_pct["degree_requirement"].values:
                    hi_deg_pct = deg_pct[
                        (deg_pct["ai_exposed"] == True) &
                        (deg_pct["degree_requirement"].isin(["required","preferred","any_degree"]))
                    ]["pct"].mean()
                    lo_deg_pct = deg_pct[
                        (deg_pct["ai_exposed"] == False) &
                        (deg_pct["degree_requirement"].isin(["required","preferred","any_degree"]))
                    ]["pct"].mean()
                    diff = hi_deg_pct - lo_deg_pct
                    direction = "more" if diff > 0 else "fewer"
                    st.info(
                        f"📌 High-AI-exposure jobs mention degree requirements in "
                        f"**{hi_deg_pct:.0f}%** of postings vs **{lo_deg_pct:.0f}%** "
                        f"for low-AI jobs — **{abs(diff):.0f} percentage points {direction}**."
                    )

        # ════════════════════════════════════════════════════════════════════
        # P-TAB 2: VS ANTHROPIC
        # ════════════════════════════════════════════════════════════════════
        with ptab2:
            st.subheader("Our Observed AI Mentions vs Anthropic's Exposure Scores")
            st.caption(
                "Compares the % of job postings that mention AI tools/skills "
                "against Anthropic's theoretical and observed exposure scores from the "
                "March 2026 Economic Index."
            )

            if not aia.empty:
                aia_sorted = aia.sort_values("pct_ai_mentioned", ascending=True)

                # Grouped bar: our observed vs Anthropic's observed vs theoretical
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    y=aia_sorted["search_title"],
                    x=aia_sorted["anthropic_theoretical"],
                    name="Anthropic: Theoretical (%)",
                    orientation="h", marker_color="#667eea", opacity=0.5,
                ))
                fig_comp.add_trace(go.Bar(
                    y=aia_sorted["search_title"],
                    x=aia_sorted["anthropic_observed"],
                    name="Anthropic: Observed (%)",
                    orientation="h", marker_color="#667eea", opacity=0.85,
                ))
                fig_comp.add_trace(go.Bar(
                    y=aia_sorted["search_title"],
                    x=aia_sorted["pct_ai_mentioned"],
                    name="Our Observed: AI Mentions in Postings (%)",
                    orientation="h", marker_color="#02C39A", opacity=0.95,
                ))
                fig_comp.update_layout(
                    barmode="overlay", height=600,
                    xaxis_title="% Exposure / % Postings Mentioning AI",
                    legend=dict(orientation="h", y=-0.15),
                    margin=dict(l=220),
                    title="AI Exposure: Anthropic Scores vs Job Posting Evidence",
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Scatter: Anthropic observed vs our observed
                col1, col2 = st.columns(2)
                with col1:
                    fig_sc = px.scatter(
                        aia,
                        x="anthropic_observed", y="pct_ai_mentioned",
                        text="search_title", color="ai_exposed",
                        color_discrete_map={True: "#E85D5D", False: "#02C39A"},
                        labels={
                            "anthropic_observed": "Anthropic Observed Exposure (%)",
                            "pct_ai_mentioned": "Our: AI Mentions in Postings (%)",
                            "ai_exposed": "High AI Exposure",
                        },
                        title="Correlation: Anthropic vs Our Analysis",
                    )
                    fig_sc.update_traces(textposition="top center", textfont_size=9)
                    # Add y=x reference line
                    max_val = max(
                        aia["anthropic_observed"].max(),
                        aia["pct_ai_mentioned"].max()
                    ) + 5
                    fig_sc.add_trace(go.Scatter(
                        x=[0, max_val], y=[0, max_val],
                        mode="lines", name="Perfect agreement",
                        line=dict(color="gray", dash="dash", width=1),
                    ))
                    st.plotly_chart(fig_sc, use_container_width=True)

                with col2:
                    # Table view
                    aia_display = aia[[
                        "search_title", "postings", "pct_ai_mentioned",
                        "pct_ai_tools", "anthropic_observed", "anthropic_theoretical",
                    ]].copy()
                    aia_display.columns = [
                        "Job Title", "Postings", "Our AI % ", "AI Tools %",
                        "Anthropic Observed %", "Anthropic Theoretical %",
                    ]
                    aia_display["Gap (Ours - Anthropic)"] = (
                        aia_display["Our AI % "] - aia_display["Anthropic Observed %"]
                    ).round(1)
                    aia_display = aia_display.sort_values("Our AI % ", ascending=False)
                    st.dataframe(
                        aia_display.reset_index(drop=True),
                        use_container_width=True, height=500,
                    )

        # ════════════════════════════════════════════════════════════════════
        # P-TAB 3: STARTUP VS ESTABLISHED
        # ════════════════════════════════════════════════════════════════════
        with ptab3:
            st.subheader("Startup vs Established Companies")

            if not sdeg.empty:
                # Degree requirements: startup vs established
                sv = sdeg[sdeg["company_type"].isin(["startup", "established"])].copy()
                if not sv.empty:
                    sv_totals = sv.groupby(["search_title","company_type"])["count"].transform("sum")
                    sv["pct"] = sv["count"] / sv_totals * 100

                    fig_sv = px.bar(
                        sv[sv["degree_requirement"].isin(["required","preferred","no_degree","not_mentioned"])],
                        x="pct", y="search_title", color="degree_requirement",
                        facet_col="company_type",
                        orientation="h", barmode="stack",
                        color_discrete_map={
                            "required": "#E85D5D", "preferred": "#F7C548",
                            "no_degree": "#02C39A", "not_mentioned": "#B0BEC5",
                        },
                        title="Degree Requirements: Startup vs Established",
                        labels={"pct": "% of Postings", "search_title": "",
                                "degree_requirement": "Requirement"},
                    )
                    fig_sv.update_layout(
                        height=600, margin=dict(l=200),
                        yaxis=dict(autorange="reversed"),
                        legend=dict(orientation="h", y=-0.15),
                    )
                    st.plotly_chart(fig_sv, use_container_width=True)

            # AI mentions: startup vs established
            if not enr.empty and "any_ai" in enr.columns and "company_type" in enr.columns:
                ai_by_type = (
                    enr[enr["company_type"].isin(["startup","established"])]
                    .groupby(["search_title","ai_exposed","company_type"])["any_ai"]
                    .agg(["mean","count"]).reset_index()
                )
                ai_by_type["mean"] *= 100
                ai_by_type.rename(columns={"mean":"pct_ai","count":"n"}, inplace=True)

                if not ai_by_type.empty:
                    fig_ai_type = px.bar(
                        ai_by_type,
                        x="pct_ai", y="search_title", color="company_type",
                        barmode="group", orientation="h",
                        color_discrete_map={"startup":"#F7C548","established":"#065A82"},
                        title="AI Skill Mentions: Startup vs Established",
                        labels={"pct_ai": "% Postings Mentioning AI",
                                "search_title": "", "company_type": "Company Type"},
                    )
                    fig_ai_type.update_layout(
                        height=580, margin=dict(l=200),
                        yaxis=dict(autorange="reversed"),
                        legend=dict(orientation="h", y=-0.15),
                    )
                    st.plotly_chart(fig_ai_type, use_container_width=True)

            # Compensation comparison
            if not comp.empty:
                st.subheader("Compensation: Startup vs Established")
                comp_sv = comp[comp["company_type"].isin(["startup","established"])].dropna(subset=["avg_salary_min"])
                if not comp_sv.empty:
                    fig_sal = px.bar(
                        comp_sv,
                        x="avg_salary_min", y="search_title", color="company_type",
                        barmode="group", orientation="h",
                        color_discrete_map={"startup":"#F7C548","established":"#065A82"},
                        title="Average Minimum Salary: Startup vs Established ($)",
                        labels={"avg_salary_min":"Avg Min Salary ($)",
                                "search_title":"","company_type":"Company Type"},
                    )
                    fig_sal.update_layout(
                        height=500, margin=dict(l=200),
                        yaxis=dict(autorange="reversed"),
                        legend=dict(orientation="h", y=-0.15),
                    )
                    st.plotly_chart(fig_sal, use_container_width=True)

        # ════════════════════════════════════════════════════════════════════
        # P-TAB 4: EXPLORE
        # ════════════════════════════════════════════════════════════════════
        with ptab4:
            st.subheader("Explore Postings")

            if not enr.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    sel_title = st.multiselect(
                        "Job Title", sorted(enr["search_title"].unique()), default=[]
                    )
                with col2:
                    sel_deg_req = st.multiselect(
                        "Degree Requirement",
                        sorted(enr["degree_requirement"].dropna().unique()) if "degree_requirement" in enr.columns else [],
                        default=[],
                    )
                with col3:
                    sel_co_type = st.multiselect(
                        "Company Type",
                        ["startup", "established", "mixed", "unknown"],
                        default=[],
                    )

                explore = enr.copy()
                if sel_title:
                    explore = explore[explore["search_title"].isin(sel_title)]
                if sel_deg_req:
                    explore = explore[explore["degree_requirement"].isin(sel_deg_req)]
                if sel_co_type and "company_type" in explore.columns:
                    explore = explore[explore["company_type"].isin(sel_co_type)]

                st.caption(f"Showing {len(explore):,} postings")

                # Choose display columns that actually exist
                possible_cols = [
                    "search_title", "ai_exposed", "title", "company", "location",
                    "date_posted", "degree_requirement", "any_ai", "company_type",
                    "min_yoe", "salary_min", "salary_max",
                ]
                show_cols = [c for c in possible_cols if c in explore.columns]
                rename_map = {
                    "search_title": "Target Title", "ai_exposed": "High AI",
                    "title": "Posted Title", "company": "Company",
                    "location": "Location", "date_posted": "Date",
                    "degree_requirement": "Degree Req.", "any_ai": "AI Mention",
                    "company_type": "Co. Type", "min_yoe": "Min YoE",
                    "salary_min": "Salary Min", "salary_max": "Salary Max",
                }
                st.dataframe(
                    explore[show_cols].rename(columns=rename_map).reset_index(drop=True),
                    use_container_width=True, height=500,
                )

                # Word frequency in descriptions for selected subset
                if "description_text" in explore.columns and len(explore) > 0:
                    st.subheader("Top Skills Mentioned")
                    from collections import Counter
                    import re as _re
                    skill_keywords = [
                        "python","sql","excel","tableau","power bi","machine learning",
                        "ai","data analysis","project management","communication",
                        "leadership","agile","scrum","java","javascript","cloud",
                        "aws","azure","r programming","statistics","finance",
                        "accounting","healthcare","electrical","hvac","plumbing",
                    ]
                    all_text = " ".join(explore["description_text"].fillna("").str.lower())
                    counts = {kw: len(_re.findall(r'\b' + kw.replace(" ", r'\s+') + r'\b', all_text))
                              for kw in skill_keywords}
                    skill_df = pd.DataFrame(
                        sorted(counts.items(), key=lambda x: -x[1]),
                        columns=["Skill", "Mentions"]
                    ).head(15)
                    fig_skills = px.bar(
                        skill_df, x="Mentions", y="Skill", orientation="h",
                        color="Mentions", color_continuous_scale="Teal",
                        title="Most Mentioned Skills in Selected Postings",
                    )
                    fig_skills.update_layout(
                        height=450, showlegend=False,
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_skills, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 7: INSTITUTION LOOKUP
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Institution Lookup")

    search = st.text_input("Search for an institution:", placeholder="e.g. University of San Francisco")

    if search:
        matches = filtered[filtered["INSTNM"].str.contains(search, case=False, na=False)]
        unique_matches = matches.drop_duplicates(subset="INSTNM")

        if unique_matches.empty:
            st.warning("No institutions found. Try a different search term.")
        else:
            selected = st.selectbox("Select institution:", unique_matches["INSTNM"].tolist())
            inst = matches[matches["INSTNM"] == selected].sort_values("YEAR")

            st.subheader(selected)

            latest = inst.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Location", f"{latest.get('CITY', 'N/A')}, {latest.get('STABBR', 'N/A')}")
            col2.metric("Type", str(latest.get("CONTROL_NAME", "N/A")))
            col3.metric("Enrollment", f"{latest['UGDS']:,.0f}" if pd.notna(latest.get("UGDS")) else "N/A")
            col4.metric("Admission Rate", f"{latest['ADM_RATE']:.0%}" if pd.notna(latest.get("ADM_RATE")) else "N/A")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("SAT Average", f"{latest['SAT_AVG']:.0f}" if pd.notna(latest.get("SAT_AVG")) else "N/A")
            col6.metric("Annual Cost", f"${latest['COSTT4_A']:,.0f}" if pd.notna(latest.get("COSTT4_A")) else "N/A")
            col7.metric("Completion Rate", f"{latest['C150_4']:.0%}" if pd.notna(latest.get("C150_4")) else "N/A")
            col8.metric("Median Earnings", f"${latest['MD_EARN_WNE_P10']:,.0f}" if pd.notna(latest.get("MD_EARN_WNE_P10")) else "N/A")

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
                    trend_col = [k for k, v in available_trends.items() if v == sel_trend][0]
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


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Group 5 Project — Analytics for Good** · "
    "Data: [U.S. DOE College Scorecard](https://collegescorecard.ed.gov/) · "
    "[Anthropic Labor Market Impact Study](https://www.anthropic.com/research/labor-market-impacts) · "
    "[BLS Employment Projections 2024–2034](https://www.bls.gov/emp/) · "
    "Built with Streamlit & Plotly"
)
