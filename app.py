"""
College Scorecard Explorer
Interactive dashboard for analyzing U.S. higher education data (2020-2024)
Group 5 Project — Analytics for Good
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "💰 Earnings & ROI", "🎯 Admissions", "🗺️ Geographic", "🔍 Institution Lookup"
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
# TAB 5: INSTITUTION LOOKUP
# ════════════════════════════════════════════════════════════════════════════
with tab5:
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
    "**Group 5 Project** · Data: [U.S. Department of Education College Scorecard]"
    "(https://collegescorecard.ed.gov/) · "
    "Built with Streamlit & Plotly"
)
