# ══════════════════════════════════════════════════════════════════════
# app.py — MacroPulse BL Portfolio Optimiser Dashboard
# Run with: streamlit run app.py
# ══════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta

# Import our engine modules
from engine.data          import fetch_prices, compute_covariance, compute_returns, TICKERS, ASSET_NAMES
from engine.implied_returns import get_implied_returns
from engine.optimiser     import run_black_litterman, optimise_portfolio
from engine.metrics       import (portfolio_returns, cumulative_returns,
                                   drawdown_series, full_metrics)

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "MacroPulse — BL Portfolio Optimiser",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS for professional styling ───────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        color: #2c3e50; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem; color: #7f8c8d; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; border-left: 4px solid #e67e22;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700;
        color: #2c3e50; border-bottom: 2px solid #e67e22;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — ALL USER INPUTS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/stocks-growth.png", width=60)
    st.markdown("## ⚙️ Model Configuration")
    st.markdown("---")

    # ── Date range ────────────────────────────────────────────────────
    st.markdown("### 📅 Historical Window")
    start_date = st.date_input(
        "Start Date",
        value=date(2019, 1, 1),
        min_value=date(2015, 1, 1),
        max_value=date.today() - timedelta(days=365)
    )
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        min_value=start_date + timedelta(days=365),
        max_value=date.today()
    )

    st.markdown("---")

    # ── Risk parameters ───────────────────────────────────────────────
    st.markdown("### 🎯 Optimisation Parameters")
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0, max_value=10.0,
        value=3.5, step=0.25,
        help="Annualised risk-free rate used in Sharpe Ratio calculation. "
             "Use the current SARB repo rate or US 10Y Treasury yield."
    ) / 100

    max_weight = st.slider(
        "Max Weight Per Asset (%)",
        min_value=10, max_value=40,
        value=20, step=5,
        help="Maximum allocation to any single asset. "
             "Lower = more diversified. Higher = more concentrated."
    ) / 100

    st.markdown("---")

    # ── Market-cap benchmark weights ──────────────────────────────────
    st.markdown("### ⚖️ Benchmark Weights")
    st.caption("Adjust the market-cap benchmark used for reverse optimisation.")

    npn_w = st.slider("Naspers (NPN.JO)",        5, 30, 20, 1)
    fsr_w = st.slider("FirstRand (FSR.JO)",       5, 20, 10, 1)
    agl_w = st.slider("Anglo American (AGL.JO)",  5, 20, 10, 1)
    sol_w = st.slider("Sasol (SOL.JO)",            2, 15,  7, 1)
    shp_w = st.slider("Shoprite (SHP.JO)",         2, 15,  7, 1)
    spy_w = st.slider("S&P 500 (SPY)",             5, 30, 18, 1)
    qqq_w = st.slider("Nasdaq 100 (QQQ)",          5, 25, 10, 1)
    eem_w = st.slider("Emerging Mkts (EEM)",       2, 15,  8, 1)
    gld_w = st.slider("Gold (GLD)",                2, 15,  5, 1)
    tlt_w = st.slider("US Bonds (TLT)",            2, 15,  5, 1)

    raw_weights = {
        "NPN.JO": npn_w, "FSR.JO": fsr_w, "AGL.JO": agl_w,
        "SOL.JO": sol_w, "SHP.JO": shp_w, "SPY":    spy_w,
        "QQQ":    qqq_w, "EEM":    eem_w, "GLD":    gld_w,
        "TLT":    tlt_w,
    }

    # Normalise to sum to 1.0
    total = sum(raw_weights.values())
    market_weights = {k: v / total for k, v in raw_weights.items()}

    weight_sum = sum(market_weights.values())
    if abs(weight_sum - 1.0) < 0.001:
        st.success(f"✓ Weights normalised to 100%")
    else:
        st.error(f"Weight error: {weight_sum:.2%}")

    st.markdown("---")

    # ── Manager views ─────────────────────────────────────────────────
    st.markdown("### 💡 Manager Views")
    st.caption("Your active views on top of the market prior.")

    st.markdown("**View 1: FirstRand vs Naspers**")
    view1_return = st.slider(
        "FSR outperforms NPN by (%)", -5.0, 10.0, 2.0, 0.5,
        key="v1_ret"
    )
    view1_conf = st.slider(
        "Confidence (%)", 10, 90, 60, 10, key="v1_conf"
    )

    st.markdown("**View 2: Gold vs Long Bonds**")
    view2_return = st.slider(
        "GLD outperforms TLT by (%)", -5.0, 10.0, 1.5, 0.5,
        key="v2_ret"
    )
    view2_conf = st.slider(
        "Confidence (%)", 10, 90, 50, 10, key="v2_conf"
    )

    st.markdown("**View 3: Emerging Markets Absolute**")
    view3_return = st.slider(
        "EEM absolute return (%)", -5.0, 15.0, 5.0, 0.5,
        key="v3_ret"
    )
    view3_conf = st.slider(
        "Confidence (%)", 10, 90, 40, 10, key="v3_conf"
    )

    st.markdown("---")
    # Add this just above the run_button line in the sidebar
    if st.button("🗑️ Clear Data Cache", use_container_width=True, type="secondary"):
        fetch_prices.clear()
        st.success("Cache cleared — next run will re-download fresh data.")

    run_button = st.button("🚀 Run Optimisation", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL — HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">📊 MacroPulse BL Portfolio Optimiser</p>', 
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-Factor Black-Litterman Wealth Portfolio Engine '
    '— Live Market Data · Ledoit-Wolf Covariance · Efficient Frontier Optimisation</p>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════
# MODEL EXECUTION
# ══════════════════════════════════════════════════════════════════════
if run_button:

    # ── Progress indicators ────────────────────────────────────────────
    progress = st.progress(0)
    status   = st.status("Initialising model...", expanded=True)

    try:
        with status:

            # ── Step 1: Fetch live data ────────────────────────────────
            st.write("📡 Fetching market data (cached for 1 hour after first load)...")
            prices = fetch_prices(
                tuple(TICKERS),        # Must be tuple not list for cache key to work
                str(start_date),
                str(end_date)
            )
            
            st.write(f"✓ Data loaded: {prices.shape[0]} days × {prices.shape[1]} assets")
            st.write(f"✓ Prices shape: {prices.shape[0]} days × {prices.shape[1]} assets")
            st.write(f"✓ Date range in data: {prices.index[0].date()} → {prices.index[-1].date()}")

            daily_ret  = compute_returns(prices)
            cov_matrix = compute_covariance(prices)
            progress.progress(25)

            # ── Step 2: Market-implied returns ─────────────────────────
            st.write("🔄 Running reverse optimisation (market-implied returns)...")
            implied_returns, delta = get_implied_returns(
                prices, market_weights, cov_matrix, risk_free_rate, daily_ret
            )
            progress.progress(50)

            # ── Step 3: BL views and optimisation ─────────────────────
            st.write("⚙️ Running Black-Litterman model with your views...")
            
            n_assets = len(TICKERS)
            idx      = {t: i for i, t in enumerate(TICKERS)}
            
            P = np.zeros((3, n_assets))
            P[0, idx["FSR.JO"]] = +1;  P[0, idx["NPN.JO"]] = -1
            P[1, idx["GLD"]]    = +1;  P[1, idx["TLT"]]    = -1
            P[2, idx["EEM"]]    = +1

            Q = np.array([
                view1_return / 100,
                view2_return / 100,
                view3_return / 100
            ])

            confidences = [
                view1_conf / 100,
                view2_conf / 100,
                view3_conf / 100
            ]

            bl_returns, bl_cov = run_black_litterman(
                cov_matrix, implied_returns, P, Q, confidences
            )
            
            cleaned_weights, exp_ret, vol, sharpe_val = optimise_portfolio(
                bl_returns, bl_cov, risk_free_rate, max_weight
            )
            progress.progress(75)

            # ── Step 4: Performance metrics ───────────────────────────
            st.write("📈 Computing performance metrics and benchmarks...")

            eq_weights  = {t: 1/n_assets for t in TICKERS}
            
            bl_port_ret  = portfolio_returns(cleaned_weights, daily_ret)
            eq_port_ret  = portfolio_returns(eq_weights,      daily_ret)
            mkt_port_ret = portfolio_returns(market_weights,  daily_ret)

            bl_metrics  = full_metrics(bl_port_ret,  eq_port_ret, risk_free_rate)
            eq_metrics  = full_metrics(eq_port_ret,  eq_port_ret, risk_free_rate)
            mkt_metrics = full_metrics(mkt_port_ret, eq_port_ret, risk_free_rate)

            bl_cum  = cumulative_returns(bl_port_ret)
            eq_cum  = cumulative_returns(eq_port_ret)
            mkt_cum = cumulative_returns(mkt_port_ret)

            bl_dd  = drawdown_series(bl_cum)
            eq_dd  = drawdown_series(eq_cum)
            mkt_dd = drawdown_series(mkt_cum)

            progress.progress(100)

        status.update(label="✅ Optimisation complete!", state="complete")

        # ══════════════════════════════════════════════════════════════
        # SECTION 1 — TOP KPI METRICS
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<p class="section-title">Portfolio Performance at a Glance</p>',
                    unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric(
            "Annualised Return",
            f"{bl_metrics['Annualised Return']:.2%}",
            delta=f"{(bl_metrics['Annualised Return'] - eq_metrics['Annualised Return']):.2%} vs Equal Weight"
        )
        k2.metric(
            "Sharpe Ratio",
            f"{bl_metrics['Sharpe Ratio']:.3f}",
            delta=f"{(bl_metrics['Sharpe Ratio'] - eq_metrics['Sharpe Ratio']):.3f} vs Equal Weight"
        )
        k3.metric(
            "Max Drawdown",
            f"{bl_metrics['Max Drawdown']:.2%}",
            delta=f"{(bl_metrics['Max Drawdown'] - eq_metrics['Max Drawdown']):.2%} vs Equal Weight",
            delta_color="inverse"
        )
        k4.metric(
            "Sortino Ratio",
            f"{bl_metrics['Sortino Ratio']:.3f}"
        )
        k5.metric(
            "Annualised Volatility",
            f"{bl_metrics['Annualised Volatility']:.2%}"
        )

        # ══════════════════════════════════════════════════════════════
        # SECTION 2 — CUMULATIVE RETURNS CHART
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<p class="section-title">Cumulative Growth of R1 Invested</p>',
                    unsafe_allow_html=True)

        fig_cum = go.Figure()

        fig_cum.add_trace(go.Scatter(
            x=bl_cum.index,  y=bl_cum,
            name="BL Optimised",  line=dict(color="#e67e22", width=2.5),
            hovertemplate="<b>BL Optimised</b><br>Date: %{x}<br>Value: R%{y:.4f}<extra></extra>"
        ))
        fig_cum.add_trace(go.Scatter(
            x=eq_cum.index,  y=eq_cum,
            name="Equal Weight",  line=dict(color="#3498db", width=1.8, dash="dash"),
            hovertemplate="<b>Equal Weight</b><br>Date: %{x}<br>Value: R%{y:.4f}<extra></extra>"
        ))
        fig_cum.add_trace(go.Scatter(
            x=mkt_cum.index, y=mkt_cum,
            name="Market Weight", line=dict(color="#2ecc71", width=1.8, dash="dot"),
            hovertemplate="<b>Market Weight</b><br>Date: %{x}<br>Value: R%{y:.4f}<extra></extra>"
        ))
        fig_cum.add_hline(y=1.0, line_dash="dash", 
                          line_color="grey", opacity=0.5, 
                          annotation_text="Break-even (R1.00)")
        fig_cum.update_layout(
            height=420, template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (R1 = start)",
            yaxis_tickprefix="R",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ══════════════════════════════════════════════════════════════
        # SECTION 3 — WEIGHTS + DRAWDOWN (SIDE BY SIDE)
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<p class="section-title">Optimised Portfolio Weights</p>',
                        unsafe_allow_html=True)

            active = {ASSET_NAMES[t]: w 
                      for t, w in cleaned_weights.items() if w > 0.001}

            fig_pie = go.Figure(go.Pie(
                labels=list(active.keys()),
                values=list(active.values()),
                hole=0.45,
                textinfo="label+percent",
                textposition="outside",
                marker=dict(colors=px.colors.qualitative.Set2),
                hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>"
            ))
            fig_pie.update_layout(
                height=400, template="plotly_white",
                showlegend=False,
                annotations=[dict(
                    text=f"Max Sharpe<br>{sharpe_val:.3f}",
                    x=0.5, y=0.5, font_size=14,
                    showarrow=False, font_color="#2c3e50"
                )]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.markdown('<p class="section-title">Portfolio Drawdown Over Time</p>',
                        unsafe_allow_html=True)

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=bl_dd.index, y=bl_dd * 100,
                name="BL Optimised", fill="tozeroy",
                line=dict(color="#e67e22"),
                fillcolor="rgba(230,126,34,0.3)",
                hovertemplate="<b>BL</b><br>%{y:.2f}%<extra></extra>"
            ))
            fig_dd.add_trace(go.Scatter(
                x=eq_dd.index, y=eq_dd * 100,
                name="Equal Weight",
                line=dict(color="#3498db", dash="dash"),
                hovertemplate="<b>Equal</b><br>%{y:.2f}%<extra></extra>"
            ))
            fig_dd.update_layout(
                height=400, template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Drawdown from Peak (%)",
                yaxis_ticksuffix="%",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # ══════════════════════════════════════════════════════════════
        # SECTION 4 — PRIOR VS POSTERIOR RETURNS
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<p class="section-title">How Your Views Shifted the Market Consensus</p>',
                    unsafe_allow_html=True)

        tickers_ordered = TICKERS
        names_ordered   = [ASSET_NAMES[t] for t in tickers_ordered]
        prior_vals      = [implied_returns.get(t, 0) * 100 for t in tickers_ordered]
        posterior_vals  = [bl_returns.get(t, 0) * 100 for t in tickers_ordered]

        fig_view = go.Figure()
        fig_view.add_trace(go.Bar(
            name="Market-Implied (Prior)",
            x=names_ordered, y=prior_vals,
            marker_color="#3498db", opacity=0.85,
            hovertemplate="<b>Prior</b>: %{y:.2f}%<extra></extra>"
        ))
        fig_view.add_trace(go.Bar(
            name="Black-Litterman (Posterior)",
            x=names_ordered, y=posterior_vals,
            marker_color="#e67e22", opacity=0.85,
            hovertemplate="<b>Posterior</b>: %{y:.2f}%<extra></extra>"
        ))
        fig_view.add_hline(y=0, line_color="grey", line_width=1)
        fig_view.update_layout(
            barmode="group", height=380,
            template="plotly_white",
            xaxis_title="Asset",
            yaxis_title="Annualised Expected Return (%)",
            yaxis_ticksuffix="%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig_view, use_container_width=True)

        # ══════════════════════════════════════════════════════════════
        # SECTION 5 — FULL METRICS TABLE
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<p class="section-title">Full Performance Metrics — Institutional Report</p>',
                    unsafe_allow_html=True)

        metric_labels = {
            "Annualised Return"    : "{:.2%}",
            "Annualised Volatility": "{:.2%}",
            "Sharpe Ratio"         : "{:.4f}",
            "Sortino Ratio"        : "{:.4f}",
            "Max Drawdown"         : "{:.2%}",
            "Calmar Ratio"         : "{:.4f}",
            "Information Ratio"    : "{:.4f}",
            "VaR 95%"              : "{:.4%}",
            "CVaR 95%"             : "{:.4%}",
        }

        rows = []
        for metric, fmt in metric_labels.items():
            rows.append({
                "Metric"        : metric,
                "BL Optimised"  : fmt.format(bl_metrics[metric]),
                "Equal Weight"  : fmt.format(eq_metrics[metric]),
                "Market Weight" : fmt.format(mkt_metrics[metric]),
            })

        metrics_df = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(
            metrics_df,
            use_container_width=True,
            height=360
        )

        # ══════════════════════════════════════════════════════════════
        # SECTION 6 — MONTHLY RETURNS HEATMAP
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<p class="section-title">BL Portfolio — Monthly Returns Heatmap</p>',
                    unsafe_allow_html=True)

        monthly = (1 + bl_port_ret).resample("ME").prod() - 1
        monthly.index = pd.MultiIndex.from_arrays([
            monthly.index.year, monthly.index.month
        ])
        pivot = monthly.unstack(level=1)
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]

        fig_heat = go.Figure(go.Heatmap(
            z       = pivot.values * 100,
            x       = pivot.columns.tolist(),
            y       = pivot.index.tolist(),
            colorscale = "RdYlGn",
            zmid    = 0,
            text    = np.round(pivot.values * 100, 1),
            texttemplate = "%{text}%",
            hovertemplate = "Month: %{x}<br>Year: %{y}<br>Return: %{z:.2f}%<extra></extra>",
            colorbar = dict(title="Return (%)", ticksuffix="%")
        ))
        fig_heat.update_layout(
            height=350, template="plotly_white",
            xaxis_title="Month", yaxis_title="Year",
            yaxis=dict(tickmode="linear")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ══════════════════════════════════════════════════════════════
        # SECTION 7 — MODEL PARAMETERS SUMMARY
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<p class="section-title">Model Parameters & Methodology</p>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.info(f"""
**Optimisation**
- Target: Max Sharpe Ratio
- Constraint: Long-only
- Max weight: {max_weight:.0%} per asset
- Regularisation: L2 (γ = 0.1)
        """)
        c2.info(f"""
**Covariance Estimation**
- Method: Ledoit-Wolf Shrinkage
- Risk aversion (δ): {delta:.4f}
- Risk-free rate: {risk_free_rate:.2%}
- Period: {start_date} → {end_date}
        """)
        c3.info(f"""
**Active Views**
- View 1: FSR vs NPN → {view1_return:+.1f}% @ {view1_conf}% conf.
- View 2: GLD vs TLT → {view2_return:+.1f}% @ {view2_conf}% conf.
- View 3: EEM abs   → {view3_return:+.1f}% @ {view3_conf}% conf.
        """)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

else:
    # ── Landing state — before the user clicks Run ────────────────────
    st.info("""
    👈 **Configure your model parameters in the sidebar, then click Run Optimisation.**
    
    This dashboard will:
    1. Fetch live market data from Yahoo Finance
    2. Compute Ledoit-Wolf shrinkage covariance matrix
    3. Extract market-implied equilibrium returns via reverse optimisation
    4. Blend with your active views using the Black-Litterman formula
    5. Optimise for maximum Sharpe Ratio under your constraints
    6. Generate a full institutional performance report
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**📊 Black-Litterman Model**\n\nAnchors to market equilibrium. Blends with active views. Avoids extreme Markowitz weights.")
    col2.markdown("**🔬 Ledoit-Wolf Covariance**\n\nRegularised covariance estimation. Robust to market shocks. Outperforms sample covariance out-of-sample.")
    col3.markdown("**📈 9 Institutional Metrics**\n\nSharpe, Sortino, Calmar, IR, Max Drawdown, VaR, CVaR and more — the same metrics used by Morningstar and Bloomberg.")
