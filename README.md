# MacroPulse — Multi-Factor Black-Litterman Wealth Portfolio Optimiser

> A live, interactive quantitative portfolio construction engine built on the Black-Litterman framework. Integrates Ledoit-Wolf covariance shrinkage with real-time market data to generate institutional-grade, macro-aware asset allocations across South African and global equity markets.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://macropulse-black-litterman-wealth-portfolio-optimiser-u6kqq2xz.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 🚀 Live Demo

**[→ Open the Live Dashboard](https://macropulse-black-litterman-wealth-portfolio-optimiser-u6kqq2xz.streamlit.app)**

The dashboard runs a full Black-Litterman optimisation pipeline in real time:
- Fetches live market data from Yahoo Finance on every run
- Computes a Ledoit-Wolf shrinkage covariance matrix
- Extracts market-implied equilibrium returns via reverse optimisation
- Blends market consensus with your active investment views
- Optimises for maximum Sharpe Ratio under configurable constraints
- Generates a full institutional performance report with 9 risk metrics

---

## The Business Problem

Retail and institutional wealth clients require customised portfolios that maximise risk-adjusted returns while respecting individual risk appetites. Traditional Markowitz mean-variance optimisation fails in practice because it is acutely sensitive to historical return estimates — small changes in input assumptions produce wildly unstable and unrealistic portfolio weights.

This project solves that problem by implementing the **Black-Litterman model**, which anchors portfolio construction to a stable market equilibrium prior and allows a portfolio manager to inject explicit, confidence-weighted views on top of that prior. The result is a portfolio that is simultaneously grounded in market consensus and responsive to active investment views — exactly the kind of model used by quantitative teams at firms like Goldman Sachs, where Black-Litterman was originally developed in 1990.

---

## Why Black-Litterman

Standard mean-variance optimisation has three well-documented failure modes:

| Problem | Effect | BL Solution |
|---|---|---|
| Noisy historical return estimates | Extreme, unstable weights | Uses market-implied equilibrium returns as prior |
| Garbage-in-garbage-out | Small input changes → massive weight changes | Bayesian blending stabilises the posterior |
| Ignores market consensus | Portfolio disconnected from reality | Reverse optimisation extracts collective market wisdom |
| No mechanism for views | Cannot incorporate active research | P/Q/Omega framework encodes manager views formally |

---

## Dashboard Features

The live dashboard exposes every model parameter as an interactive control:

**Sidebar Controls**
- Historical window — configurable start and end date
- Risk-free rate — adjustable for different rate environments (SARB repo rate or US 10Y)
- Maximum weight per asset — controls concentration vs diversification
- Benchmark weights — adjust the market-cap prior for each of the 10 assets
- Active views — set return expectations and confidence for three manager views

**Output Panels**
- KPI metrics bar — Annualised Return, Sharpe, Max Drawdown, Sortino, Volatility
- Cumulative growth chart — BL Optimised vs Equal Weight vs Market-Cap benchmark
- Portfolio weights donut chart — with Sharpe Ratio displayed at centre
- Drawdown over time — both portfolios overlaid
- Prior vs Posterior returns — how your views shifted the market consensus
- Full metrics table — all 9 institutional metrics across all 3 portfolios
- Monthly returns heatmap — full year-by-month breakdown

---

## Project Architecture

The project is structured as four research notebooks and a live application layer:

```
Step 1: Data Preparation          → notebooks/01_data_preparation.ipynb
    └── Live Yahoo Finance data
    └── Ledoit-Wolf covariance matrix
    └── Correlation heatmap

Step 2: Market-Implied Returns    → notebooks/02_implied_returns.ipynb
    └── Risk aversion coefficient (δ)
    └── Reverse optimisation: Π = δ · Σ · w_mkt

Step 3: BL Optimisation          → notebooks/03_bl_optimisation.ipynb
    └── Manager views (P matrix, Q vector, Omega)
    └── BL posterior expected returns
    └── Efficient Frontier: max Sharpe, long-only, ≤20% per asset

Step 4: Performance Report        → notebooks/04_performance_report.ipynb
    └── Backtest vs benchmarks
    └── Drawdown, Sortino, Calmar, IR, VaR, CVaR
    └── Monthly returns heatmap

Live Dashboard                    → app.py + engine/
    └── All of the above, live, interactive, deployed
```

---

## Methodology

### Step 1 — Data Preparation & Ledoit-Wolf Covariance

Daily closing prices are downloaded live via Yahoo Finance for a blended universe of South African Top 40 constituents and global ETFs. Missing values caused by non-overlapping trading calendars (JSE vs NYSE) are forward-filled using the last known price.

The covariance matrix is estimated using **Ledoit-Wolf shrinkage** rather than the standard sample covariance. Ledoit-Wolf computes a convex combination of the sample covariance and a structured target:

```
Σ_shrunk = (1 - α) · Σ_sample + α · Σ_target
```

where α is determined analytically to minimise expected out-of-sample estimation error. This produces a covariance matrix that is better conditioned and more stable across market regimes.

### Step 2 — Reverse Optimisation (Market-Implied Returns)

Rather than using historical average returns as expected return inputs, the model applies **reverse optimisation** to extract the returns implied by the current market-cap benchmark weights:

```
Π = δ · Σ · w_market
```

Where:
- **Π** — vector of market-implied equilibrium returns
- **δ** — market risk aversion: `δ = (E[Rm] - Rf) / σ²m`
- **Σ** — Ledoit-Wolf covariance matrix
- **w_market** — benchmark weights (configurable in sidebar)

### Step 3 — Black-Litterman Views and Optimisation

Three active views are encoded using the standard Black-Litterman framework:

| View | Type | Assets | Default Return | Default Confidence |
|---|---|---|---|---|
| SA Financials vs SA Tech | Relative | FSR.JO vs NPN.JO | +2.0% | 60% |
| Gold vs Long Bonds | Relative | GLD vs TLT | +1.5% | 50% |
| Emerging Markets | Absolute | EEM | +5.0% | 40% |

All view parameters are adjustable in real time via the dashboard sidebar.

The Black-Litterman formula produces posterior expected returns:

```
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
```

These are fed into an **Efficient Frontier** optimiser targeting maximum Sharpe Ratio under long-only and maximum weight constraints, with L2 regularisation to prevent corner solutions.

### Step 4 — Performance Reporting

Nine institutional metrics computed across three portfolios:

| Metric | Definition |
|---|---|
| Sharpe Ratio | Return per unit of total volatility |
| Sortino Ratio | Return per unit of downside volatility only |
| Calmar Ratio | Annualised return ÷ maximum drawdown |
| Information Ratio | Consistency of active return vs benchmark |
| Max Drawdown | Worst peak-to-trough loss in the period |
| VaR 95% | Daily loss not exceeded on 95% of trading days |
| CVaR 95% | Average loss on the worst 5% of trading days |

---

## Asset Universe

| Asset | Ticker | Role |
|---|---|---|
| Naspers | NPN.JO | SA Tech / Consumer |
| FirstRand | FSR.JO | SA Financials |
| Anglo American | AGL.JO | SA Resources |
| Sasol | SOL.JO | SA Energy |
| Shoprite | SHP.JO | SA Consumer Staples |
| S&P 500 ETF | SPY | Global Equities |
| Nasdaq 100 ETF | QQQ | Global Technology |
| Emerging Markets ETF | EEM | EM Diversification |
| Gold ETF | GLD | Inflation Hedge |
| US Long Bonds ETF | TLT | Fixed Income |

---

## Tech Stack

| Category | Library | Purpose |
|---|---|---|
| Data collection | `yfinance` | Live daily prices — JSE and global |
| Data manipulation | `pandas`, `numpy` | DataFrames, matrix algebra |
| Covariance estimation | `PyPortfolioOpt` | Ledoit-Wolf shrinkage |
| Portfolio optimisation | `PyPortfolioOpt` | Black-Litterman, Efficient Frontier |
| Statistical analysis | `scipy` | Return distribution, skewness, kurtosis |
| Visualisation | `plotly` | Interactive charts in dashboard |
| Dashboard | `streamlit` | Live web application |
| Macroeconomic data | `fredapi` | Risk-free rate, yield data |

---

## Run Locally

### Prerequisites
- Python 3.8 or higher

### Installation

```bash
git clone https://github.com/Fikilesondach/MacroPulse-Black-Litterman-Wealth-Portfolio-Optimiser.git
cd MacroPulse-Black-Litterman-Wealth-Portfolio-Optimiser/MacroPulse_BL
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Configure parameters in the sidebar and click **Run Optimisation**.

> **Note on data caching:** The first run downloads live data from Yahoo Finance (~20 seconds). All subsequent runs within the same hour use cached data and are near-instant. If Yahoo Finance rate limits your IP, wait 15 minutes before retrying.

### Run the Research Notebooks

```bash
jupyter notebook
```

Run notebooks in order: `01` → `02` → `03` → `04`.

---

## Project Structure

```
MacroPulse_BL/
│
├── app.py                          # Live Streamlit dashboard
│
├── engine/                         # Modular model layer
│   ├── __init__.py
│   ├── data.py                     # Live data fetching with retry + caching
│   ├── implied_returns.py          # Reverse optimisation
│   ├── optimiser.py                # Black-Litterman + Efficient Frontier
│   └── metrics.py                  # 9 institutional performance metrics
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_implied_returns.ipynb
│   ├── 03_bl_optimisation.ipynb
│   └── 04_performance_report.ipynb
│
└── requirements.txt
```

---

## Key Concepts Explained

**Why not just use historical average returns?**
Historical returns are extremely noisy. A single strong or weak year dramatically shifts the average, causing the optimiser to concentrate heavily in recently lucky assets. The Black-Litterman prior anchors to market equilibrium, which is far more stable.

**What does market-implied return actually mean?**
The return that makes the current benchmark weights mathematically optimal under our risk model. It represents the market's embedded consensus — not any individual's forecast.

**Why Ledoit-Wolf instead of the standard covariance matrix?**
The standard sample covariance overfits to the specific sample period. During COVID (March 2020), correlations spiked toward 1.0 across all assets. Ledoit-Wolf shrinkage regularises this, producing a more stable out-of-sample estimate.

**Why use Sortino instead of just Sharpe?**
Sharpe penalises all volatility — both upside and downside. Wealth clients do not object to rising sharply. They object to falling sharply. Sortino isolates downside deviation, making it more appropriate for client-facing portfolio management.

---

## Real-World Applications

**Wealth Management** — Systematic portfolio construction for HNW clients with explicit risk constraints and customisable manager views per client segment.

**Bank Treasury / ALM** — Equity exposure management and tactical allocation, with macro overlay from rate and inflation signals.

**Pension / Insurance Funds** — Liability-aware allocation where BL views can encode actuarial assumptions about long-run economic conditions.

**Quantitative Research** — Foundation for factor-based extensions incorporating momentum, value, quality, and low-volatility signals as structured views.

**Risk Management** — Drawdown, VaR, and CVaR outputs serve as early warning indicators for portfolio stress, informing hedging and position sizing decisions.

---

## Roadmap

- [ ] SARB repo rate integration for South African risk-free rate
- [ ] FRED macro overlay — rate hike and inflation regime shading on charts
- [ ] Rolling re-optimisation to simulate quarterly rebalancing
- [ ] Factor model extension — momentum, value, quality as additional views
- [ ] Efficient frontier visualisation — full risk-return curve
- [ ] Monte Carlo simulation for forward-looking scenario analysis
- [ ] Paid data provider integration (Polygon.io) for production reliability

---

## Known Limitations

**Yahoo Finance Rate Limiting**
The live engine uses Yahoo Finance's free API. Yahoo Finance rate limits IPs that make frequent requests. If you encounter a rate limit error, wait 10–15 minutes and try again. Subsequent runs within the same session are cached for 1 hour. A production deployment would use a paid provider such as Bloomberg, Refinitiv, or Polygon.io.

---

## Disclaimer

This project is built for educational and portfolio demonstration purposes. It does not constitute financial advice. Past performance simulated in backtests does not guarantee future results. All investment decisions should be made in consultation with a qualified financial adviser.

---

*Built with Python · Powered by PyPortfolioOpt · Data via Yahoo Finance · Deployed on Streamlit Cloud*
