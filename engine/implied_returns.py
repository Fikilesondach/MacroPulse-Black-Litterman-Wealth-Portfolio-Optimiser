import pandas as pd
import numpy as np

def compute_delta(daily_returns, market_weights, risk_free_rate, periods=252):
    """
    Derive the market risk aversion coefficient from observable data.
    δ = (E[Rm] - Rf) / σ²m

    This is not a guess — it is extracted from the market itself.
    Higher delta = more risk averse market.
    Clamped to 0.5–10.0 to handle edge cases in short date ranges.
    """
    tickers = daily_returns.columns.tolist()
    w = np.array([market_weights.get(t, 0.0) for t in tickers])

    # Normalise weights in case they don't perfectly sum to 1
    w = w / w.sum()

    # Market portfolio return = weighted average of individual asset returns
    mkt_returns  = daily_returns.values @ w
    ann_return   = mkt_returns.mean() * periods
    ann_variance = mkt_returns.var()  * periods

    # Guard against division by zero if variance is somehow zero
    if ann_variance < 1e-8:
        return 2.5   # Return a sensible default

    delta = (ann_return - risk_free_rate) / ann_variance

    # Clamp to a sensible range
    delta = float(np.clip(delta, 0.5, 10.0))
    return delta


def get_implied_returns(prices, market_weights, cov_matrix,
                        risk_free_rate, daily_returns):
    """
    Reverse optimisation — extract the returns the market must be
    expecting given the current benchmark weights.

    Formula: Π = δ · Σ · w_market

    We compute this manually rather than relying on PyPortfolioOpt's
    market_implied_returns() which changed location across versions.

    Parameters
    ----------
    prices         : DataFrame of daily closing prices
    market_weights : dict of {ticker: weight} — the benchmark
    cov_matrix     : Ledoit-Wolf covariance matrix (DataFrame or ndarray)
    risk_free_rate : float — annualised risk-free rate
    daily_returns  : DataFrame of daily percentage returns

    Returns
    -------
    implied_returns : pd.Series — one implied return per asset
    delta           : float — the market risk aversion coefficient
    """
    tickers = daily_returns.columns.tolist()

    # ── Step 1: Compute delta ─────────────────────────────────────────
    delta = compute_delta(daily_returns, market_weights, risk_free_rate)

    # ── Step 2: Build the weight vector in ticker order ───────────────
    w = np.array([market_weights.get(t, 0.0) for t in tickers])
    w = w / w.sum()   # Normalise to exactly 1.0

    # ── Step 3: Extract covariance matrix as numpy array ──────────────
    # cov_matrix may be a DataFrame or ndarray depending on how it arrived
    if isinstance(cov_matrix, pd.DataFrame):
        # Reindex to match ticker order before converting
        S = cov_matrix.reindex(index=tickers, columns=tickers).values
    else:
        S = np.array(cov_matrix)

    # ── Step 4: Reverse optimisation formula Π = δ · Σ · w ───────────
    # S @ w  →  (n × n) @ (n,)  =  (n,)
    # Each element is the weighted covariance of asset i with the market
    Pi = delta * (S @ w)

    # ── Step 5: Package as a named Series ─────────────────────────────
    implied_returns = pd.Series(Pi, index=tickers, name="ImpliedReturn")

    return implied_returns, delta
