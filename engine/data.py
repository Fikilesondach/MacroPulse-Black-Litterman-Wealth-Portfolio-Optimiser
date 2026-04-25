import pandas as pd
import numpy as np
import yfinance as yf
import time
import streamlit as st
from pypfopt import risk_models

# ── Asset universe ─────────────────────────────────────────────────────
TICKERS = [
    "NPN.JO", "FSR.JO", "AGL.JO", "SOL.JO", "SHP.JO",
    "SPY", "QQQ", "EEM", "GLD", "TLT"
]

ASSET_NAMES = {
    "NPN.JO": "Naspers",        "FSR.JO": "FirstRand",
    "AGL.JO": "Anglo American", "SOL.JO": "Sasol",
    "SHP.JO": "Shoprite",       "SPY":    "S&P 500",
    "QQQ":    "Nasdaq 100",     "EEM":    "Emerging Markets",
    "GLD":    "Gold",           "TLT":    "US Long Bonds",
}


def _download_single_ticker(ticker, start_str, end_str, retries=3):
    """
    Download one ticker at a time with retries.
    
    Downloading individually rather than in bulk is gentler on
    Yahoo Finance's rate limiter. A small sleep between each 
    ticker prevents triggering the rate limit at all.
    """
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                ticker,
                start       = start_str,
                end         = end_str,
                auto_adjust = True,
                progress    = False,
            )

            if raw is None or raw.empty:
                print(f"  [{ticker}] Attempt {attempt}: empty response")
                if attempt < retries:
                    time.sleep(5 * attempt)
                continue

            # Extract Close column — single ticker returns flat columns
            if "Close" in raw.columns:
                series = raw["Close"].copy()
                series.name = ticker
                print(f"  [{ticker}] ✓ {len(series)} rows")
                return series

            # Handle MultiIndex from single ticker (rare but possible)
            if isinstance(raw.columns, pd.MultiIndex):
                level_0 = raw.columns.get_level_values(0).unique().tolist()
                if "Close" in level_0:
                    series = raw["Close"].squeeze()
                    series.name = ticker
                    return series

            print(f"  [{ticker}] Attempt {attempt}: no Close column found")

        except Exception as e:
            print(f"  [{ticker}] Attempt {attempt} error: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)

    # Return None if all retries failed — caller handles missing tickers
    print(f"  [{ticker}] ✗ All {retries} attempts failed")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, start_date, end_date):
    """
    Download and clean daily closing prices from Yahoo Finance.

    KEY IMPROVEMENT — st.cache_data:
        Streamlit caches this function's output for 1 hour (ttl=3600 seconds).
        If you click Run Optimisation multiple times with the same date range,
        the data is only downloaded ONCE from Yahoo Finance and reused from
        cache every subsequent time. This completely prevents rate limiting
        from repeated button clicks.

    KEY IMPROVEMENT — individual ticker downloads:
        Instead of requesting all 10 tickers simultaneously (which looks like
        a scraping attack to Yahoo's servers), we download one at a time with
        a small pause between each. Much friendlier to the rate limiter.
    """
    start_str = str(start_date)
    end_str   = str(end_date)

    print(f"\nfetch_prices called (will cache result for 1 hour):")
    print(f"  start : {start_str}")
    print(f"  end   : {end_str}")
    print(f"  assets: {tickers}")

    if start_str >= end_str:
        raise ValueError(
            f"Start date ({start_str}) must be before end date ({end_str})."
        )

    # ── Download each ticker individually ──────────────────────────────
    series_list = []
    failed      = []

    for i, ticker in enumerate(tickers):
        print(f"\nDownloading {ticker} ({i+1}/{len(tickers)})...")
        series = _download_single_ticker(ticker, start_str, end_str)

        if series is not None and len(series) > 0:
            series_list.append(series)
        else:
            failed.append(ticker)

        # Polite pause between requests — prevents rate limiting
        # Skip pause after the last ticker
        if i < len(tickers) - 1:
            time.sleep(1.5)

    # ── Report what succeeded and what failed ─────────────────────────
    print(f"\nDownload summary:")
    print(f"  Succeeded : {[s.name for s in series_list]}")
    print(f"  Failed    : {failed}")

    if not series_list:
        raise ValueError(
            "Yahoo Finance returned no data for any ticker.\n\n"
            "You are still rate limited. Please:\n"
            "  1. Wait 15 minutes before trying again\n"
            "  2. Do not click Run Optimisation repeatedly\n"
            "  3. Each successful run is cached for 1 hour — "
            "     clicking again within that hour uses cached data"
        )

    if failed:
        print(f"Warning: {len(failed)} tickers failed and will be excluded: {failed}")

    # ── Combine into a single DataFrame ───────────────────────────────
    prices = pd.concat(series_list, axis=1)

    # ── Forward-fill non-trading days and clean ───────────────────────
    prices = prices.ffill().dropna(how="all")

    # ── Minimum data check ────────────────────────────────────────────
    if len(prices) < 30:
        raise ValueError(
            f"Only {len(prices)} rows after cleaning — need at least 30. "
            f"Widen your date range."
        )

    print(f"\n✓ Final prices: {prices.shape[0]} days × {prices.shape[1]} assets")
    print(f"✓ Range: {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices


def compute_covariance(prices):
    """Ledoit-Wolf shrinkage covariance matrix."""
    clean = prices.dropna(axis=1, how="all")
    if clean.shape[0] < 30:
        raise ValueError(f"Need ≥30 rows, got {clean.shape[0]}.")
    if clean.shape[1] < 2:
        raise ValueError("Need ≥2 assets.")
    return risk_models.CovarianceShrinkage(clean).ledoit_wolf()


def compute_returns(prices):
    """Daily percentage returns."""
    returns = prices.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Returns DataFrame is empty after cleaning.")
    return returns
