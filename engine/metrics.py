import numpy as np
import pandas as pd
from scipy import stats

def portfolio_returns(weights_dict, daily_returns):
    """Compute daily portfolio return stream from weights."""
    tickers = daily_returns.columns.tolist()
    w = np.array([weights_dict.get(t, 0.0) for t in tickers])
    return pd.Series(
        daily_returns.values @ w, 
        index=daily_returns.index
    )

def cumulative_returns(daily_ret):
    return (1 + daily_ret).cumprod()

def drawdown_series(cum_ret):
    peak = cum_ret.cummax()
    return (cum_ret / peak) - 1

def max_drawdown(cum_ret):
    return drawdown_series(cum_ret).min()

def annualised_return(daily_ret, periods=252):
    total   = (1 + daily_ret).prod()
    n_years = len(daily_ret) / periods
    return total ** (1 / n_years) - 1

def annualised_vol(daily_ret, periods=252):
    return daily_ret.std() * np.sqrt(periods)

def sharpe(daily_ret, rf, periods=252):
    excess = annualised_return(daily_ret, periods) - rf
    return excess / annualised_vol(daily_ret, periods)

def sortino(daily_ret, rf, periods=252):
    ann_ret  = annualised_return(daily_ret, periods) - rf
    neg_only = daily_ret[daily_ret < 0]
    downside = neg_only.std() * np.sqrt(periods)
    return ann_ret / downside

def calmar(daily_ret, periods=252):
    ann_ret = annualised_return(daily_ret, periods)
    mdd     = max_drawdown(cumulative_returns(daily_ret))
    return ann_ret / abs(mdd)

def information_ratio(port_ret, bench_ret, periods=252):
    active   = port_ret - bench_ret
    return (active.mean() * periods) / (active.std() * np.sqrt(periods))

def var_95(daily_ret):
    return np.percentile(daily_ret, 5)

def cvar_95(daily_ret):
    v    = var_95(daily_ret)
    tail = daily_ret[daily_ret <= v]
    return tail.mean()

def full_metrics(port_ret, bench_ret, rf):
    """Compute all nine institutional performance metrics."""
    cum = cumulative_returns(port_ret)
    return {
        "Annualised Return"  : annualised_return(port_ret),
        "Annualised Volatility": annualised_vol(port_ret),
        "Sharpe Ratio"       : sharpe(port_ret, rf),
        "Sortino Ratio"      : sortino(port_ret, rf),
        "Max Drawdown"       : max_drawdown(cum),
        "Calmar Ratio"       : calmar(port_ret),
        "Information Ratio"  : information_ratio(port_ret, bench_ret),
        "VaR 95%"            : var_95(port_ret),
        "CVaR 95%"           : cvar_95(port_ret),
    }
