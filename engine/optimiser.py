import numpy as np
import pandas as pd
from pypfopt import BlackLittermanModel, EfficientFrontier
from pypfopt.objective_functions import L2_reg

def run_black_litterman(cov_matrix, implied_returns, 
                        P, Q, view_confidences):
    """
    Blend market-implied returns (prior) with manager views (likelihood)
    to produce posterior expected returns.
    
    P               — pick matrix (which assets each view involves)
    Q               — view returns vector
    view_confidences — how confident we are in each view (0 to 1)
    """
    bl = BlackLittermanModel(
        cov_matrix       = cov_matrix,
        pi               = implied_returns,
        P                = P,
        Q                = Q,
        omega            = "idzorek",
        view_confidences = view_confidences
    )
    
    bl_returns = bl.bl_returns()
    bl_cov     = bl.bl_cov()
    
    return bl_returns, bl_cov

def optimise_portfolio(bl_returns, bl_cov, 
                       risk_free_rate, max_weight=0.20):
    """
    Find weights that maximise the Sharpe Ratio subject to:
    - Long-only (no short selling)
    - Maximum allocation per asset = max_weight
    - L2 regularisation to prevent corner solutions
    """
    ef = EfficientFrontier(
        expected_returns = bl_returns,
        cov_matrix       = bl_cov,
        weight_bounds    = (0, max_weight)
    )
    ef.add_objective(L2_reg, gamma=0.1)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    
    cleaned   = ef.clean_weights(cutoff=0.001)
    exp_ret, vol, sharpe = ef.portfolio_performance(
        risk_free_rate=risk_free_rate, verbose=False
    )
    
    return cleaned, exp_ret, vol, sharpe
