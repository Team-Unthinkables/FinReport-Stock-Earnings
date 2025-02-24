import pandas as pd
import numpy as np
from arch import arch_model

def compute_volatility(returns_series, dist='Normal'):
    """
    Fit an EGARCH(1,1) model on the returns series and return volatility metrics.
    
    Parameters:
    -----------
    returns_series : pd.Series
        A time series of returns.
    
    Returns:
    --------
    last_volatility : float
        The last estimated conditional volatility.
    var_95 : float
        An approximate 95% Value at Risk (VaR) computed using the conditional volatility.
    """
    # Ensure returns_series is a pandas Series
    if not isinstance(returns_series, pd.Series):
        returns_series = pd.Series(returns_series)
    
    # Clean the returns series
    returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    if returns_series.empty:
        raise ValueError("Returns series is empty after cleaning.")
    
    # Fit an EGARCH(1,1) model; adjust parameters as needed.
    model = arch_model(returns_series, vol='EGARCH', p=1, o=0, q=1, dist=dist, rescale=False)
    res = model.fit(disp="off")
    
    # Get the latest conditional volatility estimate.
    last_volatility = res.conditional_volatility.iloc[-1]
    # For a Normal distribution, the 95% VaR is approximately 1.65 * volatility (this is a rough estimate).
    var_95 = last_volatility * 1.65
    return last_volatility, var_95

def compute_max_drawdown(returns_series):
    """
    Compute the maximum drawdown of a returns series.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns.
    
    Returns:
    --------
    max_drawdown : float
        The maximum drawdown (as a negative number) observed.
    """
    # Compute cumulative returns
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def compute_cvar(returns_series, alpha=0.05):
    """
    Compute Conditional Value at Risk (CVaR) using historical simulation.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns.
    alpha : float
        Confidence level (e.g., 0.05 for 95% CVaR).
    
    Returns:
    --------
    cvar : float
        The expected shortfall (CVaR) at the given confidence level.
    """
    if returns_series.empty:
        return np.nan
    # Compute VaR as the alpha percentile.
    var = np.percentile(returns_series, 100 * alpha)
    # CVaR is the average return of those returns below VaR.
    cvar = returns_series[returns_series <= var].mean()
    return cvar

def compute_integrated_risk(expected_return, volatility):
    """
    Compute a simple risk-adjusted measure (e.g., a Sharpe-like ratio).
    
    Parameters:
    -----------
    expected_return : float
        The LSTM-based forecasted return.
    volatility : float
        The EGARCH-estimated volatility.
        
    Returns:
    --------
    risk_adjusted_ratio : float
        A ratio of expected return to volatility.
    """
    return expected_return / volatility if volatility != 0 else np.nan
