# src/calibration.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from sklearn.linear_model import LinearRegression

@dataclass
class SchwartzParameters:
    kappa: float
    mu_p: float
    mu_q: float
    sigma: float

class SchwartzCalibrator:
    """
    Executes structural parameter calibration for the Schwartz (1997) mean-reverting 
    process via exact AR(1) discrete-time mapping.
    """
    def __init__(self, dt: float = 1/252):
        self.dt = dt

    def calibrate(self, prices: np.ndarray, market_price_of_risk: float = 0.0) -> SchwartzParameters:
        """
        Derives OU parameters from empirical price series using OLS.
        """
        if np.any(prices <= 0):
            raise ValueError("Prices must be strictly positive for log-transformation.")
            
        log_prices = np.log(prices)
        X_prev = log_prices[:-1].reshape(-1, 1)
        X_next = log_prices[1:]
        
        reg = LinearRegression().fit(X_prev, X_next)
        b_hat = reg.coef_[0]
        a_hat = reg.intercept_
        
        residuals = X_next - reg.predict(X_prev)
        sigma_eta = np.std(residuals, ddof=2)
        
        # Exact discretization mapping
        kappa = -np.log(b_hat) / self.dt
        mu_p = a_hat / (1 - b_hat)
        
        # Exact variance recovery (avoiding Euler approximation bias at large dt)
        sigma = sigma_eta * np.sqrt((2 * kappa) / (1 - b_hat**2))
        
        # Risk-neutral adjustment
        mu_q = mu_p - (market_price_of_risk * sigma) / kappa
        
        return SchwartzParameters(kappa=kappa, mu_p=mu_p, mu_q=mu_q, sigma=sigma)
