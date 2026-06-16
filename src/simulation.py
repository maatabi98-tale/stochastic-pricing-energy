# src/simulation.py
import numpy as np
from src.calibration import SchwartzParameters

class EnergyPathSimulator:
    """
    Monte Carlo engine generating continuous-time multivariate stochastic trajectories
    under the risk-neutral measure Q.
    """
    def __init__(self, n_paths: int = 10000, n_steps: int = 252, dt: float = 1/252, seed: int = 42):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def simulate_bivariate_ou(self, param_g: SchwartzParameters, param_e: SchwartzParameters, 
                              S0_g: float, S0_e: float, rho: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates correlated bivariate trajectories (e.g., Gas and Electricity) 
        using Cholesky factorization of the covariance structure.
        """
        X0_g, X0_e = np.log(S0_g), np.log(S0_e)
        
        Xg = np.empty((self.n_paths, self.n_steps + 1))
        Xe = np.empty((self.n_paths, self.n_steps + 1))
        Xg[:, 0], Xe[:, 0] = X0_g, X0_e
        
        # Cholesky matrix for structural correlation
        L = np.array([[1.0, 0.0],
                      [rho, np.sqrt(1.0 - rho**2)]])
        
        sqrt_dt = np.sqrt(self.dt)
        
        for t in range(self.n_steps):
            Z = self.rng.standard_normal((self.n_paths, 2))
            dW = (Z @ L.T) * sqrt_dt
            
            Xg[:, t+1] = Xg[:, t] + param_g.kappa * (param_g.mu_q - Xg[:, t]) * self.dt + param_g.sigma * dW[:, 0]
            Xe[:, t+1] = Xe[:, t] + param_e.kappa * (param_e.mu_q - Xe[:, t]) * self.dt + param_e.sigma * dW[:, 1]
            
        return np.exp(Xg), np.exp(Xe)

    def simulate_univariate_gbm(self, S0: float, mu: float, sigma: float) -> np.ndarray:
        """
        Simulates standard Geometric Brownian Motion paths as a baseline for 
        asymptotic variance comparison.
        """
        paths = np.empty((self.n_paths, self.n_steps + 1))
        paths[:, 0] = S0
        
        dW = self.rng.standard_normal((self.n_paths, self.n_steps)) * np.sqrt(self.dt)
        
        for t in range(self.n_steps):
            paths[:, t+1] = paths[:, t] * np.exp((mu - 0.5 * sigma**2) * self.dt + sigma * dW[:, t])
            
        return paths
