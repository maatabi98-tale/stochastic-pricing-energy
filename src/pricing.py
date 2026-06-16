# src/pricing.py
import numpy as np
from abc import ABC, abstractmethod

class RealOptionsPricer(ABC):
    def __init__(self, risk_free_rate: float, maturity: float):
        self.r = risk_free_rate
        self.T = maturity
        self.discount_factor = np.exp(-self.r * self.T)

class SparkSpreadPricer(RealOptionsPricer):
    """
    Evaluates power plant operational flexibility via Margrabe's exchange option formulation.
    """
    def __init__(self, heat_rate: float, variable_cost: float, risk_free_rate: float, maturity: float):
        super().__init__(risk_free_rate, maturity)
        self.HR = heat_rate
        self.C = variable_cost

    def price(self, paths_gas: np.ndarray, paths_elec: np.ndarray) -> dict:
        """
        Computes the expected discounted payoff of the Spark Spread.
        paths shape: [n_paths, n_steps + 1]. Terminal payoff relies on index [-1].
        """
        ST_gas = paths_gas[:, -1]
        ST_elec = paths_elec[:, -1]
        
        payoffs = np.maximum(ST_elec - self.HR * ST_gas - self.C, 0.0)
        option_value = self.discount_factor * np.mean(payoffs)
        mc_error = self.discount_factor * (np.std(payoffs) / np.sqrt(len(payoffs)))
        
        return {'value': option_value, 'confidence_interval_95': 1.96 * mc_error}

class CalendarSpreadPricer(RealOptionsPricer):
    """
    Evaluates physical gas storage optionality via time-spread valuation.
    """
    def __init__(self, storage_cost: float, risk_free_rate: float, maturity: float):
        super().__init__(risk_free_rate, maturity)
        self.c_storage = storage_cost

    def price(self, paths_gas: np.ndarray) -> dict:
        """
        Simplified proxy: Injection at t=0, withdrawal at t=T.
        Strike K accounts for spot price + physical carry costs.
        """
        S0 = paths_gas[0, 0]
        ST = paths_gas[:, -1]
        
        K = S0 * np.exp((self.r + self.c_storage) * self.T)
        payoffs = np.maximum(ST - K, 0.0)
        
        option_value = self.discount_factor * np.mean(payoffs)
        mc_error = self.discount_factor * (np.std(payoffs) / np.sqrt(len(payoffs)))
        
        return {'value': option_value, 'confidence_interval_95': 1.96 * mc_error}
