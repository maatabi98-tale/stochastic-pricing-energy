# scripts/visualizations.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.calibration import SchwartzParameters
from src.simulation import EnergyPathSimulator
from src.pricing import SparkSpreadPricer

# Quantitative aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

def plot_variance_bounding(simulator: EnergyPathSimulator, param_ou: SchwartzParameters, S0: float):
    """
    Demonstrates the ergodicity of the OU process vs the unbounded variance of GBM.
    """
    paths_ou, _ = simulator.simulate_bivariate_ou(param_ou, param_ou, S0, S0, rho=0.0)
    paths_gbm = simulator.simulate_univariate_gbm(S0, mu=0.0, sigma=param_ou.sigma)
    
    time_grid = np.linspace(0, simulator.n_steps * simulator.dt, simulator.n_steps + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Plot OU
    axes[0].plot(time_grid, paths_ou[:100].T, color='#2980b9', alpha=0.1)
    axes[0].axhline(np.exp(param_ou.mu_q), color='#c0392b', linestyle='--', label=r'Equilibrium $\exp(\mu^\mathbb{Q})$')
    axes[0].set_title('Schwartz 1997: Mean-Reverting Dynamics (Bounded Variance)')
    axes[0].set_xlabel('Time (Years)')
    axes[0].set_ylabel('Asset Price')
    axes[0].legend()

    # Plot GBM
    axes[1].plot(time_grid, paths_gbm[:100].T, color='#27ae60', alpha=0.1)
    axes[1].set_title('Black-Scholes: Geometric Brownian Motion (Diverging Variance)')
    axes[1].set_xlabel('Time (Years)')
    
    plt.tight_layout()
    plt.show()

def plot_kappa_sensitivity(simulator: EnergyPathSimulator, base_param_g: SchwartzParameters, 
                           base_param_e: SchwartzParameters, S0_g: float, S0_e: float, rho: float):
    """
    Evaluates the sensitivity of the Spark Spread real option to the mean-reversion speed.
    """
    kappa_grid = np.linspace(0.1, 5.0, 15)
    prices = []
    
    pricer = SparkSpreadPricer(heat_rate=7.0, variable_cost=2.0, risk_free_rate=0.05, maturity=1.0)
    
    for k in kappa_grid:
        test_param_g = SchwartzParameters(k, base_param_g.mu_p, base_param_g.mu_q, base_param_g.sigma)
        test_param_e = SchwartzParameters(k, base_param_e.mu_p, base_param_e.mu_q, base_param_e.sigma)
        
        paths_g, paths_e = simulator.simulate_bivariate_ou(test_param_g, test_param_e, S0_g, S0_e, rho)
        res = pricer.price(paths_g, paths_e)
        prices.append(res['value'])
        
    plt.figure(figsize=(10, 6))
    plt.plot(kappa_grid, prices, marker='o', color='#8e44ad', linewidth=2)
    plt.title(r'Spark Spread Sensitivity to Mean-Reversion Speed ($\kappa$)')
    plt.xlabel(r'Speed of Mean Reversion ($\kappa$)')
    plt.ylabel('Option Value ($/MWh)')
    
    # Annotate analytical implication
    plt.annotate('Higher $\kappa$ rapidly dissipates price spikes,\ncompressing option extrinsic value.', 
                 xy=(kappa_grid[-3], prices[-3]), xytext=(0.5, 0.8), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
    
    plt.tight_layout()
    plt.show()
