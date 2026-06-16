# Stochastic Energy Real Options Valuation: Schwartz 1997 Framework

## Structural Overview
This repository implements a rigorous quantitative framework for the valuation of physical energy assets (e.g., CCGT power plants, underground gas storage) conceptualized as Real Options. It overcomes the structural limitations of the traditional Discounted Cash Flow (DCF) models and Geometric Brownian Motion (GBM) formulations, which fail to capture the empirical mean-reverting dynamics and finite asymptotic variance of energy commodities.

The architecture relies on the Schwartz (1997) single-factor Ornstein-Uhlenbeck (OU) process. The transition from the historical measure $\mathbb{P}$ to the equivalent martingale measure $\mathbb{Q}$ is executed via the Girsanov theorem, embedding a constant market price of risk $\lambda$. Bivariate trajectories (e.g., Gas and Electricity) are simulated using a Cholesky decomposition of the covariance matrix to preserve structural market dependencies.

## Mathematical Formulation

### 1. Equivalent Martingale Measure Dynamics
Under the historical measure $\mathbb{P}$, the log-price $X_t = \ln S_t$ follows an OU process:
$$ dX_t = \kappa(\mu^{\mathbb{P}} - X_t)dt + \sigma dW_t^{\mathbb{P}} $$
Applying the Radon-Nikodym derivative and the Girsanov theorem, the risk-neutral $\mathbb{Q}$-dynamics incorporate the market price of risk $\lambda$:
$$ dX_t = \kappa(\mu^{\mathbb{Q}} - X_t)dt + \sigma dW_t^{\mathbb{Q}} \quad \text{where} \quad \mu^{\mathbb{Q}} = \mu^{\mathbb{P}} - \frac{\lambda \sigma}{\kappa} $$

### 2. Bivariate Simulation via Cholesky Factorization
To evaluate cross-commodity derivatives such as the Spark Spread, the stochastic increments $dW_t^{g}$ (Gas) and $dW_t^{e}$ (Electricity) with structural correlation $\mathbb{E}[dW_t^{g} dW_t^{e}] = \rho dt$ are generated via:
$$ \begin{pmatrix} dW_t^{g} \\ dW_t^{e} \end{pmatrix} = \sqrt{\Delta t} \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix} \begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix}, \quad Z_1, Z_2 \overset{\text{iid}}{\sim} \mathcal{N}(0,1) $$

### 3. Econometric Calibration (Exact Discretization)
The continuous OU process admits an exact AR(1) discrete-time mapping:
$$ X_{t+\Delta t} = a + b X_t + \eta_t $$
Using Ordinary Least Squares (OLS) estimators $(\hat{a}, \hat{b}, \hat{\sigma}_\eta)$, the structural parameters are recovered as:
$$ \hat{\kappa} = -\frac{\ln(\hat{b})}{\Delta t}, \quad \hat{\mu}^{\mathbb{P}} = \frac{\hat{a}}{1 - \hat{b}}, \quad \hat{\sigma} = \hat{\sigma}_\eta \sqrt{\frac{2\hat{\kappa}}{1 - \hat{b}^2}} $$

## Directory Architecture
```text
├── README.md
├── src/
│   ├── __init__.py
│   ├── calibration.py          # OLS Econometric estimators
│   ├── simulation.py           # Bivariate Cholesky Monte Carlo engine
│   ├── pricing.py              # Spark/Calendar spread valuation
├── scripts/
│   ├── visualizations.py       # Ergodicity, trajectory, and sensitivity analytics
