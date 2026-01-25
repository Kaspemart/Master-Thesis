"""
--------------------------------------------------------------------------------------------------------------------------------------
- Maximum Likelihood Estimation (MLE) for Basic Stochastic Volatility Model
--------------------------------------------------------------------------------------------------------------------------------------
PURPOSE:
This module implements classical Maximum Likelihood Estimation for the basic SV model.
Since the likelihood function involves intractable integrals over latent volatility,
we use a particle filter to approximate the likelihood, then optimize numerically.
--------------------------------------------------------------------------------------------------------------------------------------
METHOD:
1. Particle filter approximates the likelihood function
2. Numerical optimization (scipy.optimize) finds parameter estimates
3. Returns estimated parameters [φ, σ_v]
--------------------------------------------------------------------------------------------------------------------------------------
REFERENCES:
- Kim, Shephard & Chib (1998) - Standard SV model reference
- Particle filter methods for SV models
- MLE with particle filter likelihood approximation
--------------------------------------------------------------------------------------------------------------------------------------
"""

# IMPORTS:
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import logging
from typing import Callable
# ---------------------------------------------------------------------------------------------------------------------------

# Setting up module logger
logger = logging.getLogger(__name__)


def particle_filter_likelihood(
    returns: np.ndarray,
    phi: float,
    sigma_v: float,
    n_particles: int = 1000,
    h0: float = 0.0
) -> float:
    """
    Approximate log-likelihood using particle filter.
    
    This function uses a bootstrap particle filter to approximate the likelihood
    of the observed returns given parameters (φ, σ_v). The particle filter
    handles the latent volatility by maintaining a set of particles that
    represent possible volatility paths.
    
    Parameters
    ----------
    returns : np.ndarray
        Observed return sequence of shape (T,).
    phi : float
        Persistence parameter (to be estimated).
    sigma_v : float
        Volatility-of-volatility parameter (to be estimated).
    n_particles : int, default=1000
        Number of particles in the filter.
        More particles = more accurate but slower.
    h0 : float, default=0.0
        Initial log-volatility value.
    
    Returns
    -------
    log_likelihood : float
        Approximate log-likelihood of the data given parameters.
    
    Notes
    -----
    The particle filter algorithm:
    1. Initialize particles for log-volatility
    2. For each time step:
       a. Propagate particles forward (volatility evolution)
       b. Weight particles by observation likelihood
       c. Resample particles (bootstrap filter)
    3. Sum log-likelihood contributions across time steps
    """
    T = len(returns)
    
    # Initializing particles
    h_particles = np.full(n_particles, h0)
    log_likelihood = 0.0
    
    for t in range(T):
        # Propagating particles forward (volatility evolution)
        # h_t = phi * h_{t-1} + sigma_v * eta_t
        eta = np.random.randn(n_particles)
        h_particles = phi * h_particles + sigma_v * eta
        
        # Computing observation likelihood for each particle
        # r_t | h_t ~ N(0, exp(h_t))
        # So: p(r_t | h_t) = (1/sqrt(2*pi*exp(h_t))) * exp(-r_t^2 / (2*exp(h_t)))
        variances = np.exp(h_particles)
        log_weights = -0.5 * np.log(2 * np.pi * variances) - 0.5 * (returns[t]**2) / variances
        
        # Normalizing weights (in log space for numerical stability)
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        weights = weights / np.sum(weights)
        
        # Computing likelihood contribution
        # p(r_t | r_{1:t-1}) ≈ sum_i w_i * p(r_t | h_t^i)
        likelihood_contrib = np.sum(weights * np.exp(log_weights))
        log_likelihood += np.log(likelihood_contrib + 1e-10)  # Adding small epsilon for stability
        
        # Resampling particles (bootstrap filter)
        # Resample according to weights
        indices = np.random.choice(n_particles, size=n_particles, p=weights, replace=True)
        h_particles = h_particles[indices]
    
    return log_likelihood


def estimate_parameters_mle(
    returns: np.ndarray,
    n_particles: int = 1000,
    phi_init: float = 0.95,
    sigma_v_init: float = 0.2,
    phi_bounds: tuple[float, float] = (0.01, 0.999),
    sigma_v_bounds: tuple[float, float] = (0.01, 1.0),
    method: str = "L-BFGS-B",
    n_restarts: int = 3,
    seed: int | None = None
) -> tuple[float, float, dict[str, float | bool | int]]:
    """
    Estimate SV model parameters using Maximum Likelihood Estimation.
    
    This function estimates (φ, σ_v) by maximizing the log-likelihood approximated
    using a particle filter. The optimization uses numerical methods from scipy.
    
    Parameters
    ----------
    returns : np.ndarray
        Observed return sequence of shape (T,).
    n_particles : int, default=1000
        Number of particles for likelihood approximation.
        More particles = more accurate but slower.
    phi_init : float, default=0.95
        Initial guess for φ parameter.
    sigma_v_init : float, default=0.2
        Initial guess for σ_v parameter.
    phi_bounds : tuple[float, float], default=(0.01, 0.999)
        Bounds for φ parameter (must be in (0, 1) for stationarity).
    sigma_v_bounds : tuple[float, float], default=(0.01, 1.0)
        Bounds for σ_v parameter (must be positive).
    method : str, default="L-BFGS-B"
        Optimization method. Options: "L-BFGS-B", "Nelder-Mead", "Powell".
        "L-BFGS-B" is recommended for bounded optimization.
    n_restarts : int, default=3
        Number of random restarts to avoid local minima.
        The best result across restarts is returned.
    seed : int | None, default=None
        Random seed for reproducibility.
    
    Returns
    -------
    phi_est : float
        Estimated persistence parameter φ.
    sigma_v_est : float
        Estimated volatility-of-volatility parameter σ_v.
    info : dict[str, float | bool | int]
        Dictionary containing optimization information:
        - 'log_likelihood': Final log-likelihood value (float)
        - 'success': Whether optimization converged (bool)
        - 'n_iterations': Number of iterations (int)
        - 'optimization_time': Time taken for optimization (float)
    
    Examples
    --------
    >>> import numpy as np
    >>> from src.models.sv_basic import BasicSVModel
    >>> 
    >>> # Simulate data with known parameters
    >>> model = BasicSVModel(phi=0.95, sigma_v=0.2)
    >>> returns, _ = model.simulate(T=252, seed=42)
    >>> 
    >>> # Estimate parameters
    >>> phi_est, sigma_v_est, info = estimate_parameters_mle(returns)
    >>> print(f"True: phi=0.95, sigma_v=0.2")
    >>> print(f"Estimated: phi={phi_est:.4f}, sigma_v={sigma_v_est:.4f}")
    >>> print(f"Log-likelihood: {info['log_likelihood']:.2f}")
    
    Notes
    -----
    The particle filter likelihood is stochastic (depends on random particles),
    so results may vary slightly between runs. Using more particles reduces
    this variance but increases computation time.
    
    The optimization may take several seconds to minutes depending on:
    - Sequence length T
    - Number of particles
    - Number of restarts
    - Hardware speed
    """
    if seed is not None:
        np.random.seed(seed)
    
    import time
    start_time = time.time()
    
    # Defining negative log-likelihood function (for minimization)
    def neg_log_likelihood(params: np.ndarray) -> float:
        phi, sigma_v = params
        # Ensuring parameters are in valid ranges
        if not (phi_bounds[0] < phi < phi_bounds[1]):
            return 1e10  # Large penalty for invalid phi
        if not (sigma_v_bounds[0] < sigma_v < sigma_v_bounds[1]):
            return 1e10  # Large penalty for invalid sigma_v
        
        try:
            log_lik = particle_filter_likelihood(
                returns=returns,
                phi=phi,
                sigma_v=sigma_v,
                n_particles=n_particles,
                h0=0.0
            )
            return -log_lik  # Negative because we're minimizing
        except Exception as e:
            logger.warning(f"Error in likelihood computation: {e}")
            return 1e10  # Large penalty for errors
    
    # Running optimization with multiple restarts
    best_result = None
    best_likelihood = -np.inf
    
    for restart in range(n_restarts):
        # Randomizing initial guess for each restart
        if restart > 0:
            phi_init_rand = np.random.uniform(phi_bounds[0] + 0.1, phi_bounds[1] - 0.1)
            sigma_v_init_rand = np.random.uniform(sigma_v_bounds[0] + 0.05, min(0.5, sigma_v_bounds[1] - 0.05))
        else:
            phi_init_rand = phi_init
            sigma_v_init_rand = sigma_v_init
        
        try:
            result = minimize(
                neg_log_likelihood,
                x0=[phi_init_rand, sigma_v_init_rand],
                method=method,
                bounds=[phi_bounds, sigma_v_bounds],
                options={'maxiter': 100, 'disp': False}
            )
            
            if result.success and -result.fun > best_likelihood:
                best_likelihood = -result.fun
                best_result = result
        except Exception as e:
            logger.warning(f"Optimization restart {restart} failed: {e}")
            continue
    
    if best_result is None:
        # Fallback: use initial guess if all optimizations fail
        logger.warning("All optimization restarts failed, using initial guess")
        phi_est = phi_init
        sigma_v_est = sigma_v_init
        log_likelihood = particle_filter_likelihood(returns, phi_init, sigma_v_init, n_particles)
        success = False
        n_iterations = 0
    else:
        phi_est = float(best_result.x[0])
        sigma_v_est = float(best_result.x[1])
        log_likelihood = float(-best_result.fun)
        success = bool(best_result.success)
        n_iterations = int(best_result.nit) if hasattr(best_result, 'nit') else 0
    
    optimization_time = time.time() - start_time
    
    info = {
        'log_likelihood': log_likelihood,
        'success': success,
        'n_iterations': n_iterations,
        'optimization_time': optimization_time
    }
    
    logger.info(
        f"MLE estimation completed: phi={phi_est:.4f}, sigma_v={sigma_v_est:.4f}, "
        f"log_lik={log_likelihood:.2f}, time={optimization_time:.2f}s"
    )
    
    return phi_est, sigma_v_est, info

