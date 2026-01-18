"""
--------------------------------------------------------------------------------------------------------------------------------------
- This module implements the Canonical Discrete-Time Stochastic Volatility Model
--------------------------------------------------------------------------------------------------------------------------------------
MODEL STRUCTURE: The model consists of two equations:
1. **Volatility process (latent state):**
   h_t = φ * h_{t-1} + σ_v * η_t,  η_t ~ N(0,1)

2. **Return process (observed):**
   r_t = exp(h_t/2) * ε_t,          ε_t ~ N(0,1)
--------------------------------------------------------------------------------------------------------------------------------------
VARIABLES:
h_t : log-volatility (latent state variable, unobserved)
r_t : return (observed)
η_t, ε_t : independent standard normal innovations
--------------------------------------------------------------------------------------------------------------------------------------
PARAMETERS:
φ (phi) : persistence parameter
    Controls how persistent volatility is over time.
    Typically between 0.8 and 0.99 for daily financial data.
    Higher values → more persistent volatility clustering.

σ_v (sigma_v) : volatility of volatility
    Controls the variability of the volatility process itself.
    Typically between 0.1 and 0.5.
    Higher values → more jumpy/unstable volatility.
--------------------------------------------------------------------------------------------------------------------------------------
KEY FEATURES:
- **Latent volatility:** The volatility h_t cannot be directly observed, only
  the returns r_t are observable. This creates estimation difficulty.

- **Nonlinearity:** The exponential link function exp(h_t/2) creates a nonlinear
  relationship between latent volatility and observed returns.

- **Volatility clustering:** The AR(1) structure of h_t produces the stylized
  fact that large (small) price movements tend to be followed by large (small)
  movements of either sign.
--------------------------------------------------------------------------------------------------------------------------------------
WHY THIS MODEL?
This is chosen as the baseline model because:
- 1. It is standard in the literature (Kim, Shephard & Chib 1998)
- 2. It already has latent volatility → creates estimation difficulty
- 3. It has only 2 parameters → minimizes complexity risk
- 4. It is simpler than Heston but captures key features
- 5. Extensive research exists for benchmarking
--------------------------------------------------------------------------------------------------------------------------------------
REFERENCE:
- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: likelihood
  inference and comparison with ARCH models. The Review of Economic Studies,
  65(3), 361-393.
- It is the standard reference in the literature.
--------------------------------------------------------------------------------------------------------------------------------------
"""


# IMPORTS:
import numpy as np
# ------------------------------------------------------------------------------------------------------------------------------------


class BasicSVModel:
    """
    Canonical discrete-time stochastic volatility model (Kim, Shephard & Chib 1998).
    
    This class implements the standard discrete-time SV model used throughout the econometrics and financial modeling literature.
    The model features:
    - An AR(1) process for log-volatility (latent state)
    - Exponential link function connecting volatility to returns
    - Two parameters to estimate: persistence (φ) and vol-of-vol (σ_v)
    
    The model captures key stylized facts of financial returns:
    - Volatility clustering (periods of high/low volatility persist)
    - Heavy tails in the return distribution
    - Time-varying conditional variance
    
    This is the baseline model for the Master thesis because it:
    - 1. Has latent volatility → creates the parameter estimation challenge
    - 2. Is well-studied → extensive literature for comparison
    - 3. Has minimal parameters → reduces complexity risk
    - 4. Is computationally tractable for both classical and NN methods
    
    PARAMETERS:
    ------------------------------------------------------------------------------------------------------------------------------------
    phi : float, default=0.9
        Persistence parameter controlling volatility clustering.
        Must be in (0, 1) for stationarity.
        Typical range: 0.8 to 0.99 for daily financial data.
        
    sigma_v : float, default=0.2
        Volatility of volatility parameter controlling how variable
        the volatility process itself is.
        Must be positive.
        Typical range: 0.1 to 0.5.
    
    ATTRIBUTES:
    ------------------------------------------------------------------------------------------------------------------------------------
    phi : float
        The persistence parameter
    sigma_v : float
        The volatility of volatility parameter
    
    EXAMPLES:
    ------------------------------------------------------------------------------------------------------------------------------------
    >>> # Create model with high persistence
    >>> model = BasicSVModel(phi=0.95, sigma_v=0.15)
    >>> 
    >>> # Simulate one year of daily returns
    >>> returns, log_vol = model.simulate(T=252, seed=42)
    >>> 
    >>> # Check volatility clustering visually
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(returns)
    >>> plt.title("Simulated Returns with Stochastic Volatility")
    >>> plt.show()
    
    NOTES:
    ------------------------------------------------------------------------------------------------------------------------------------
    The model equations are:
        h_t = φ * h_{t-1} + σ_v * η_t,  η_t ~ N(0,1)  (volatility evolution)
        r_t = exp(h_t/2) * ε_t,          ε_t ~ N(0,1)  (return generation)
    
    where h_t is the latent log-volatility and r_t is the observed return.
    The exponential transformation ensures volatility is always positive.
    
    REFERENCES:
    ------------------------------------------------------------------------------------------------------------------------------------
    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: likelihood
    inference and comparison with ARCH models. The Review of Economic Studies,
    65(3), 361-393.
    """
    
    def __init__(self, phi: float = 0.9, sigma_v: float = 0.2):
        """This method initializes the SV model with given parameters."""
        self.phi = phi
        self.sigma_v = sigma_v
        
        # Validating the parameters
        if not 0 < phi < 1:
            raise ValueError(f"phi must be in (0, 1), got {phi}")
        if sigma_v <= 0:
            raise ValueError(f"sigma_v must be positive, got {sigma_v}")
    

    def simulate(self, T: int = 252, h0: float = 0.0, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        This method simulates returns and log-volatility trajectories from the SV model.
        
        This method generates synthetic data by simulating both the latent log-volatility process and the observed returns.
        The simulation follows the model equations:
        
            h_t = φ * h_{t-1} + σ_v * η_t    (volatility evolution, latent)
            r_t = exp(h_t/2) * ε_t            (return generation, observed)
        
        The latent volatility h_t follows an AR(1) process, creating persistence
        (volatility clustering). The observed returns r_t are conditionally normal
        with time-varying variance exp(h_t).
        
        This simulation is used to:
        1. Generate training data for neural networks (with known true parameters)
        2. Test estimation methods (both classical and NN-based)
        3. Validate model implementation
        4. Create visualizations for exploratory analysis
        
        PARAMETERS:
        ------------------------------------------------------------------------------------------------------------------------------------
        T : int, default=252
            Number of time steps to simulate.
            Default is 252 (one trading year of daily data).
            For testing, start with T=100-500.
            For training NNs, typical choices are T=252 or T=500.
            
        h0 : float, default=0.0
            Initial log-volatility value.
            Zero corresponds to volatility = exp(0/2) = 1.
            Typically set to 0.0 (unconditional mean of the process when stationary).
            
        seed : int or None, default=None
            Random seed for reproducibility.
            If None, results will vary each time.
            Set to a fixed integer (e.g., 42) for reproducible experiments.
        
        RETURNS:
        ------------------------------------------------------------------------------------------------------------------------------------
        returns : np.ndarray
            Simulated returns of shape (T,).
            These are the **observed** data that would be available in practice.
            Mean is approximately 0, variance is time-varying.
            
        log_volatility : np.ndarray
            Simulated log-volatility (latent state) of shape (T,).
            These are **unobserved** in practice but known here since we simulated.
            Useful for analysis and debugging but not available when estimating
            parameters from real data.
        
        EXAMPLES:
        ------------------------------------------------------------------------------------------------------------------------------------
        >>> # Basic simulation
        >>> model = BasicSVModel(phi=0.9, sigma_v=0.2)
        >>> returns, log_vol = model.simulate(T=252, seed=42)
        >>> 
        >>> # Check shapes
        >>> returns.shape
        (252,)
        >>> log_vol.shape
        (252,)
        >>> 
        >>> # Visualize volatility clustering
        >>> import matplotlib.pyplot as plt
        >>> fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        >>> axes[0].plot(returns)
        >>> axes[0].set_title('Returns (Observed)')
        >>> axes[1].plot(np.exp(log_vol/2))
        >>> axes[1].set_title('Volatility (Latent)')
        >>> plt.show()
        >>> 
        >>> # Generate multiple sequences with different parameters
        >>> sequences = []
        >>> for phi in [0.85, 0.90, 0.95]:
        ...     model = BasicSVModel(phi=phi, sigma_v=0.2)
        ...     r, h = model.simulate(T=500, seed=phi*100)
        ...     sequences.append((r, h, phi))
        
        NOTES:
        ------------------------------------------------------------------------------------------------------------------------------------
        - The first return r[0] is generated using the initial volatility h[0] = h0
        - Innovations η_t and ε_t are independent standard normals
        - The exponential transformation ensures positive volatility
        - For stationary behavior, ensure |φ| < 1 (enforced in __init__)
        - Typical autocorrelation in returns^2 will be visible due to volatility clustering
        """
        # Setting the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Initializing the arrays for the log-volatility and returns
        h = np.zeros(T)  # log-volatility (latent)
        r = np.zeros(T)  # returns (observed)
        
        # Setting the initial log-volatility
        h[0] = h0
        
        # Simulating the first return (no previous volatility)
        r[0] = np.exp(h[0] / 2) * np.random.randn()
        
        # Simulating the remaining time steps
        for t in range(1, T):
            # Volatility evolution: h_t = φ * h_{t-1} + σ_v * η_t
            h[t] = self.phi * h[t-1] + self.sigma_v * np.random.randn()
            
            # Return generation: r_t = exp(h_t/2) * ε_t
            r[t] = np.exp(h[t] / 2) * np.random.randn()
        
        return r, h
    

    def __repr__(self) -> str:
        """This method returns a string representation of the model."""
        return f"BasicSVModel(phi={self.phi:.3f}, sigma_v={self.sigma_v:.3f})"

