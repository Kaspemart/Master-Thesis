
# Master Thesis Context

**Topic:** Neural Network-Based Estimation of Stochastic Volatility Model Parameters

**Czech Title:** Aplikace neuronových sítí pro odhad parametrů modelů se stochastickou volatilitou

---

## My Background & Interests

- Study Financial Engineering / Quantitative Finance
- Interested in AI, neural networks, and data science in finance
- Want thesis to involve **real AI/deep learning**, not just light regression
- Want topic that is academically solid, explainable, and feasible
- Afraid of getting stuck in overly complex problems - robustness and clear "minimum viable thesis" are important
- Interested in stochastic volatility, option pricing, and risk modeling
- May work in quant / trading / research roles - industry relevance matters

---

## What Is This Thesis About?

This thesis studies whether **neural networks can be used as an alternative to classical statistical methods** for estimating unknown parameters of financial models with latent state variables.

**Focus:** Stochastic volatility models where:
- Asset returns are **observable**
- Volatility is **unobserved (latent)**

**Approach:**
- Train neural networks on synthetically generated data
- Infer model parameters directly from observed return sequences
- Compare NN-based estimation to classical approaches

**Key Insight:** Financial returns are driven by volatility, which cannot be directly observed. Classical methods estimate parameters by repeatedly solving difficult optimization or sampling problems involving latent variables, which is **computationally expensive**.

This thesis investigates whether neural networks can learn the **inverse mapping** from observed returns to model parameters, providing **fast and robust estimation once trained**.

Because data is simulated, the true parameters are known, allowing **objective evaluation** of estimation performance.

---

## Latent State Variables

**What are latent state variables?**

Quantities that:
- Exist in the model
- Drive the observed data
- **Cannot be directly observed in reality**

**In this thesis:**
- Returns and prices are **observed**
- Volatility is **latent**
- Volatility evolves over time according to a stochastic process

The fact that volatility is latent makes parameter estimation difficult and motivates both classical filtering methods and neural-network-based approaches.

---

## Models & Parameters

**Basic SV Model (minimum viable):**
```
r_t = μ + exp(h_t/2) * ε_t,    ε_t ~ N(0,1)
h_t = φ * h_{t-1} + σ_v * η_t,  η_t ~ N(0,1)
```

**Parameters to estimate:**
- **φ (phi)** - volatility persistence: controls volatility clustering
- **σ_v (sigma_v)** - volatility of volatility: controls how jumpy volatility is

**Extended scope (if time permits):**
- Heston-type models
- Additional parameters:
  - **κ (kappa)** - speed of mean reversion
  - **θ (theta)** - long-run variance
  - **ρ (rho)** - correlation between price and volatility shocks

**Important:** I do not need to estimate all parameters. The minimum viable thesis only requires **φ and σᵥ**.

---

## Neural Network Task: Amortized Inference

The neural network:
- Takes a **sequence of returns** as input
- Outputs **estimates of the model parameters**
- Is trained on **simulated data**

This is an example of **amortized inference**:
- Classical methods estimate parameters **separately for each dataset**
- The NN learns a **general mapping** and provides **instant estimates after training**

Once trained, the NN can estimate parameters for any new return sequence instantly, without costly optimization.

---

## Benchmarks & Comparison

**Compare NN-based estimation against:**
- Maximum Likelihood Estimation (MLE)
- Particle filter-based estimation
- Bayesian methods (MCMC), if feasible

**Comparison focuses on:**
- Estimation accuracy
- Robustness
- Computational efficiency

**Key limitation of classical methods:**
Every new dataset requires a **costly optimization or sampling procedure**. Neural networks estimate instantly after training.

---

## Data Strategy

**Synthetic data:** Mandatory
**Real market data:** Optional (bonus)

**Data generation process:**
1. Choose model parameters (from distributions)
2. Simulate volatility paths
3. Simulate returns from volatility
4. Pretend parameters are unknown
5. Try to recover them using NN and classical methods

Because I generate the data myself, I know the **true parameters** → objective evaluation possible.

---

## Why Do We Care?

Stable and reliable parameter estimates are important because they tell us:
- Whether volatility clusters
- How persistent risk is
- How fast volatility mean-reverts
- How jumpy or unstable markets are

**These parameters matter for:**
- Risk management
- Option pricing
- Market behavior analysis
- Financial modeling and forecasting

---

## Minimum Viable Thesis

1. Implement basic SV model simulation
2. Generate synthetic data (train/val/test)
3. Build LSTM to estimate φ and σ_v from returns
4. Implement one classical benchmark (e.g., MLE)
5. Compare accuracy and computation time
6. Write full thesis document

**That's it. Everything else is bonus.**

---

## Technical Stack

- Python (NumPy, SciPy, Pandas)
- PyTorch (neural networks)
- Matplotlib (visualization)
- statsmodels (classical methods)
- Jupyter notebooks (exploration)

---

## Backup Topic

**Topic 5 – Application of Machine Learning in Option Pricing**

If needed, the thesis can pivot to:
- Using ML / neural networks to price options
- Training models on synthetic data generated by stochastic volatility models
- Comparing NN pricing speed and accuracy to Monte Carlo methods

This backup topic is closely related and can be integrated naturally if required.

---

## How You Should Help Me (Instructions for AI)

Please:
- Explain concepts clearly and intuitively
- Help me design experiments and structure the thesis
- Help with model choices and simplifications
- Help me avoid unnecessary complexity
- Keep the focus on AI, neural networks, and learning-based inference
- Remember that this is a **master's thesis, not a PhD**

---

## Current Status & Next Steps

**Last Updated:** January 17, 2026

**Status:** Project structure created, dependencies configured, ready to start implementation

**Next:**
1. Run `uv sync` to install dependencies
2. Implement basic SV model in `src/models/`
3. Generate first synthetic dataset
4. Build simple LSTM estimator

---

## Notes & Ideas

### Open Questions
- How much training data needed? (10k, 100k sequences?)
- What sequence length (100 days? 252 days?)
- Loss function: MSE on parameters directly?
- How to normalize returns/parameters?

### Decisions Made
- Start with simplest 2-parameter model (φ, σ_v)
- LSTM as first architecture
- Synthetic data only initially
- Using uv for package management
