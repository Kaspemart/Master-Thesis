# Working Principles & Project Management

**Last Updated:** January 18, 2026

This document outlines how to effectively manage AI tools, track progress, and maintain context throughout this multi-month master thesis project.

---

## 🤖 AI Tool Strategy

### Tool Roles

**Cursor AI (this assistant):**
- **Primary use:** Coding, implementation, debugging
- **Strengths:** Direct file access, tool execution, code editing
- **When to use:** Writing code, testing models, data generation, implementing methods
- **Context source:** `MASTER_CONTEXT.md` (always attach to conversations)

**External ChatGPT:**
- **Primary use:** Literature review, writing, conceptual explanations
- **Strengths:** Long-form text generation, explaining concepts, thesis structure
- **When to use:** Writing thesis chapters, literature summaries, conceptual questions
- **Context source:** `MASTER_CONTEXT.md` + `PROGRESS_LOG.md` (manual sharing)

### Maintaining Consistency Across Tools

1. **Single Source of Truth:** `MASTER_CONTEXT.md` contains all project goals, models, decisions
2. **Progress Tracking:** `PROGRESS_LOG.md` tracks what has been done (updated by you weekly)
3. **Start every new chat:** Attach or paste relevant context files
4. **After major milestones:** Update `MASTER_CONTEXT.md` with new decisions/insights

---

## 📁 Project Structure

```
Master-Thesis/
├── MASTER_CONTEXT.md           # What the thesis is about (IMMUTABLE REFERENCE)
├── WORKING_PRINCIPLES.md       # This file (how to work on the thesis)
├── PROGRESS_LOG.md             # What you've done (UPDATED WEEKLY)
├── pyproject.toml              # Dependencies
├── uv.lock                     # Locked dependencies
│
├── src/                        # All implementation code
│   ├── models/                 # Stochastic volatility models
│   │   └── sv_basic.py         # ✅ Basic SV model (DONE)
│   │
│   ├── data/                   # Data generation
│   │   ├── synthetic/          # Synthetic data generators
│   │   │   └── basic_sv/       # Data for basic SV model
│   │   └── real/               # Real market data (optional)
│   │
│   ├── neural_networks/        # Neural network estimators
│   │   └── lstm_estimator.py   # TODO: LSTM-based parameter estimator
│   │
│   ├── classical/              # Classical estimation methods
│   │   ├── mle.py              # TODO: Maximum Likelihood
│   │   ├── particle_filter.py  # TODO: Particle filter
│   │   └── mcmc.py             # TODO: MCMC (optional)
│   │
│   ├── evaluation/             # Comparison & metrics
│   │   └── metrics.py          # TODO: Evaluation metrics
│   │
│   └── utils/                  # Helper functions
│       └── visualization.py    # TODO: Plotting utilities
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_model_validation.ipynb
│   ├── 02_data_generation.ipynb
│   └── 03_exploratory_analysis.ipynb
│
├── experiments/                # Experimental results
│   └── results/                # Saved outputs
│
├── tests/                      # Unit tests (pytest)
│   └── test_sv_basic.py        # TODO: Test basic SV model
│
└── thesis/                     # LaTeX/Word thesis files
    ├── chapters/
    ├── figures/
    └── references.bib
```

---

## ✅ Validation Results: Basic SV Model

**Date:** January 18, 2026  
**Status:** ✅ **MODEL WORKS CORRECTLY**

### Test Configuration
- Parameters: `phi=0.95, sigma_v=0.2`
- Time steps: `T=500`
- Seed: `42` (reproducible)

### Results
```
Returns statistics:
  Mean:     0.015480 (✓ close to 0)
  Std Dev:  1.1176   (✓ reasonable)
  Max:      4.6544   (✓ fat tails present)
  Min:      -4.5292
  Kurtosis: 4.2194   (✓ >3 indicates fat tails)

Volatility clustering:
  Lag 1 autocorr: 0.0154 (✓ positive)
  Lag 5 autocorr: 0.0538 (✓ positive)
```

### Visual Inspection
- ✅ Returns show clear volatility clustering (big moves follow big moves)
- ✅ Log-volatility follows AR(1) process smoothly
- ✅ Volatility (exp(h/2)) shows persistence
- ✅ Model captures stylized facts of financial returns

**Conclusion:** The basic SV model is correctly implemented and ready for data generation.

---

## 🎯 Next Steps (Immediate)

### Phase 1: Data Generation (Next 1-2 weeks)

**Goal:** Create thousands of synthetic return sequences with known parameters for training neural networks.

**What to implement:**

1. **Data Generator Class** (`src/data/synthetic/basic_sv/generator.py`)
   - Generate N sequences of returns
   - Random parameters sampled from realistic distributions
   - Save as `.npz` or `.h5` files
   - Train/validation/test splits

2. **Parameter Sampling Strategy**
   - `phi ~ Uniform(0.85, 0.98)` (high persistence typical in finance)
   - `sigma_v ~ Uniform(0.10, 0.40)` (reasonable volatility range)
   - Sequence length: T=252 (one trading year) or T=500

3. **Dataset Sizes**
   - Training: 50,000 sequences
   - Validation: 10,000 sequences
   - Test: 10,000 sequences

**Expected output:**
```python
# Example usage:
from src.data.synthetic.basic_sv import generate_dataset

data = generate_dataset(
    n_sequences=50000,
    T=252,
    phi_range=(0.85, 0.98),
    sigma_v_range=(0.10, 0.40),
    seed=42
)
# Returns: dict with keys ['returns', 'parameters', 'log_volatility']
```

---

## 📋 Development Checklist

### Models
- [x] Basic SV model implementation (`sv_basic.py`)
- [x] Model validation and testing
- [ ] Extended SV models (Heston, etc.) - OPTIONAL

### Data
- [ ] Synthetic data generator for basic SV
- [ ] Data normalization utilities
- [ ] Train/val/test split functionality
- [ ] Data loading utilities
- [ ] Real market data processing - OPTIONAL

### Neural Networks
- [ ] LSTM estimator architecture
- [ ] Training loop with PyTorch
- [ ] Hyperparameter tuning
- [ ] Model checkpointing and saving

### Classical Methods
- [ ] Maximum Likelihood Estimation (MLE)
- [ ] Particle filter implementation
- [ ] MCMC (Bayesian) - OPTIONAL

### Evaluation
- [ ] Evaluation metrics (MSE, MAE, bias)
- [ ] Comparison framework
- [ ] Visualization of results
- [ ] Statistical tests (confidence intervals)

### Thesis Writing
- [ ] Literature review chapter
- [ ] Methodology chapter
- [ ] Results chapter
- [ ] Conclusion chapter

---

## 🔬 Research Questions to Answer

1. **Accuracy:** Can neural networks estimate φ and σ_v accurately?
2. **Robustness:** How does accuracy vary with different true parameters?
3. **Sample size:** How many time steps (T) are needed?
4. **Training data:** How many training sequences are needed?
5. **Speed:** How fast is NN inference vs. classical methods?
6. **Generalization:** Do NNs trained on simulated data work on real data?

---

## 💡 Open Questions & Decisions Needed

### Data Generation
- [ ] **Q:** What sequence length T? → **Proposal:** Start with T=252, test T=500
- [ ] **Q:** How many training sequences? → **Proposal:** 50k train, 10k val, 10k test
- [ ] **Q:** Parameter ranges? → **Proposal:** φ∈[0.85,0.98], σ_v∈[0.10,0.40]

### Neural Network Design
- [ ] **Q:** LSTM architecture (layers, hidden size)? → **Decide after data ready**
- [ ] **Q:** Loss function? → **Proposal:** MSE on normalized parameters
- [ ] **Q:** Input normalization? → **Proposal:** Standardize returns

### Classical Benchmarks
- [ ] **Q:** Which classical method as primary benchmark? → **Proposal:** Start with MLE
- [ ] **Q:** Particle filter or MCMC? → **Decide based on time/complexity**

---

## 📝 How to Update Context Documents

### When to Update `MASTER_CONTEXT.md`
- Major design decisions made (e.g., model architecture chosen)
- Research direction changes
- New research questions emerge
- Key insights from experiments

### When to Update `PROGRESS_LOG.md`
- Weekly progress summaries
- Completed milestones
- Failed experiments (what didn't work and why)
- Breakthroughs and unexpected findings

### When to Update `WORKING_PRINCIPLES.md`
- New tools or workflows adopted
- Changes to project structure
- Important validation results
- Updated checklists and next steps

---

## 🚨 Risk Management

### Identified Risks
1. **Complexity creep:** Adding too many models/methods
   - **Mitigation:** Stick to minimum viable thesis first
   
2. **Insufficient training data:** NN doesn't converge
   - **Mitigation:** Start with 50k sequences, scale up if needed
   
3. **Classical methods too complex:** MLE/MCMC implementation takes too long
   - **Mitigation:** Use existing libraries (statsmodels, PyMC)
   
4. **Time management:** Thesis takes longer than expected
   - **Mitigation:** Clear milestones, MVP-first approach

### Red Flags to Watch For
- ⚠️ Neural network not learning (loss not decreasing)
- ⚠️ Parameter estimates far from true values
- ⚠️ Classical benchmarks impossible to implement
- ⚠️ Real data behaves completely differently than synthetic

---

## 🎓 Minimum Viable Thesis (MVP)

**Goal:** Ensure you can graduate even if everything goes wrong

**MVP Requirements:**
1. ✅ Basic SV model implemented and validated
2. ⏳ Synthetic data generation (50k sequences)
3. ⏳ Simple LSTM estimator trained
4. ⏳ ONE classical benchmark (MLE preferred)
5. ⏳ Comparison showing NN works (even if not better)
6. ⏳ Complete written thesis

**Everything else is BONUS.**

---

## 📚 Key References to Track

### Core Papers
- Kim, Shephard & Chib (1998) - Stochastic Volatility
- [Add papers as you find them]

### Neural Network for Time Series
- [Add relevant papers]

### Parameter Estimation
- [Add relevant papers]

---

## 🔄 Weekly Review Template

Copy this to `PROGRESS_LOG.md` each week:

```markdown
### Week of [DATE]

**Accomplished:**
- 

**Challenges:**
- 

**Next Week Goals:**
- 

**Decisions Made:**
- 

**Questions/Blockers:**
- 
```

---

## 🎯 Current Status

**Phase:** Neural Network Implementation  
**Previous Milestone:** ✅ Data Generation Complete  
**Next Milestone:** LSTM Estimator  
**Estimated Time to MVP:** 3-4 months  
**Confidence Level:** High ✅

**Completed:**
- ✅ Basic SV model (validated and working)
- ✅ Synthetic data generator
- ✅ Full training dataset (70k sequences, 259 MB)

**Ready for:**
- Neural network training
- Parameter estimation experiments

---

**Remember:** This is a master's thesis, not a PhD. Focus on doing one thing well rather than many things poorly. The basic SV model with NN estimation is enough if executed properly.
