# Progress Log

**Project:** Neural Network-Based Estimation of Stochastic Volatility Model Parameters  
**Author:** Martin Kasperlik  
**Started:** January 17, 2026

---

## Week of January 13-18, 2026

### Accomplished
- ✅ Set up project structure with `uv` package manager
- ✅ Configured dependencies (NumPy, PyTorch, matplotlib, statsmodels, etc.)
- ✅ Implemented `BasicSVModel` class in `src/models/sv_basic.py`
- ✅ Validated model produces realistic stochastic volatility:
  - Returns show zero mean, fat tails (kurtosis > 3)
  - Volatility clustering confirmed (positive autocorrelation in squared returns)
  - Visual inspection shows persistence in volatility
- ✅ Created comprehensive documentation:
  - `MASTER_CONTEXT.md` - project overview and goals
  - `WORKING_PRINCIPLES.md` - workflow and project management strategy

### Challenges
- None significant; initial setup went smoothly

### Next Week Goals
- ✅ Implement synthetic data generator for basic SV model
- ✅ Generate first training dataset (50k sequences)
- ✅ Create data loading utilities
- ✅ Set up proper train/val/test splits

### Decisions Made
- Using canonical discrete-time SV model (Kim, Shephard & Chib 1998) as baseline
- Parameters to estimate: φ (persistence) and σ_v (volatility of volatility)
- Starting with synthetic data only; real data is optional bonus
- Using `uv` for dependency management (not Poetry)
- Parameter ranges: φ ∈ [0.85, 0.98], σ_v ∈ [0.10, 0.40]
- Sequence length: T=252 (one trading year)
- Dataset sizes: 50k train, 10k val, 10k test

### New Implementations
- ✅ `BasicSVDataGenerator` class with:
  - Parameter sampling from uniform distributions
  - Efficient batch data generation
  - Train/val/test split functionality
  - Save/load utilities (.npz format)
  - Statistics computation
- ✅ Generated full dataset (~259 MB total):
  - Training: 50,000 sequences (185 MB)
  - Validation: 10,000 sequences (37 MB)
  - Test: 10,000 sequences (37 MB)

### Data Quality Verification
- ✅ Returns have mean ≈ 0 (as expected for zero-drift model)
- ✅ Parameters uniformly distributed across specified ranges
- ✅ All three splits have similar statistics (no bias)
- ✅ Save/load functionality tested and working
- ✅ Generation completed in ~15 seconds (faster than estimated)

### Questions/Blockers
- None currently

---

## Week of January 18-23, 2026

### Accomplished
- ✅ Implemented complete LSTM parameter estimator (`src/neural_networks/lstm_estimator.py`)
  - `LSTMParameterEstimator` class: LSTM architecture for parameter estimation
  - `SVParameterDataset` class: PyTorch Dataset wrapper for synthetic data
  - `train_model` function: Complete training loop with validation, early stopping, checkpointing
  - `evaluate_model` function: Comprehensive evaluation metrics (MSE, MAE, bias)
- ✅ Tested implementation on small dataset (1000 sequences)
  - Model trains successfully
  - Loss decreases properly (train: 0.333 → 0.039, val: 0.073 → 0.007)
  - Evaluation metrics computed correctly
- ✅ Fixed Git issues with large data files
  - Added .npz files to .gitignore
  - Removed large files from Git history
  - Successfully synced to GitHub

### New Implementations
- ✅ Neural network architecture:
  - Input: Return sequences (252 timesteps)
  - LSTM layers (configurable: hidden_size, num_layers, dropout)
  - Fully connected layers mapping to parameter space
  - Output: 2 parameters [φ, σ_v]
- ✅ Training pipeline:
  - Adam optimizer with learning rate scheduling
  - Early stopping based on validation loss
  - Model checkpointing (saves best model)
  - Progress logging
- ✅ Data preprocessing:
  - Automatic return normalization (zero mean, unit variance)
  - PyTorch DataLoader integration

### Test Results
- Quick test (1000 sequences, 5 epochs):
  - Model parameters: 53,074 trainable
  - Training successful
  - Final validation loss: 0.007418
  - Test metrics computed successfully

### Next Week Goals
- Train LSTM on full dataset (50k sequences)
- Tune hyperparameters (hidden_size, num_layers, learning_rate)
- Implement classical benchmark (MLE or particle filter)
- Compare NN vs classical methods

### Decisions Made
- Using LSTM architecture (standard for time series)
- MSE loss function for parameter estimation
- Adam optimizer with ReduceLROnPlateau scheduling
- Early stopping patience: 10 epochs
- Model checkpointing to save best model

### Questions/Blockers
- None currently

---

## Future Weeks

_To be updated weekly..._

