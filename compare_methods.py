# IMPORTS:
import numpy as np
import torch
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from src.neural_networks.lstm_estimator import LSTMParameterEstimator, SVParameterDataset, evaluate_model
from src.classical.mle import estimate_parameters_mle
from src.data.synthetic.basic_sv import BasicSVDataGenerator
from torch.utils.data import DataLoader
import logging
# ---------------------------------------------------------------------------------------------------------------------------

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("=" * 80)
print("COMPARING NEURAL NETWORK vs MAXIMUM LIKELIHOOD ESTIMATION")
print("=" * 80)

# Creating output directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Loading test data
print("\n" + "=" * 80)
print("LOADING TEST DATA")
print("=" * 80)

generator = BasicSVDataGenerator(seed=42)
test_data = generator.load_dataset('data/synthetic/basic_sv/test.npz')

# Using subset for comparison (MLE is slow, so we'll use first 100 sequences)
N_COMPARE = 100  # Number of sequences to compare
print(f"Using first {N_COMPARE} sequences from test set for comparison")
print(f"(MLE is computationally expensive, so using subset)")

test_returns = test_data['returns'][:N_COMPARE]
test_parameters = test_data['parameters'][:N_COMPARE]

print(f"Test sequences: {len(test_returns)}")
print(f"Sequence length: {test_returns.shape[1]}")

# ============================================================================
# NEURAL NETWORK ESTIMATION
# ============================================================================
print("\n" + "=" * 80)
print("NEURAL NETWORK ESTIMATION")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMParameterEstimator(hidden_size=128, num_layers=2, dropout=0.2)

# Loading trained model
model_path = Path("models/lstm_estimator_full.pt")
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("Loaded trained LSTM model")

# Creating dataset for NN
test_dataset = SVParameterDataset(
    returns=test_returns,
    parameters=test_parameters,
    normalize_returns=True
)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Running NN predictions
print("Running neural network predictions...")
nn_start_time = time.time()

nn_predictions = []
with torch.no_grad():
    for returns, _ in test_loader:
        returns = returns.to(device)
        predictions = model(returns)
        nn_predictions.append(predictions.cpu().numpy())

nn_predictions = np.concatenate(nn_predictions, axis=0)
nn_time = time.time() - nn_start_time

print(f"NN estimation completed in {nn_time:.4f} seconds")
print(f"Average time per sequence: {nn_time / N_COMPARE:.4f} seconds")

# ============================================================================
# MAXIMUM LIKELIHOOD ESTIMATION
# ============================================================================
print("\n" + "=" * 80)
print("MAXIMUM LIKELIHOOD ESTIMATION")
print("=" * 80)
print("This will take several minutes...")

mle_predictions = []
mle_times = []
mle_success = []

for i, returns_seq in enumerate(test_returns):
    if (i + 1) % 10 == 0:
        print(f"  Processing sequence {i+1}/{N_COMPARE}...")
    
    mle_start = time.time()
    try:
        phi_est, sigma_v_est, info = estimate_parameters_mle(
            returns=returns_seq,
            n_particles=500,  # Using fewer particles for speed
            n_restarts=2,
            seed=42 + i  # Different seed for each sequence
        )
        mle_predictions.append([phi_est, sigma_v_est])
        mle_times.append(time.time() - mle_start)
        mle_success.append(info['success'])
    except Exception as e:
        logger.warning(f"MLE failed for sequence {i+1}: {e}")
        # Using fallback: mean of parameter ranges
        mle_predictions.append([0.915, 0.25])  # Approximate mean
        mle_times.append(0.0)
        mle_success.append(False)

mle_predictions = np.array(mle_predictions)
total_mle_time = sum(mle_times)

print(f"\nMLE estimation completed")
print(f"Total time: {total_mle_time:.2f} seconds")
print(f"Average time per sequence: {np.mean(mle_times):.4f} seconds")
print(f"Success rate: {np.mean(mle_success):.2%}")

# ============================================================================
# COMPARING RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("COMPARING RESULTS")
print("=" * 80)

# Computing metrics for both methods
targets = test_parameters[:N_COMPARE]

# NN metrics
nn_errors = nn_predictions - targets
nn_metrics = {
    'mse_phi': float(np.mean((nn_errors[:, 0]) ** 2)),
    'mse_sigma_v': float(np.mean((nn_errors[:, 1]) ** 2)),
    'mae_phi': float(np.mean(np.abs(nn_errors[:, 0]))),
    'mae_sigma_v': float(np.mean(np.abs(nn_errors[:, 1]))),
    'bias_phi': float(np.mean(nn_errors[:, 0])),
    'bias_sigma_v': float(np.mean(nn_errors[:, 1])),
    'mse_total': float(np.mean(nn_errors ** 2)),
    'avg_time_per_sequence': nn_time / N_COMPARE,
    'total_time': nn_time
}

# MLE metrics
mle_errors = mle_predictions - targets
mle_metrics = {
    'mse_phi': float(np.mean((mle_errors[:, 0]) ** 2)),
    'mse_sigma_v': float(np.mean((mle_errors[:, 1]) ** 2)),
    'mae_phi': float(np.mean(np.abs(mle_errors[:, 0]))),
    'mae_sigma_v': float(np.mean(np.abs(mle_errors[:, 1]))),
    'bias_phi': float(np.mean(mle_errors[:, 0])),
    'bias_sigma_v': float(np.mean(mle_errors[:, 1])),
    'mse_total': float(np.mean(mle_errors ** 2)),
    'avg_time_per_sequence': float(np.mean(mle_times)),
    'total_time': total_mle_time,
    'success_rate': float(np.mean(mle_success))
}

# Printing comparison
print("\nNeural Network Results:")
print(f"  MSE (phi):     {nn_metrics['mse_phi']:.6f}")
print(f"  MSE (sigma_v): {nn_metrics['mse_sigma_v']:.6f}")
print(f"  MAE (phi):     {nn_metrics['mae_phi']:.6f}")
print(f"  MAE (sigma_v): {nn_metrics['mae_sigma_v']:.6f}")
print(f"  Bias (phi):    {nn_metrics['bias_phi']:.6f}")
print(f"  Bias (sigma_v): {nn_metrics['bias_sigma_v']:.6f}")
print(f"  Avg time/seq:  {nn_metrics['avg_time_per_sequence']:.4f} seconds")

print("\nMLE Results:")
print(f"  MSE (phi):     {mle_metrics['mse_phi']:.6f}")
print(f"  MSE (sigma_v): {mle_metrics['mse_sigma_v']:.6f}")
print(f"  MAE (phi):     {mle_metrics['mae_phi']:.6f}")
print(f"  MAE (sigma_v): {mle_metrics['mae_sigma_v']:.6f}")
print(f"  Bias (phi):    {mle_metrics['bias_phi']:.6f}")
print(f"  Bias (sigma_v): {mle_metrics['bias_sigma_v']:.6f}")
print(f"  Avg time/seq:  {mle_metrics['avg_time_per_sequence']:.4f} seconds")
print(f"  Success rate:  {mle_metrics['success_rate']:.2%}")

print("\nComparison:")
speedup = mle_metrics['avg_time_per_sequence'] / nn_metrics['avg_time_per_sequence']
print(f"  Speed: NN is {speedup:.1f}x faster than MLE")
print(f"  Accuracy (MSE phi): NN {nn_metrics['mse_phi']:.6f} vs MLE {mle_metrics['mse_phi']:.6f}")
print(f"  Accuracy (MSE sigma_v): NN {nn_metrics['mse_sigma_v']:.6f} vs MLE {mle_metrics['mse_sigma_v']:.6f}")

# ============================================================================
# CREATING VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING COMPARISON VISUALIZATIONS")
print("=" * 80)

# 1. Side-by-side scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# NN scatter plots
axes[0, 0].scatter(targets[:, 0], nn_predictions[:, 0], alpha=0.5, s=20)
axes[0, 0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                [targets[:, 0].min(), targets[:, 0].max()], 
                'r--', linewidth=2, label='Perfect')
axes[0, 0].set_xlabel('True phi', fontsize=11)
axes[0, 0].set_ylabel('Predicted phi', fontsize=11)
axes[0, 0].set_title('Neural Network: phi', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(targets[:, 1], nn_predictions[:, 1], alpha=0.5, s=20, color='orange')
axes[0, 1].plot([targets[:, 1].min(), targets[:, 1].max()], 
                [targets[:, 1].min(), targets[:, 1].max()], 
                'r--', linewidth=2, label='Perfect')
axes[0, 1].set_xlabel('True sigma_v', fontsize=11)
axes[0, 1].set_ylabel('Predicted sigma_v', fontsize=11)
axes[0, 1].set_title('Neural Network: sigma_v', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# MLE scatter plots
axes[1, 0].scatter(targets[:, 0], mle_predictions[:, 0], alpha=0.5, s=20)
axes[1, 0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                [targets[:, 0].min(), targets[:, 0].max()], 
                'r--', linewidth=2, label='Perfect')
axes[1, 0].set_xlabel('True phi', fontsize=11)
axes[1, 0].set_ylabel('Predicted phi', fontsize=11)
axes[1, 0].set_title('MLE: phi', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(targets[:, 1], mle_predictions[:, 1], alpha=0.5, s=20, color='orange')
axes[1, 1].plot([targets[:, 1].min(), targets[:, 1].max()], 
                [targets[:, 1].min(), targets[:, 1].max()], 
                'r--', linewidth=2, label='Perfect')
axes[1, 1].set_xlabel('True sigma_v', fontsize=11)
axes[1, 1].set_ylabel('Predicted sigma_v', fontsize=11)
axes[1, 1].set_title('MLE: sigma_v', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
comparison_path = RESULTS_DIR / "nn_vs_mle_comparison.png"
plt.savefig(comparison_path, dpi=150)
print(f"  Saved comparison plots to: {comparison_path}")
plt.close()

# 2. Error comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Phi errors
axes[0].hist(nn_errors[:, 0], bins=30, alpha=0.6, label='NN', edgecolor='black')
axes[0].hist(mle_errors[:, 0], bins=30, alpha=0.6, label='MLE', edgecolor='black')
axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Error (Predicted - True)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Error Distribution: phi', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Sigma_v errors
axes[1].hist(nn_errors[:, 1], bins=30, alpha=0.6, label='NN', edgecolor='black', color='orange')
axes[1].hist(mle_errors[:, 1], bins=30, alpha=0.6, label='MLE', edgecolor='black', color='green')
axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Error (Predicted - True)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Error Distribution: sigma_v', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
error_comparison_path = RESULTS_DIR / "error_comparison.png"
plt.savefig(error_comparison_path, dpi=150)
print(f"  Saved error comparison to: {error_comparison_path}")
plt.close()

# 3. Speed comparison
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['Neural Network', 'MLE']
times = [nn_metrics['avg_time_per_sequence'], mle_metrics['avg_time_per_sequence']]
colors = ['blue', 'orange']

bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Time per Sequence (seconds)', fontsize=12)
ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Adding value labels on bars
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{time_val:.4f}s',
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
speed_path = RESULTS_DIR / "speed_comparison.png"
plt.savefig(speed_path, dpi=150)
print(f"  Saved speed comparison to: {speed_path}")
plt.close()

# Saving comparison results
comparison_results = {
    'neural_network': nn_metrics,
    'mle': mle_metrics,
    'speedup': float(speedup),
    'n_sequences': N_COMPARE
}

comparison_path = RESULTS_DIR / "nn_vs_mle_comparison.json"
with open(comparison_path, 'w') as f:
    json.dump(comparison_results, f, indent=2)
print(f"  Saved comparison results to: {comparison_path}")

# Summary
print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
print(f"\nKey Findings:")
print(f"  1. Speed: NN is {speedup:.1f}x faster than MLE")
print(f"  2. Accuracy (phi): NN MSE={nn_metrics['mse_phi']:.6f}, MLE MSE={mle_metrics['mse_phi']:.6f}")
print(f"  3. Accuracy (sigma_v): NN MSE={nn_metrics['mse_sigma_v']:.6f}, MLE MSE={mle_metrics['mse_sigma_v']:.6f}")

if nn_metrics['mse_phi'] < mle_metrics['mse_phi']:
    print(f"  → NN is more accurate for phi estimation")
else:
    print(f"  → MLE is more accurate for phi estimation")

if nn_metrics['mse_sigma_v'] < mle_metrics['mse_sigma_v']:
    print(f"  → NN is more accurate for sigma_v estimation")
else:
    print(f"  → MLE is more accurate for sigma_v estimation")

print(f"\nAll results saved to: {RESULTS_DIR}/")

