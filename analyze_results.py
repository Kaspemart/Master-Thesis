# IMPORTS:
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from src.neural_networks.lstm_estimator import LSTMParameterEstimator, SVParameterDataset
from src.data.synthetic.basic_sv import BasicSVDataGenerator
from torch.utils.data import DataLoader
# ---------------------------------------------------------------------------------------------------------------------------

"""
Post-Training Analysis Script

This script analyzes the trained LSTM model and prepares results for comparison
with classical methods.
"""

def analyze_training_results() -> None:
    """Analyze training results and create visualizations."""
    
    print("=" * 80)
    print("ANALYZING TRAINING RESULTS")
    print("=" * 80)
    
    # Loading results
    results_dir = Path("results")
    model_path = Path("models/lstm_estimator_full.pt")
    
    # Loading metrics
    metrics_path = results_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("\nTest Set Performance:")
        print(f"  MSE (φ): {metrics['mse_phi']:.6f}")
        print(f"  MSE (σ_v): {metrics['mse_sigma_v']:.6f}")
        print(f"  MAE (φ): {metrics['mae_phi']:.6f}")
        print(f"  MAE (σ_v): {metrics['mae_sigma_v']:.6f}")
        print(f"  Bias (φ): {metrics['bias_phi']:.6f}")
        print(f"  Bias (σ_v): {metrics['bias_sigma_v']:.6f}")
        print(f"  Total MSE: {metrics['mse_total']:.6f}")
    else:
        print("Warning: Test metrics not found. Run evaluation first.")
        return
    
    # Loading model and creating detailed predictions
    print("\n" + "=" * 80)
    print("CREATING DETAILED PREDICTIONS")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMParameterEstimator(hidden_size=128, num_layers=2, dropout=0.2)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Loading test data
    generator = BasicSVDataGenerator(seed=42)
    test_data = generator.load_dataset('data/synthetic/basic_sv/test.npz')
    test_dataset = SVParameterDataset(
        returns=test_data['returns'],
        parameters=test_data['parameters'],
        normalize_returns=True
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Collecting all predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for returns, parameters in test_loader:
            returns = returns.to(device)
            predictions = model(returns)
            all_predictions.append(predictions.cpu().detach().numpy())
            all_targets.append(parameters.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Creating visualizations
    print("\nCreating visualizations...")
    
    # 1. Scatter plots: predicted vs true
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Phi scatter plot
    axes[0].scatter(targets[:, 0], predictions[:, 0], alpha=0.3, s=10)
    axes[0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                 [targets[:, 0].min(), targets[:, 0].max()], 
                 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('True φ', fontsize=12)
    axes[0].set_ylabel('Predicted φ', fontsize=12)
    axes[0].set_title('Predicted vs True φ', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sigma_v scatter plot
    axes[1].scatter(targets[:, 1], predictions[:, 1], alpha=0.3, s=10, color='orange')
    axes[1].plot([targets[:, 1].min(), targets[:, 1].max()], 
                 [targets[:, 1].min(), targets[:, 1].max()], 
                 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('True σ_v', fontsize=12)
    axes[1].set_ylabel('Predicted σ_v', fontsize=12)
    axes[1].set_title('Predicted vs True σ_v', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = results_dir / "predictions_scatter.png"
    plt.savefig(scatter_path, dpi=150)
    print(f"  Saved scatter plots to: {scatter_path}")
    plt.close()
    
    # 2. Error distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    errors_phi = predictions[:, 0] - targets[:, 0]
    errors_sigma_v = predictions[:, 1] - targets[:, 1]
    
    axes[0].hist(errors_phi, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    axes[0].axvline(errors_phi.mean(), color='g', linestyle='--', linewidth=2, 
                    label=f'Mean: {errors_phi.mean():.4f}')
    axes[0].set_xlabel('Error (Predicted - True)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution for φ', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(errors_sigma_v, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    axes[1].axvline(errors_sigma_v.mean(), color='g', linestyle='--', linewidth=2,
                    label=f'Mean: {errors_sigma_v.mean():.4f}')
    axes[1].set_xlabel('Error (Predicted - True)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Error Distribution for σ_v', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_path = results_dir / "error_distributions.png"
    plt.savefig(error_path, dpi=150)
    print(f"  Saved error distributions to: {error_path}")
    plt.close()
    
    # 3. Performance by parameter range
    print("\nAnalyzing performance by parameter range...")
    
    # Binning by true parameter values
    phi_bins = np.linspace(targets[:, 0].min(), targets[:, 0].max(), 5)
    sigma_v_bins = np.linspace(targets[:, 1].min(), targets[:, 1].max(), 5)
    
    phi_mae_by_range = []
    sigma_v_mae_by_range = []
    
    for i in range(len(phi_bins) - 1):
        mask = (targets[:, 0] >= phi_bins[i]) & (targets[:, 0] < phi_bins[i+1])
        if mask.sum() > 0:
            mae = np.mean(np.abs(errors_phi[mask]))
            phi_mae_by_range.append(mae)
        else:
            phi_mae_by_range.append(0)
    
    for i in range(len(sigma_v_bins) - 1):
        mask = (targets[:, 1] >= sigma_v_bins[i]) & (targets[:, 1] < sigma_v_bins[i+1])
        if mask.sum() > 0:
            mae = np.mean(np.abs(errors_sigma_v[mask]))
            sigma_v_mae_by_range.append(mae)
        else:
            sigma_v_mae_by_range.append(0)
    
    # Creating bar plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    phi_labels = [f"{phi_bins[i]:.2f}-{phi_bins[i+1]:.2f}" for i in range(len(phi_bins)-1)]
    axes[0].bar(phi_labels, phi_mae_by_range, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('True φ Range', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0].set_title('MAE by φ Range', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    sigma_v_labels = [f"{sigma_v_bins[i]:.2f}-{sigma_v_bins[i+1]:.2f}" for i in range(len(sigma_v_bins)-1)]
    axes[1].bar(sigma_v_labels, sigma_v_mae_by_range, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('True σ_v Range', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1].set_title('MAE by σ_v Range', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    range_path = results_dir / "performance_by_range.png"
    plt.savefig(range_path, dpi=150)
    print(f"  Saved performance by range to: {range_path}")
    plt.close()
    
    # Saving detailed predictions
    predictions_data = {
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'errors_phi': errors_phi.tolist(),
        'errors_sigma_v': errors_sigma_v.tolist()
    }
    
    predictions_path = results_dir / "detailed_predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"  Saved detailed predictions to: {predictions_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the visualizations in results/")
    print("  2. Implement classical benchmark (MLE or particle filter)")
    print("  3. Compare NN vs classical methods")
    print("  4. Document findings for thesis")


if __name__ == "__main__":
    analyze_training_results()

