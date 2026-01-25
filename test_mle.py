# IMPORTS:
import time
from src.models.sv_basic import BasicSVModel
from src.classical.mle import estimate_parameters_mle
# ---------------------------------------------------------------------------------------------------------------------------

print("=" * 80)
print("TESTING MLE IMPLEMENTATION")
print("=" * 80)

# Creating test data with known parameters
print("\nGenerating test data...")
model = BasicSVModel(phi=0.95, sigma_v=0.2)
returns, log_vol = model.simulate(T=252, seed=42)

print(f"True parameters: phi={model.phi}, sigma_v={model.sigma_v}")
print(f"Return sequence length: {len(returns)}")

# Testing MLE estimation
print("\nRunning MLE estimation (this may take 30-60 seconds)...")
print("Using particle filter with 500 particles for faster testing...")

start_time = time.time()
phi_est, sigma_v_est, info = estimate_parameters_mle(
    returns=returns,
    n_particles=500,  # Smaller for faster testing
    phi_init=0.9,
    sigma_v_init=0.2,
    n_restarts=2,  # Fewer restarts for testing
    seed=42
)
elapsed_time = time.time() - start_time

print(f"\nResults:")
print(f"  True:        phi={model.phi:.4f}, sigma_v={model.sigma_v:.4f}")
print(f"  Estimated:   phi={phi_est:.4f}, sigma_v={sigma_v_est:.4f}")
print(f"  Error:       phi={abs(phi_est - model.phi):.4f}, sigma_v={abs(sigma_v_est - model.sigma_v):.4f}")
print(f"  Log-likelihood: {info['log_likelihood']:.2f}")
print(f"  Optimization time: {elapsed_time:.2f} seconds")
print(f"  Success: {info['success']}")

print("\n" + "=" * 80)
print("MLE TEST COMPLETE")
print("=" * 80)

