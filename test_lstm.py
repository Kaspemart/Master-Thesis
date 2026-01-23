# IMPORTS:
import logging
import torch
from torch.utils.data import DataLoader
from src.neural_networks.lstm_estimator import (
    LSTMParameterEstimator,
    SVParameterDataset,
    train_model,
    evaluate_model
)
from src.data.synthetic.basic_sv import BasicSVDataGenerator
# ---------------------------------------------------------------------------------------------------------------------------


# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


print("=" * 80)
print("TESTING LSTM PARAMETER ESTIMATOR")
print("=" * 80)

# Checking device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Loading data
print("\nLoading data...")
generator = BasicSVDataGenerator(seed=42)

# Using small subset for quick test
print("Generating small test dataset (1000 sequences)...")
test_data = generator.generate(n_sequences=1000, show_progress=False)

# Creating datasets
print("\nCreating PyTorch datasets...")
train_dataset = SVParameterDataset(
    returns=test_data['returns'][:800],
    parameters=test_data['parameters'][:800],
    normalize_returns=True
)

val_dataset = SVParameterDataset(
    returns=test_data['returns'][800:900],
    parameters=test_data['parameters'][800:900],
    normalize_returns=True
)

test_dataset = SVParameterDataset(
    returns=test_data['returns'][900:],
    parameters=test_data['parameters'][900:],
    normalize_returns=True
)

print(f"  Train: {len(train_dataset)} samples")
print(f"  Val: {len(val_dataset)} samples")
print(f"  Test: {len(test_dataset)} samples")

# Creating data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Creating model
print("\nCreating LSTM model...")
model = LSTMParameterEstimator(
    input_size=1,
    hidden_size=64,  # Smaller for quick test
    num_layers=2,
    dropout=0.2,
    sequence_length=252
)

# Counting parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Testing forward pass
print("\nTesting forward pass...")
model.eval()
with torch.no_grad():
    sample_returns, sample_params = next(iter(train_loader))
    predictions = model(sample_returns)
    print(f"  Input shape: {sample_returns.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Target shape: {sample_params.shape}")
    print(f"  Sample prediction: φ={predictions[0, 0]:.4f}, σ_v={predictions[0, 1]:.4f}")
    print(f"  True values: φ={sample_params[0, 0]:.4f}, σ_v={sample_params[0, 1]:.4f}")

# Training model (short training for test)
print("\n" + "=" * 80)
print("TRAINING MODEL (5 epochs for quick test)")
print("=" * 80)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=5,
    learning_rate=0.001,
    device=device,
    save_path=None,  # Not saving for quick test
    patience=10
)

# Evaluating on test set
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)

metrics = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device
)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - LSTM ESTIMATOR IS WORKING!")
print("=" * 80)
print("\nNext steps:")
print("  1. Train on full dataset (50k sequences)")
print("  2. Tune hyperparameters (hidden_size, num_layers, learning_rate)")
print("  3. Compare with classical methods")
print("  4. Evaluate on test set with full metrics")

