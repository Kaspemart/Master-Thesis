# IMPORTS:
import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import json
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

print("=" * 80)
print("FULL-SCALE LSTM TRAINING ON 50K DATASET")
print("=" * 80)

# Checking device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Creating output directories
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Loading full datasets
print("\n" + "=" * 80)
print("LOADING DATASETS")
print("=" * 80)

generator = BasicSVDataGenerator(seed=42)

print("Loading training data...")
train_data = generator.load_dataset('data/synthetic/basic_sv/train.npz')
print(f"  Loaded {train_data['returns'].shape[0]:,} sequences")

print("Loading validation data...")
val_data = generator.load_dataset('data/synthetic/basic_sv/val.npz')
print(f"  Loaded {val_data['returns'].shape[0]:,} sequences")

print("Loading test data...")
test_data = generator.load_dataset('data/synthetic/basic_sv/test.npz')
print(f"  Loaded {test_data['returns'].shape[0]:,} sequences")

# Creating PyTorch datasets
print("\n" + "=" * 80)
print("CREATING PYTORCH DATASETS")
print("=" * 80)

train_dataset = SVParameterDataset(
    returns=train_data['returns'],
    parameters=train_data['parameters'],
    normalize_returns=True
)

val_dataset = SVParameterDataset(
    returns=val_data['returns'],
    parameters=val_data['parameters'],
    normalize_returns=True
)

test_dataset = SVParameterDataset(
    returns=test_data['returns'],
    parameters=test_data['parameters'],
    normalize_returns=True
)

# Creating data loaders
BATCH_SIZE = 128
NUM_WORKERS = 0  # Set to 0 on Windows, can increase on Linux

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if device == "cuda" else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if device == "cuda" else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if device == "cuda" else False
)

print(f"Batch size: {BATCH_SIZE}")
print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Creating model
print("\n" + "=" * 80)
print("CREATING LSTM MODEL")
print("=" * 80)

model = LSTMParameterEstimator(
    input_size=1,
    hidden_size=128,  # Reasonable size for full dataset
    num_layers=2,
    dropout=0.2,
    sequence_length=252
)

# Counting parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Training configuration
NUM_EPOCHS = 30  # Reduced from 50 for faster training
LEARNING_RATE = 0.001
PATIENCE = 10
MODEL_PATH = MODEL_DIR / "lstm_estimator_full.pt"

print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Early stopping patience: {PATIENCE}")
print(f"Model save path: {MODEL_PATH}")

# Training model
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print("This may take 1-3 hours depending on your hardware...")
print("Progress will be logged to 'training.log' and console\n")

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    device=device,
    save_path=MODEL_PATH,
    patience=PATIENCE
)

# Plotting training history
print("\n" + "=" * 80)
print("PLOTTING TRAINING HISTORY")
print("=" * 80)

if history and len(history['train_loss']) > 0 and len(history['val_loss']) > 0:
    try:
        fig, ax = plt.subplots(figsize=(10, 6))  # type: ignore[assignment]
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)  # type: ignore[arg-type]
        ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)  # type: ignore[arg-type]
        ax.set_xlabel('Epoch', fontsize=12)  # type: ignore[arg-type]
        ax.set_ylabel('Loss (MSE)', fontsize=12)  # type: ignore[arg-type]
        ax.set_title('Training History - LSTM Parameter Estimator', fontsize=14, fontweight='bold')  # type: ignore[arg-type]
        ax.legend(fontsize=11)  # type: ignore[arg-type]
        ax.grid(True, alpha=0.3)  # type: ignore[arg-type]
        plt.tight_layout()
        
        history_path = RESULTS_DIR / "training_history.png"
        plt.savefig(history_path, dpi=150)  # type: ignore[arg-type]
        print(f"Training history plot saved to: {history_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create training history plot: {e}")
        print(f"Warning: Could not create training history plot: {e}")
else:
    print("Warning: Training history is empty or incomplete, skipping plot")

# Evaluating on test set
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)

# Loading best model
if MODEL_PATH.exists():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
        
        # Running evaluation
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        # Saving metrics
        metrics_path = RESULTS_DIR / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        # Summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"Model saved to: {MODEL_PATH}")
        if history and len(history['train_loss']) > 0:
            print(f"Training history plot: {RESULTS_DIR / 'training_history.png'}")
        print(f"Test metrics: {metrics_path}")
        print(f"\nFinal Test Metrics:")
        print(f"  MSE (phi): {metrics['mse_phi']:.6f}")
        print(f"  MSE (sigma_v): {metrics['mse_sigma_v']:.6f}")
        print(f"  MAE (phi): {metrics['mae_phi']:.6f}")
        print(f"  MAE (sigma_v): {metrics['mae_sigma_v']:.6f}")
        print(f"  Bias (phi): {metrics['bias_phi']:.6f}")
        print(f"  Bias (sigma_v): {metrics['bias_sigma_v']:.6f}")
        print(f"  Total MSE: {metrics['mse_total']:.6f}")
        print("\n✓ Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        print(f"Error during evaluation: {e}")
        print("Training completed, but evaluation failed. Model is saved.")
else:
    print("Warning: Model checkpoint not found. Training may have been interrupted.")

