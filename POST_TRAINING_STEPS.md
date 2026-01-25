# Post-Training Checklist and Next Steps

After `train_full.py` completes, follow these steps:

## 1. Check Training Results

```bash
# View training log
cat training.log | tail -50

# Check if model was saved
ls -lh models/lstm_estimator_full.pt

# Check results
ls -lh results/
```

## 2. Analyze Results

Run the analysis script:
```bash
uv run python analyze_results.py
```

This will:
- Load test metrics
- Create scatter plots (predicted vs true)
- Create error distribution plots
- Analyze performance by parameter range
- Save detailed predictions

## 3. Review Key Metrics

Check `results/test_metrics.json`:
- **MSE (φ)**: Should be < 0.01 for good performance
- **MSE (σ_v)**: Should be < 0.01 for good performance
- **Bias**: Should be close to 0 (no systematic over/under-estimation)

## 4. Next Steps Based on Results

### If Results Are Good (MSE < 0.01):
✅ Proceed to implement classical benchmark
✅ Prepare for comparison

### If Results Need Improvement:
- Try hyperparameter tuning
- Increase model size (hidden_size=256)
- Train for more epochs
- Check for overfitting (val loss >> train loss)

## 5. Implement Classical Benchmark

Next priority: Implement MLE or particle filter for comparison.

See: `src/classical/` (to be created)

## 6. Comparison Analysis

Once classical method is implemented:
- Run both on test set
- Compare accuracy (MSE, MAE)
- Compare computation time
- Create comparison visualizations

## 7. Document Findings

Update thesis with:
- Training results
- Model performance
- Comparison with classical methods
- Visualizations

