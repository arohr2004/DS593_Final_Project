# Model EDA

This folder is for error analysis of the saved height prediction model.

## Files

- `model_error_eda.py`
  Runs the model on the held-out `test` split and generates prediction-focused EDA artifacts.
- `model_error_eda.ipynb`
  Notebook version for exploration and presentation.
- `model_predictions.csv`
  Per-image predictions and residuals after running the script.
- `model_comparison_metrics.csv`
  Summary metrics for the saved checkpoints on the test set.
- `model_error_report.md`
  Written summary of where the primary model struggles.
- `plots/`
  Generated figures such as residual plots and MAE by height bin.

## Focus of the Analysis

The analysis is designed to answer questions like:

- Which true-height ranges have the highest absolute error?
- Does the model systematically overpredict shorter people or underpredict taller people?
- How large are the worst misses on the held-out test set?
- How does the augmented checkpoint compare with the unaugmented checkpoint?

It reports `MSE`, `MAE`, `RMSE`, and signed bias so the model-EDA outputs stay consistent with the rest of the project while still being easy to interpret.

## Run

From the repo root:

```bash
python Model_EDA/model_error_eda.py
```

The script uses the saved checkpoint `dinov2_BEST_aug.pth` as the primary model and evaluates on `cleaned_data/test`.
