Human Height Prediction from Images (DINOv2 + Regression)

This project trains a vision model to **predict a person’s height (in centimeters)** from a single image and ships an interactive **Streamlit web app** for inference.

## Results (Test Set)
- **MSE on DINOv2 (frozen backbone + regression head):** `220.82`
- **MSE on fine-tuned model:** `134.64`

> These values are also referenced/compared in `Model_EDA/` artifacts.

---

## Project Overview

### What this repo contains
- **Data cleaning & splitting** using the Hugging Face `datasets` library.
- **Optional data augmentation** (bottom occlusion) to improve robustness when the full body is not visible.
- **Model training** using **facebook/dinov2-base** features + a small MLP regressor.
- **Fine-tuning** to further reduce error.
- A **Streamlit inference app** (`height-app/`) that loads one or more checkpoints and predicts height from uploaded images.

### High-level approach
1. Load a dataset of face/body images with associated height labels.
2. Resize and normalize images to the model’s expected format.
3. Pass images through a DINOv2 backbone, take the CLS embedding, and regress to a single height value.
4. Evaluate using **MSE** (and additional metrics in the model EDA workflow).

---

## Repository Structure

- `height-app/`
  - Streamlit app for interactive height prediction (`height-app/app.py`).
- `train_dinov2.py`
  - Trains a regression head on top of a **frozen** DINOv2 backbone and saves weights.
- `finetune.py`
  - Fine-tunes the DINOv2 backbone + regressor (differential learning rates) and saves “best” weights.
  - Writes `mse_history.csv` for plotting learning curves.
- `clean_data.py`
  - Pulls the dataset, filters missing height labels, and creates an 80/10/10 split saved to disk.
- `data_aug.py`
  - Creates the same split and applies **bottom occlusion augmentation** to the **training split only**, saved to disk.
- `Model_EDA/`
  - Error analysis utilities and reports for model performance.
- `EDA/`, `Plots/`, `Mean_baseline/`, `CNN_baseline/`
  - Additional exploration, baselines, and figures (project-specific experimentation).
- `Deliverables/` contains the Write-up & Presentation	

---

## Setup

### 1) Install dependencies
From the repo root:

```bash
pip install -r requirements.txt
```

Key libraries used:
- `torch`, `torchvision`, `transformers`
- `datasets`
- `streamlit`
- `numpy`, `pandas`, `matplotlib`

### 2) (Optional) Create the cleaned dataset
This downloads and prepares the dataset and saves it locally:

```bash
python clean_data.py
```

This produces:
- `./cleaned_data` (a Hugging Face dataset saved to disk)

### 3) (Optional) Create the augmented dataset
This applies bottom-occlusion augmentation to the training set only and saves the result:

```bash
python data_aug.py
```

This produces:
- `./augmented_data`

> Note: The training scripts in this repo expect `./augmented_data` by default.

---

## Training

### Train (frozen DINOv2 backbone)
Runs a short training loop (default: 5 epochs) and saves weights:

```bash
python train_dinov2.py
```

Output:
- `./dinov2.pth`

### Fine-tune (unfrozen backbone + regressor)
Fine-tunes with differential learning rates and saves best validation checkpoint:

```bash
python finetune.py
```

Outputs:
- `./dinov2_BEST.pth`
- `./mse_history.csv`

---

## Height Prediction App (Streamlit)

### Run the app
From the repo root:

```bash
streamlit run height-app/app.py
```

### How the app works
- Accepts `.jpg`, `.jpeg`, and `.png`.
- Automatically resizes and normalizes inputs for inference.
- If no image is uploaded, it uses `SAM.png` from the repo root as the default example.
- It attempts to load **any available** checkpoint from this candidate list (first found is used; the app can also load multiple and average predictions):

  - `dinov2_BEST_aug.pth`
  - `dinov2_BEST_unaug.pth`
  - `dinov2_base.pth`

### Output format
- Predicts height in **cm**
- Converts to **feet/inches**
- Displays an approximate range (based on prediction spread when multiple checkpoints are loaded)

---

## Model EDA (Error Analysis)

The `Model_EDA/` directory is dedicated to analyzing model errors on the held-out test set.

Included files (from `Model_EDA/README.md`):
- `model_error_eda.py` – runs evaluation and generates EDA artifacts
- `model_error_eda.ipynb` – notebook version
- `model_predictions.csv` – per-image predictions + residuals
- `model_comparison_metrics.csv` – summary metrics for checkpoints
- `model_error_report.md` – written summary of failures
- `plots/` – residual plots and MAE by height bin

Run from repo root:
```bash
python Model_EDA/model_error_eda.py
```

The analysis focuses on:
- Which true-height ranges have highest error
- Over/under-prediction bias patterns
- Worst misses on test set
- Augmented vs unaugmented comparisons (where available)

---

## Notes / Common Troubleshooting

- **No checkpoint found (app error):** ensure at least one of the expected `.pth` files exists in the repo root (see list above).
- **CPU vs GPU:** the app and scripts will use CUDA if available; otherwise they run on CPU.
- **Dataset not found:** if you haven’t created `./augmented_data`, run `python data_aug.py` (or modify the training scripts to point to `./cleaned_data` instead).


## Quickstart (minimal)
```bash
pip install -r requirements.txt
streamlit run height-app/app.py
```