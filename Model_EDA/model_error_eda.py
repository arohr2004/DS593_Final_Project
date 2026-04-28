from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mpl-cache"))
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".hf-home"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(SCRIPT_DIR / ".hf-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_from_disk
from torchvision import transforms
from transformers import Dinov2Config, Dinov2Model


DATASET_PATH = REPO_ROOT / "cleaned_data"
OUTPUT_DIR = SCRIPT_DIR
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_PATH = OUTPUT_DIR / "model_error_report.md"
PREDICTIONS_PATH = OUTPUT_DIR / "model_predictions.csv"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison_metrics.csv"
HEIGHT_BIN_EDGES = [0, 150, 160, 170, 180, 190, 200, 300]
HEIGHT_BIN_LABELS = ["<150", "150-159", "160-169", "170-179", "180-189", "190-199", "200+"]


class DINOv2ForHeightRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = Dinov2Model(
            Dinov2Config(
                image_size=518,
                patch_size=14,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
        )
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_token)
        return prediction.squeeze(-1)


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

PRIMARY_MODEL_NAME = "augmented_finetune"
PRIMARY_CHECKPOINT = REPO_ROOT / "dinov2_BEST_aug.pth"
OPTIONAL_COMPARISON_CHECKPOINTS = {
    "unaugmented_finetune": REPO_ROOT / "dinov2_BEST_unaug.pth",
}
BATCH_SIZE = 16


def make_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: Path, device: str) -> nn.Module:
    model = DINOv2ForHeightRegression().to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_base_dataframe(test_only: bool = False) -> tuple[dict, pd.DataFrame]:
    dataset = load_from_disk(str(DATASET_PATH))
    rows = []
    for split_name, split_dataset in dataset.items():
        if test_only and split_name != "test":
            continue
        for row in split_dataset:
            image = row["image"]
            rows.append(
                {
                    "split": split_name,
                    "id": row["id"],
                    "true_height_cm": float(row["height"]),
                    "weight_kg": float(row["weight"]),
                    "gender_code": int(row["gender"]),
                    "age": int(row["age"]),
                    "image_width": int(image.size[0]),
                    "image_height": int(image.size[1]),
                    "image_mode": image.mode,
                }
            )
    df = pd.DataFrame(rows)
    df["height_bin"] = pd.cut(
        df["true_height_cm"],
        bins=HEIGHT_BIN_EDGES,
        labels=HEIGHT_BIN_LABELS,
        right=False,
    )
    return dataset, df


def predict_for_checkpoint(dataset, metadata_df: pd.DataFrame, checkpoint_name: str, checkpoint_path: Path, device: str) -> pd.DataFrame:
    model = load_model(checkpoint_path, device)
    predictions = []

    for split_name, split_dataset in dataset.items():
        if split_name != "test":
            continue
        batch_tensors = []
        batch_meta = []
        for row in split_dataset:
            pixel_values = IMAGE_TRANSFORM(row["image"].convert("RGB"))
            batch_tensors.append(pixel_values)
            batch_meta.append((row["id"], split_name))

            if len(batch_tensors) < BATCH_SIZE:
                continue

            with torch.no_grad():
                batch = torch.stack(batch_tensors).to(device)
                batch_preds = model(batch).detach().cpu().numpy().tolist()
            for pred, (row_id, row_split) in zip(batch_preds, batch_meta):
                predictions.append(
                    {
                        "split": row_split,
                        "id": row_id,
                        "model_name": checkpoint_name,
                        "predicted_height_cm": float(pred),
                    }
                )
            batch_tensors = []
            batch_meta = []

        if batch_tensors:
            with torch.no_grad():
                batch = torch.stack(batch_tensors).to(device)
                batch_preds = model(batch).detach().cpu().numpy().tolist()
            for pred, (row_id, row_split) in zip(batch_preds, batch_meta):
                predictions.append(
                    {
                        "split": row_split,
                        "id": row_id,
                        "model_name": checkpoint_name,
                        "predicted_height_cm": float(pred),
                    }
                )

    pred_df = pd.DataFrame(predictions)
    merged = metadata_df.merge(pred_df, on=["split", "id"], how="inner")
    merged["error_cm"] = merged["predicted_height_cm"] - merged["true_height_cm"]
    merged["abs_error_cm"] = merged["error_cm"].abs()
    merged["squared_error_cm2"] = merged["error_cm"] ** 2
    return merged


def evaluate_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset, metadata_df = build_base_dataframe(test_only=True)
    device = resolve_device()
    all_predictions = [
        predict_for_checkpoint(dataset, metadata_df, PRIMARY_MODEL_NAME, PRIMARY_CHECKPOINT, device)
    ]

    for checkpoint_name, checkpoint_path in OPTIONAL_COMPARISON_CHECKPOINTS.items():
        if checkpoint_path.exists():
            all_predictions.append(
                predict_for_checkpoint(dataset, metadata_df, checkpoint_name, checkpoint_path, device)
            )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    comparison_df = (
        predictions_df.groupby(["model_name", "split"])
        .agg(
            samples=("id", "size"),
            mse_cm2=("squared_error_cm2", "mean"),
            mae_cm=("abs_error_cm", "mean"),
            rmse_cm=("squared_error_cm2", lambda s: float(np.sqrt(np.mean(s)))),
            bias_cm=("error_cm", "mean"),
        )
        .reset_index()
        .sort_values(["split", "rmse_cm"])
    )
    return predictions_df, comparison_df


def save_predictions(predictions_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    comparison_df.to_csv(COMPARISON_PATH, index=False)


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    test_df = comparison_df.loc[comparison_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(test_df["model_name"], test_df["rmse_cm"], color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Test RMSE by Saved Height Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE (cm)")
    for bar, value in zip(bars, test_df["rmse_cm"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison_test_rmse.png", dpi=200)
    plt.close(fig)


def plot_actual_vs_predicted(model_df: pd.DataFrame, model_name: str) -> None:
    test_df = model_df.loc[model_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(test_df["true_height_cm"], test_df["predicted_height_cm"], alpha=0.45, s=20, color="#2a6f97")
    min_height = min(test_df["true_height_cm"].min(), test_df["predicted_height_cm"].min())
    max_height = max(test_df["true_height_cm"].max(), test_df["predicted_height_cm"].max())
    fit_slope, fit_intercept = np.polyfit(test_df["true_height_cm"], test_df["predicted_height_cm"], 1)
    fit_x = np.array([min_height, max_height])
    fit_y = fit_slope * fit_x + fit_intercept
    ax.plot(
        [min_height, max_height],
        [min_height, max_height],
        linestyle="--",
        color="#d62728",
        label="Ideal line (y = x)",
    )
    ax.plot(
        fit_x,
        fit_y,
        linestyle="-",
        color="#1b4332",
        linewidth=2,
        label=f"Best-fit line (y = {fit_slope:.2f}x + {fit_intercept:.2f})",
    )
    ax.set_title(f"Test Set: Actual vs Predicted Height ({model_name})")
    ax.set_xlabel("True height (cm)")
    ax.set_ylabel("Predicted height (cm)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"actual_vs_predicted_{model_name}.png", dpi=200)
    plt.close(fig)


def plot_error_distribution(model_df: pd.DataFrame, model_name: str) -> None:
    test_df = model_df.loc[model_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(test_df["error_cm"], bins=30, color="#8ecae6", edgecolor="white")
    ax.axvline(0, color="#d62728", linestyle="--")
    ax.set_title(f"Test Residual Distribution ({model_name})")
    ax.set_xlabel("Prediction error (cm)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"error_distribution_{model_name}.png", dpi=200)
    plt.close(fig)


def plot_residuals_by_height(model_df: pd.DataFrame, model_name: str) -> None:
    test_df = model_df.loc[model_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        test_df["true_height_cm"],
        test_df["error_cm"],
        c=test_df["abs_error_cm"],
        cmap="viridis",
        alpha=0.55,
        s=22,
    )
    fit_slope, fit_intercept = np.polyfit(test_df["true_height_cm"], test_df["error_cm"], 1)
    fit_x = np.array([test_df["true_height_cm"].min(), test_df["true_height_cm"].max()])
    fit_y = fit_slope * fit_x + fit_intercept
    ax.axhline(0, color="#d62728", linestyle="--", label="Zero-error line")
    ax.plot(
        fit_x,
        fit_y,
        linestyle="-",
        color="#1b4332",
        linewidth=2,
        label=f"Best-fit line (y = {fit_slope:.2f}x + {fit_intercept:.2f})",
    )
    ax.set_title(f"Test Residuals by True Height ({model_name})")
    ax.set_xlabel("True height (cm)")
    ax.set_ylabel("Prediction error (cm)")
    fig.colorbar(scatter, ax=ax, label="Absolute error (cm)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"residuals_by_true_height_{model_name}.png", dpi=200)
    plt.close(fig)


def plot_height_bin_errors(model_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    test_df = model_df.loc[model_df["split"] == "test"].copy()
    bin_summary = (
        test_df.groupby("height_bin", observed=False)
        .agg(
            samples=("id", "size"),
            mean_true_height_cm=("true_height_cm", "mean"),
            mse_cm2=("squared_error_cm2", "mean"),
            mae_cm=("abs_error_cm", "mean"),
            rmse_cm=("squared_error_cm2", lambda s: float(np.sqrt(np.mean(s)))),
            bias_cm=("error_cm", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(bin_summary["height_bin"].astype(str), bin_summary["mse_cm2"], color="#577590")
    ax.set_title(f"Test MSE by True Height Bin ({model_name})")
    ax.set_xlabel("True height bin (cm)")
    ax.set_ylabel("MSE (cm^2)")
    for bar, value, samples in zip(bars, bin_summary["mse_cm2"], bin_summary["samples"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}\n(n={samples})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"mse_by_height_bin_{model_name}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(bin_summary["height_bin"].astype(str), bin_summary["mae_cm"], color="#219ebc")
    ax.set_title(f"Test MAE by True Height Bin ({model_name})")
    ax.set_xlabel("True height bin (cm)")
    ax.set_ylabel("MAE (cm)")
    for bar, value, samples in zip(bars, bin_summary["mae_cm"], bin_summary["samples"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}\n(n={samples})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"mae_by_height_bin_{model_name}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(bin_summary["height_bin"].astype(str), bin_summary["bias_cm"], color="#fb8500")
    ax.axhline(0, color="#222222", linestyle="--")
    ax.set_title(f"Test Bias by True Height Bin ({model_name})")
    ax.set_xlabel("True height bin (cm)")
    ax.set_ylabel("Mean signed error (cm)")
    for bar, value in zip(bars, bin_summary["bias_cm"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"bias_by_height_bin_{model_name}.png", dpi=200)
    plt.close(fig)

    return bin_summary


def plot_worst_examples(model_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    test_df = model_df.loc[model_df["split"] == "test"].copy()
    worst = test_df.sort_values("abs_error_cm", ascending=False).head(15)
    worst.to_csv(OUTPUT_DIR / f"worst_test_examples_{model_name}.csv", index=False)
    return worst


def markdown_table(df: pd.DataFrame) -> str:
    table_df = df.copy()
    headers = [str(col) for col in table_df.columns]
    rows = [[str(value) for value in row] for row in table_df.to_numpy().tolist()]

    separator = ["---"] * len(headers)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    markdown_rows.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(markdown_rows)


def build_report(predictions_df: pd.DataFrame, comparison_df: pd.DataFrame, bin_summary: pd.DataFrame, worst_examples: pd.DataFrame) -> None:
    primary_df = predictions_df.loc[predictions_df["model_name"] == PRIMARY_MODEL_NAME].copy()
    test_df = primary_df.loc[primary_df["split"] == "test"].copy()

    overall = {
        "mse": test_df["squared_error_cm2"].mean(),
        "mae": test_df["abs_error_cm"].mean(),
        "rmse": float(np.sqrt(np.mean(test_df["squared_error_cm2"]))),
        "bias": test_df["error_cm"].mean(),
        "median_ae": test_df["abs_error_cm"].median(),
        "p90_ae": test_df["abs_error_cm"].quantile(0.9),
    }

    hardest_bin = bin_summary.loc[bin_summary["mae_cm"].idxmax()]
    easiest_bin = bin_summary.loc[bin_summary["mae_cm"].idxmin()]
    strongest_under = bin_summary.loc[bin_summary["bias_cm"].idxmin()]
    strongest_over = bin_summary.loc[bin_summary["bias_cm"].idxmax()]

    lines = [
        "# Height Model Error Analysis",
        "",
        "This folder analyzes where the saved height prediction models succeed and fail, with special focus on the held-out `test` split.",
        "",
        "## Saved Models Compared",
        "",
        markdown_table(comparison_df.round(2)),
        "",
        f"Primary analysis model: **{PRIMARY_MODEL_NAME}**",
        "",
        "## Test-Set Summary for Primary Model",
        "",
        f"- MSE: **{overall['mse']:.2f} cm^2**",
        f"- MAE: **{overall['mae']:.2f} cm**",
        f"- RMSE: **{overall['rmse']:.2f} cm**",
        f"- Mean signed error (bias): **{overall['bias']:.2f} cm**",
        f"- Median absolute error: **{overall['median_ae']:.2f} cm**",
        f"- 90th percentile absolute error: **{overall['p90_ae']:.2f} cm**",
        "",
        "## Which Heights the Model Struggles With",
        "",
        f"- The hardest true-height range on the test set is **{hardest_bin['height_bin']} cm**, with MAE **{hardest_bin['mae_cm']:.2f} cm** across **{int(hardest_bin['samples'])}** samples.",
        f"- The easiest true-height range is **{easiest_bin['height_bin']} cm**, with MAE **{easiest_bin['mae_cm']:.2f} cm**.",
        f"- The strongest underprediction happens in **{strongest_under['height_bin']} cm**, where mean signed error is **{strongest_under['bias_cm']:.2f} cm**.",
        f"- The strongest overprediction happens in **{strongest_over['height_bin']} cm**, where mean signed error is **{strongest_over['bias_cm']:.2f} cm**.",
        "",
        "## Error by Height Bin (Test Set, Primary Model)",
        "",
        markdown_table(bin_summary.round(2)),
        "",
        "## Worst Test Predictions (Primary Model)",
        "",
        markdown_table(
            worst_examples[
                ["id", "true_height_cm", "predicted_height_cm", "error_cm", "abs_error_cm", "height_bin", "gender_code", "age"]
            ].round(2)
        ),
        "",
        "## Generated Files",
        "",
        f"- `Model_EDA/model_predictions.csv`",
        f"- `Model_EDA/model_comparison_metrics.csv`",
        f"- `Model_EDA/worst_test_examples_{PRIMARY_MODEL_NAME}.csv`",
        f"- `Model_EDA/plots/model_comparison_test_rmse.png`",
        f"- `Model_EDA/plots/actual_vs_predicted_{PRIMARY_MODEL_NAME}.png`",
        f"- `Model_EDA/plots/error_distribution_{PRIMARY_MODEL_NAME}.png`",
        f"- `Model_EDA/plots/residuals_by_true_height_{PRIMARY_MODEL_NAME}.png`",
        f"- `Model_EDA/plots/mse_by_height_bin_{PRIMARY_MODEL_NAME}.png`",
        f"- `Model_EDA/plots/mae_by_height_bin_{PRIMARY_MODEL_NAME}.png`",
        f"- `Model_EDA/plots/bias_by_height_bin_{PRIMARY_MODEL_NAME}.png`",
        "",
        "A positive error means the model predicted too tall. A negative error means the model predicted too short.",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    make_output_dirs()
    predictions_df, comparison_df = evaluate_models()
    save_predictions(predictions_df, comparison_df)
    plot_model_comparison(comparison_df)

    primary_df = predictions_df.loc[predictions_df["model_name"] == PRIMARY_MODEL_NAME].copy()
    plot_actual_vs_predicted(primary_df, PRIMARY_MODEL_NAME)
    plot_error_distribution(primary_df, PRIMARY_MODEL_NAME)
    plot_residuals_by_height(primary_df, PRIMARY_MODEL_NAME)
    bin_summary = plot_height_bin_errors(primary_df, PRIMARY_MODEL_NAME)
    worst_examples = plot_worst_examples(primary_df, PRIMARY_MODEL_NAME)
    build_report(predictions_df, comparison_df, bin_summary, worst_examples)

    print(f"Saved report to {REPORT_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
