from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_from_disk


DATASET_PATH = REPO_ROOT / "cleaned_data"
OUTPUT_DIR = SCRIPT_DIR
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_PATH = OUTPUT_DIR / "combined_eda_report.md"
SAMPLE_SIZE = 12


def build_dataframe():
    dataset = load_from_disk(str(DATASET_PATH))

    frames = []
    sample_images = []

    for split_name, split_dataset in dataset.items():
        records = []

        for index, row in enumerate(split_dataset):
            image = row["image"]
            width, height_px = image.size
            aspect_ratio = width / height_px if height_px else np.nan
            orientation = (
                "square"
                if width == height_px
                else "landscape"
                if width > height_px
                else "portrait"
            )

            record = {
                "split": split_name,
                "id": row["id"],
                "height_cm": float(row["height"]),
                "weight_kg": float(row["weight"]),
                "gender_code": int(row["gender"]),
                "age": int(row["age"]),
                "image_width": int(width),
                "image_height": int(height_px),
                "aspect_ratio": float(aspect_ratio),
                "image_mode": image.mode,
                "orientation": orientation,
                "channels": len(image.getbands()),
            }
            records.append(record)

            if len(sample_images) < SAMPLE_SIZE:
                sample_images.append(
                    {
                        "split": split_name,
                        "id": row["id"],
                        "height_cm": float(row["height"]),
                        "image": image.copy(),
                    }
                )

        frames.append(pd.DataFrame.from_records(records))

    combined_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    combined_df = pd.concat(frames, ignore_index=True)
    combined_df["weight_kg_clean"] = combined_df["weight_kg"].replace(-1, np.nan)
    combined_df["age_clean"] = combined_df["age"].replace(-1, np.nan)
    return dataset, combined_dataset, combined_df, sample_images


def make_output_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def save_split_counts(df: pd.DataFrame):
    counts = df["split"].value_counts().reindex(["train", "validation", "test"])
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Samples Per Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of samples")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "split_counts.png", dpi=200)
    plt.close(fig)


def save_height_distributions(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["height_cm"], bins=30, color="#4c78a8", edgecolor="white")
    ax.set_title("Overall Height Distribution")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "height_distribution_overall.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}
    for split_name in ["train", "validation", "test"]:
        split_series = df.loc[df["split"] == split_name, "height_cm"]
        ax.hist(split_series, bins=25, alpha=0.45, label=split_name, color=colors[split_name])
    ax.set_title("Height Distribution By Split")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "height_distribution_by_split.png", dpi=200)
    plt.close(fig)


def save_boxplots(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = sorted(df["gender_code"].unique())
    data = [df.loc[df["gender_code"] == code, "height_cm"] for code in ordered]
    ax.boxplot(data, tick_labels=[f"gender_{code}" for code in ordered], patch_artist=True)
    ax.set_title("Height Distribution By Gender Code")
    ax.set_xlabel("Gender code")
    ax.set_ylabel("Height (cm)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "height_by_gender_boxplot.png", dpi=200)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame):
    cols = ["height_cm", "weight_kg_clean", "age_clean", "image_width", "image_height", "aspect_ratio", "channels"]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels=cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labels=cols)
    ax.set_title("Numeric Feature Correlation Heatmap")
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "numeric_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def save_scatterplots(df: pd.DataFrame):
    weight_df = df.dropna(subset=["weight_kg_clean"])
    fig, ax = plt.subplots(figsize=(8, 5))
    hexbin = ax.hexbin(weight_df["weight_kg_clean"], weight_df["height_cm"], gridsize=25, cmap="viridis", mincnt=1)
    ax.set_title("Height vs Weight")
    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Height (cm)")
    fig.colorbar(hexbin, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "height_vs_weight.png", dpi=200)
    plt.close(fig)

    age_df = df.dropna(subset=["age_clean"])
    fig, ax = plt.subplots(figsize=(8, 5))
    hexbin = ax.hexbin(age_df["age_clean"], age_df["height_cm"], gridsize=25, cmap="magma", mincnt=1)
    ax.set_title("Height vs Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Height (cm)")
    fig.colorbar(hexbin, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "height_vs_age.png", dpi=200)
    plt.close(fig)


def save_image_plots(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["image_width"],
        df["image_height"],
        c=df["height_cm"],
        cmap="plasma",
        alpha=0.5,
        s=18,
    )
    ax.set_title("Image Resolution Distribution")
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Image height (px)")
    fig.colorbar(scatter, ax=ax, label="Height (cm)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "image_resolution_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [df.loc[df["split"] == split_name, "aspect_ratio"] for split_name in ["train", "validation", "test"]]
    ax.boxplot(groups, tick_labels=["train", "validation", "test"], patch_artist=True)
    ax.set_title("Aspect Ratio By Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Aspect ratio (width / height)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "aspect_ratio_by_split.png", dpi=200)
    plt.close(fig)


def save_sample_grid(sample_images):
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten()

    for ax, sample in zip(axes, sample_images):
        ax.imshow(sample["image"].convert("RGB"))
        ax.set_title(f'{sample["split"]} | id={sample["id"]}\n{sample["height_cm"]:.1f} cm', fontsize=9)
        ax.axis("off")

    for ax in axes[len(sample_images):]:
        ax.axis("off")

    fig.suptitle("Sample Images Across Combined Dataset", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sample_images.png", dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def build_report(df: pd.DataFrame, combined_dataset):
    split_summary = (
        df.groupby("split")
        .agg(
            samples=("id", "size"),
            unique_ids=("id", "nunique"),
            mean_height_cm=("height_cm", "mean"),
            median_height_cm=("height_cm", "median"),
            std_height_cm=("height_cm", "std"),
            mean_weight_kg=("weight_kg_clean", "mean"),
            mean_age=("age_clean", "mean"),
        )
        .reset_index()
    )

    numeric_summary = (
        df[["height_cm", "weight_kg_clean", "age_clean", "image_width", "image_height", "aspect_ratio"]]
        .describe()
        .reset_index()
    )

    gender_summary = (
        df.groupby("gender_code")
        .agg(
            samples=("id", "size"),
            mean_height_cm=("height_cm", "mean"),
            median_height_cm=("height_cm", "median"),
            mean_weight_kg=("weight_kg_clean", "mean"),
            mean_age=("age_clean", "mean"),
        )
        .reset_index()
        .sort_values("gender_code")
    )

    orientation_summary = (
        df.groupby(["split", "orientation"])
        .size()
        .rename("samples")
        .reset_index()
        .sort_values(["split", "samples"], ascending=[True, False])
    )

    missing_summary = (
        df[["height_cm", "weight_kg_clean", "age_clean", "gender_code", "image_width", "image_height", "aspect_ratio", "channels"]]
        .isna()
        .sum()
        .rename_axis("column")
        .reset_index(name="missing_values")
    )

    sentinel_summary = pd.DataFrame(
        {
            "column": ["height_cm", "weight_kg", "age"],
            "sentinel_neg1_count": [
                int((df["height_cm"] == -1).sum()),
                int((df["weight_kg"] == -1).sum()),
                int((df["age"] == -1).sum()),
            ],
        }
    )

    duplicate_ids = int(df["id"].duplicated().sum())
    corr = df[["height_cm", "weight_kg_clean", "age_clean", "image_width", "image_height", "aspect_ratio", "channels"]].corr(numeric_only=True)
    strongest_height_corr = corr["height_cm"].drop("height_cm").abs().sort_values(ascending=False)
    top_feature = strongest_height_corr.index[0]
    top_feature_value = corr.loc["height_cm", top_feature]

    overall = {
        "samples": len(df),
        "unique_ids": df["id"].nunique(),
        "height_mean": df["height_cm"].mean(),
        "height_std": df["height_cm"].std(),
        "height_min": df["height_cm"].min(),
        "height_max": df["height_cm"].max(),
        "weight_mean": df["weight_kg_clean"].mean(),
        "age_mean": df["age_clean"].mean(),
        "portrait_share": (df["orientation"] == "portrait").mean(),
        "rgba_share": (df["image_mode"] == "RGBA").mean(),
        "weight_missing": int(df["weight_kg_clean"].isna().sum()),
        "age_missing": int(df["age_clean"].isna().sum()),
    }

    lines = [
        "# Combined EDA Report",
        "",
        "This report combines the `train`, `validation`, and `test` splits from `cleaned_data` into one analysis view while still tracking split-level differences.",
        "",
        "## Dataset Coverage",
        "",
        f"- Total samples: **{overall['samples']:,}**",
        f"- Unique IDs: **{overall['unique_ids']:,}**",
        f"- Duplicate IDs across combined splits: **{duplicate_ids:,}**",
        f"- Underlying Hugging Face dataset rows after concatenation: **{len(combined_dataset):,}**",
        "",
        "## Key Findings",
        "",
        f"- Heights span **{overall['height_min']:.1f} cm** to **{overall['height_max']:.1f} cm**, with mean **{overall['height_mean']:.2f} cm** and standard deviation **{overall['height_std']:.2f} cm**.",
        f"- The feature with the strongest linear relationship to height is **{top_feature}** with correlation **{top_feature_value:.2f}**.",
        f"- Average weight is **{overall['weight_mean']:.2f} kg** and average age is **{overall['age_mean']:.2f} years**.",
        f"- The dataset still contains **{overall['weight_missing']:,}** weight entries and **{overall['age_missing']:,}** age entries encoded as `-1`; these are excluded from cleaned summaries below.",
        f"- **{overall['portrait_share'] * 100:.1f}%** of images are portrait-oriented, which matters for resize/crop choices.",
        f"- **{overall['rgba_share'] * 100:.1f}%** of images use `RGBA`, so explicit RGB conversion during training is important and already matches the current training pipeline.",
        "",
        "## Split Summary",
        "",
        markdown_table(split_summary.round(2)),
        "",
        "## Numeric Summary",
        "",
        markdown_table(numeric_summary.round(2)),
        "",
        "## Gender-Code Summary",
        "",
        "The dataset stores gender as integer codes; this report preserves those codes instead of guessing label names.",
        "",
        markdown_table(gender_summary.round(2)),
        "",
        "## Orientation Summary",
        "",
        markdown_table(orientation_summary),
        "",
        "## Missing-Value Summary",
        "",
        markdown_table(missing_summary),
        "",
        "## Sentinel `-1` Values In Raw Metadata",
        "",
        markdown_table(sentinel_summary),
        "",
        "## Generated Figures",
        "",
        "- `EDA/plots/split_counts.png`",
        "- `EDA/plots/height_distribution_overall.png`",
        "- `EDA/plots/height_distribution_by_split.png`",
        "- `EDA/plots/height_by_gender_boxplot.png`",
        "- `EDA/plots/numeric_correlation_heatmap.png`",
        "- `EDA/plots/height_vs_weight.png`",
        "- `EDA/plots/height_vs_age.png`",
        "- `EDA/plots/image_resolution_distribution.png`",
        "- `EDA/plots/aspect_ratio_by_split.png`",
        "- `EDA/plots/sample_images.png`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    make_output_dirs()
    _, combined_dataset, df, sample_images = build_dataframe()
    save_split_counts(df)
    save_height_distributions(df)
    save_boxplots(df)
    save_correlation_heatmap(df)
    save_scatterplots(df)
    save_image_plots(df)
    save_sample_grid(sample_images)
    build_report(df, combined_dataset)
    print(f"Saved report to {REPORT_PATH}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
