# Combined EDA Report

This report combines the `train`, `validation`, and `test` splits from `cleaned_data` into one analysis view while still tracking split-level differences.

## Dataset Coverage

- Total samples: **6,720**
- Unique IDs: **6,720**
- Duplicate IDs across combined splits: **0**
- Underlying Hugging Face dataset rows after concatenation: **6,720**

## Key Findings

- Heights span **79.2 cm** to **259.1 cm**, with mean **171.42 cm** and standard deviation **12.29 cm**.
- The feature with the strongest linear relationship to height is **weight_kg_clean** with correlation **0.51**.
- Average weight is **65.87 kg** and average age is **42.29 years**.
- The dataset still contains **546** weight entries and **8** age entries encoded as `-1`; these are excluded from cleaned summaries below.
- **96.5%** of images are portrait-oriented, which matters for resize/crop choices.
- **67.9%** of images use `RGBA`, so explicit RGB conversion during training is important and already matches the current training pipeline.

## Split Summary

| split      |   samples |   unique_ids |   mean_height_cm |   median_height_cm |   std_height_cm |   mean_weight_kg |   mean_age |
|:-----------|----------:|-------------:|-----------------:|-------------------:|----------------:|-----------------:|-----------:|
| test       |       673 |          673 |           171.19 |             170.69 |           11.88 |            65.83 |      42.4  |
| train      |      5442 |         5442 |           171.49 |             170.69 |           12.28 |            65.74 |      42.28 |
| validation |       605 |          605 |           171.07 |             170.69 |           12.82 |            67.12 |      42.33 |

## Numeric Summary

| index   |   height_cm |   weight_kg_clean |   age_clean |   image_width |   image_height |   aspect_ratio |
|:--------|------------:|------------------:|------------:|--------------:|---------------:|---------------:|
| count   |     6720    |           6174    |     6712    |       6720    |        6720    |        6720    |
| mean    |      171.42 |             65.87 |       42.29 |        378.05 |         587.03 |           0.62 |
| std     |       12.29 |             15.9  |       10.7  |        233.07 |         287.85 |           0.18 |
| min     |       79.25 |             38    |       14    |         70    |         159    |           0.21 |
| 25%     |      161.54 |             55    |       35    |        183    |         370    |           0.48 |
| 50%     |      170.69 |             60    |       41    |        371    |         562    |           0.64 |
| 75%     |      179.83 |             75    |       47    |        465    |         680    |           0.71 |
| max     |      259.08 |            202    |       97    |       2848    |        4032    |           2    |

## Gender-Code Summary

The dataset stores gender as integer codes; this report preserves those codes instead of guessing label names.

|   gender_code |   samples |   mean_height_cm |   median_height_cm |   mean_weight_kg |   mean_age |
|--------------:|----------:|-----------------:|-------------------:|-----------------:|-----------:|
|             0 |      2647 |           176.84 |             179.83 |            80.14 |      44.73 |
|             1 |      4073 |           167.9  |             167.64 |            56.37 |      40.72 |

## Orientation Summary

| split      | orientation   |   samples |
|:-----------|:--------------|----------:|
| test       | portrait      |       649 |
| test       | landscape     |        24 |
| train      | portrait      |      5248 |
| train      | landscape     |       174 |
| train      | square        |        20 |
| validation | portrait      |       590 |
| validation | landscape     |        15 |

## Missing-Value Summary

| column          |   missing_values |
|:----------------|-----------------:|
| height_cm       |                0 |
| weight_kg_clean |              546 |
| age_clean       |                8 |
| gender_code     |                0 |
| image_width     |                0 |
| image_height    |                0 |
| aspect_ratio    |                0 |
| channels        |                0 |

## Sentinel `-1` Values In Raw Metadata

| column    |   sentinel_neg1_count |
|:----------|----------------------:|
| height_cm |                     0 |
| weight_kg |                   546 |
| age       |                     8 |

## Generated Figures

- `EDA/plots/split_counts.png`
- `EDA/plots/height_distribution_overall.png`
- `EDA/plots/height_distribution_by_split.png`
- `EDA/plots/height_by_gender_boxplot.png`
- `EDA/plots/numeric_correlation_heatmap.png`
- `EDA/plots/height_vs_weight.png`
- `EDA/plots/height_vs_age.png`
- `EDA/plots/image_resolution_distribution.png`
- `EDA/plots/aspect_ratio_by_split.png`
- `EDA/plots/sample_images.png`
