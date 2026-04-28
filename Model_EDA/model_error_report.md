# Height Model Error Analysis

This folder analyzes where the saved height prediction models succeed and fail, with special focus on the held-out `test` split.

## Saved Models Compared

| model_name | split | samples | mse_cm2 | mae_cm | rmse_cm | bias_cm |
| --- | --- | --- | --- | --- | --- | --- |
| augmented_finetune | test | 673 | 127.97 | 8.93 | 11.31 | 2.49 |
| unaugmented_finetune | test | 673 | 132.97 | 9.69 | 11.53 | -2.56 |

Primary analysis model: **augmented_finetune**

## Test-Set Summary for Primary Model

- MSE: **127.97 cm^2**
- MAE: **8.93 cm**
- RMSE: **11.31 cm**
- Mean signed error (bias): **2.49 cm**
- Median absolute error: **7.17 cm**
- 90th percentile absolute error: **19.73 cm**

## Which Heights the Model Struggles With

- The hardest true-height range on the test set is **<150 cm**, with MAE **25.63 cm** across **4** samples.
- The easiest true-height range is **170-179 cm**, with MAE **4.40 cm**.
- The strongest underprediction happens in **200+ cm**, where mean signed error is **-20.03 cm**.
- The strongest overprediction happens in **<150 cm**, where mean signed error is **25.63 cm**.

## Error by Height Bin (Test Set, Primary Model)

| height_bin | samples | mean_true_height_cm | mse_cm2 | mae_cm | rmse_cm | bias_cm |
| --- | --- | --- | --- | --- | --- | --- |
| <150 | 4 | 144.02 | 680.81 | 25.63 | 26.09 | 25.63 |
| 150-159 | 165 | 155.92 | 333.44 | 17.62 | 18.26 | 17.62 |
| 160-169 | 114 | 165.23 | 36.53 | 5.31 | 6.04 | 5.29 |
| 170-179 | 264 | 175.65 | 29.7 | 4.4 | 5.45 | -2.81 |
| 180-189 | 92 | 185.21 | 75.35 | 7.67 | 8.68 | -7.26 |
| 190-199 | 28 | 193.98 | 243.0 | 14.49 | 15.59 | -14.49 |
| 200+ | 6 | 204.72 | 439.58 | 20.03 | 20.97 | -20.03 |

## Worst Test Predictions (Primary Model)

| id | true_height_cm | predicted_height_cm | error_cm | abs_error_cm | height_bin | gender_code | age |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2628 | 134.11 | 167.84 | 33.72 | 33.72 | <150 | 1 | 36 |
| 8732 | 155.75 | 187.86 | 32.11 | 32.11 | 150-159 | 0 | 43 |
| 4056 | 207.26 | 177.3 | -29.96 | 29.96 | 200+ | 0 | 43 |
| 6071 | 155.75 | 183.53 | 27.78 | 27.78 | 150-159 | 0 | 50 |
| 8757 | 155.75 | 183.41 | 27.66 | 27.66 | 150-159 | 0 | 38 |
| 451 | 155.45 | 182.82 | 27.37 | 27.37 | 150-159 | 0 | 46 |
| 6023 | 155.75 | 182.81 | 27.06 | 27.06 | 150-159 | 0 | 37 |
| 8787 | 155.45 | 181.94 | 26.49 | 26.49 | 150-159 | 0 | 43 |
| 552 | 155.75 | 181.68 | 25.93 | 25.93 | 150-159 | 0 | 30 |
| 6735 | 155.75 | 181.27 | 25.51 | 25.51 | 150-159 | 0 | 34 |
| 7432 | 155.75 | 181.23 | 25.48 | 25.48 | 150-159 | 0 | 46 |
| 7126 | 155.45 | 180.89 | 25.44 | 25.44 | 150-159 | 0 | 39 |
| 9469 | 155.45 | 180.81 | 25.36 | 25.36 | 150-159 | 0 | 48 |
| 1582 | 155.45 | 180.74 | 25.29 | 25.29 | 150-159 | 0 | 46 |
| 3988 | 198.12 | 172.89 | -25.23 | 25.23 | 190-199 | 0 | 41 |

## Generated Files

- `Model_EDA/model_predictions.csv`
- `Model_EDA/model_comparison_metrics.csv`
- `Model_EDA/worst_test_examples_augmented_finetune.csv`
- `Model_EDA/plots/model_comparison_test_rmse.png`
- `Model_EDA/plots/actual_vs_predicted_augmented_finetune.png`
- `Model_EDA/plots/error_distribution_augmented_finetune.png`
- `Model_EDA/plots/residuals_by_true_height_augmented_finetune.png`
- `Model_EDA/plots/mse_by_height_bin_augmented_finetune.png`
- `Model_EDA/plots/mae_by_height_bin_augmented_finetune.png`
- `Model_EDA/plots/bias_by_height_bin_augmented_finetune.png`

A positive error means the model predicted too tall. A negative error means the model predicted too short.
