# Refactoring Notes

This repository is a portfolio-oriented refactor of an earlier group academic project.

## What was preserved

- The original Euphoria dataset and terminology.
- The two core analytical ideas: supervised prediction and island segmentation.
- Tree-based machine learning, including XGBoost.
- Exploratory analysis of the supplied variables.

## What was changed

### 1. Raw CSV parsing

The dataset contains two row formats: standard CSV records and records wrapped as a single quoted field. The original notebook attempted to recover values using repeated positional string splits. That approach could assign the wrong source value to a feature.

The portfolio loader now parses every line with Python's `csv` module. If a row is received as one field, it is parsed a second time. This reconstructs all 19 columns for all 99,492 rows.

### 2. Target definition

The earlier project mixed classification language with regressors. In the supplied data, `happiness_index` is a continuous numeric field, so the corrected supervised task is regression.

### 3. Timestamp handling

`creation_time` is treated as a Unix timestamp and converted with `pd.to_datetime(..., unit="s")`. This avoids accidental 1970 timestamps caused by extracting the wrong token from a row.

### 4. Feature reconstruction

The portfolio version derives:
- `creation_year`
- `creation_month`
- `amenity_count`
- `cats_allowed`
- `dogs_allowed`

It does not duplicate `happiness_index` into another field.

### 5. Leakage prevention

The train/test split occurs before preprocessing. Imputation and encoding are fitted only on the training data and then applied to the test data.

### 6. Baseline and evaluation

A median baseline is reported alongside Extra Trees and XGBoost. Regression is evaluated with MAE, RMSE, and R² on an untouched 20% test split.

### 7. Clustering

K-Means is sensitive to extreme values, so clustering variables are winsorized at the 1st/99th percentiles before imputation and standardization. The number of clusters is no longer fixed in advance: candidate values are evaluated with both Silhouette Score and Davies-Bouldin Index. The final result is described cautiously because separation is weak.

### 8. Repository quality

The original 200+ cell notebook is replaced by:
- modular source files,
- a concise notebook,
- reproducible outputs,
- automated parser/feature tests,
- clear documentation.
