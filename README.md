# Masai_AIML_Capstone

Masai Capstone — Delhivery Last-Mile Delay Intelligence Pipeline

## Project overview

This project uses synthetic shipment data to answer three operational questions:

1. How many hours will a parcel take to arrive?
2. Which parcels are likely to miss their promised delivery time?
3. Which hubs need different operational playbooks?

No real Delhivery shipment records are used.

### Current progress

- Part A: Data cleaning, feature preparation and exploratory analysis completed.
- Part B: Delivery-hours regression and evaluation completed.
- Part C: Delay classification, feature selection, tuning, threshold selection and final test evaluation completed.
- Part D: Hub segmentation, model packaging and batch scoring pending.

## How to run

Run commands from the repository's main folder.

### Environment setup

Create and activate a virtual environment on macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The exact versions used are pinned in `requirements.txt`:

```text
pandas==2.3.3
numpy==2.5.4
scikit-learn==1.8.0
joblib==1.6.0
```

### Execution order

The three generated input CSVs are committed in `data/`. To reproduce them using the provided, unchanged generator:

```bash
python generate_data.py
```

Run the completed analysis scripts in this order:

```bash
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
```

Commands for Part D will be added when its scripts are completed.

### Main files

| File | Purpose |
|---|---|
| `generate_data.py` | Provided synthetic-data generator |
| `data/hubs.csv` | Information about the 24 hubs |
| `data/raw_shipments.csv` | Historical shipments before cleaning |
| `data/new_shipments.csv` | 48 new shipments for later batch scoring |
| `src/shipment_prep.py` | Shared constants, cleaning and feature preparation |
| `part_a_clean_eda.py` | Creates the modelling table and Part A outputs |
| `data/clean_shipments.csv` | Intermediate cleaned historical data |
| `data/modelling_table.csv` | Prepared historical data used for modelling |
| `part_b_regression.py` | Compares delivery-hours regression models |
| `part_c_classification.py` | Compares delay classifiers, selects features, tunes the forest and selects an alert threshold |
| `inspect_data.py` | Optional helper for inspecting data |
| `requirements.txt` | Exact library versions |
| `outputs/` | Calculated results saved as CSV files |

## Data findings

### Cleaning and preparation

The historical input contained 1,224 rows. Removing 24 exact duplicates left 1,200 shipments.

Before filling missing values, the deduplicated data contained:

- 36 missing distances.
- 24 missing parcel weights.
- 12 missing declared values.
- 18 missing payment modes.

Five additional parcel weights above 100 kg were treated as missing according to the project instructions.

Cleaning standardized service names from 12 spellings to three categories, filled missing payment modes with `Unknown`, converted declared values to numeric INR amounts, and parsed dispatch timestamps.

Missing numeric values were filled with the historical medians:

| Column | Median |
|---|---:|
| `distance_km` | 314.95 km |
| `parcel_weight_kg` | 1.83 kg |
| `declared_value_inr` | INR 1,121.50 |

Feature preparation joined hub information and created `promised_hours`, `is_cod`, `is_weekend`, `dispatch_hour` and `log_declared_value`.

The modelling table retained all 1,200 shipments and had no missing values in the required features and targets. There were 278 delayed shipments, representing 23.17% of the total.

A shipment was labelled delayed when its actual delivery hours exceeded its promised hours.

The same preparation functions were checked on new shipments using the historical medians. All 48 rows were preserved, with no missing model-feature values.

### Service type

Express had the highest delay rate:

| Service | Promised hours | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|---:|
| Express | 36 | 243 | 119 | 48.97% |
| Standard | 60 | 642 | 132 | 20.56% |
| Economy | 96 | 315 | 27 | 8.57% |

Express has the shortest delivery promise. Its higher delay rate does not mean it has the longest delivery times; lateness is measured against each service's promise.

### Hub tier

Delay rates increased from Tier 1 to Tier 3:

| Hub tier | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| 1 | 400 | 16 | 4.00% |
| 2 | 400 | 86 | 21.50% |
| 3 | 400 | 176 | 44.00% |

Hub tier is associated with delay risk in this dataset. This does not establish causation.

### Distance

Delay rates increased across the distance bands:

| Distance band | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| 0–200 km | 274 | 6 | 2.19% |
| Above 200–400 km | 563 | 77 | 13.68% |
| Above 400–600 km | 275 | 120 | 43.64% |
| Above 600 km | 88 | 75 | 85.23% |

Longer journeys were associated with a greater likelihood of missing the delivery promise.

### Correlations

The three strongest absolute correlations with `delivery_hours`, excluding the target itself, were:

| Column | Pearson correlation |
|---|---:|
| `promised_hours` | +0.8372 |
| `distance_km` | +0.4013 |
| `n_stops` | +0.3247 |

Longer service promises, longer distances and more stops were associated with longer delivery times. Correlation does not prove causation.

### Declared value and log transformation

After filling missing values, mean declared value was INR 1,533.05 and the median was INR 1,121.50. The higher mean suggests a right-skewed distribution.

After applying the natural logarithm, the mean was 7.0277 and the median was 7.0224. These values were much closer. The transformation compresses large values and helps reduce skew. Log-transformed values are not amounts in INR.

### Delivery-time outliers

The first and third quartiles of delivery hours were 43.2550 and 72.5625. The IQR-based lower and upper bounds were approximately -0.7063 and 116.5238 hours.

One shipment fell outside these bounds. This diagnostic counted the outlier; it was retained in the modelling data.

### Part A outputs

- `outputs/part_a_checks.csv`
- `outputs/part_a_delay_by_group.csv`
- `outputs/part_a_correlations.csv`
- `data/modelling_table.csv`

## Model results (regression)

### Inputs, target and split

The target was `delivery_hours`. All models used the same 17 input features in the shared `FEATURE_COLUMNS` order.

Neither `delivery_hours` nor `is_delayed` was included in the model inputs.

The data was split using two calls to `train_test_split`, both with `random_state=42`:

| Dataset | Rows | Purpose |
|---|---:|---|
| Training | 840 | Fit the models |
| Validation | 180 | Compare models and choose settings |
| Test | 180 | Evaluate the selected model |

Linear Regression used a pipeline containing `StandardScaler` followed by the regression model. The scaler was fitted only on training rows.

The largest absolute mean among scaled training columns was approximately `4.017950e-16`, effectively zero. For scaled test columns, it was approximately `0.2186886`. Test means need not be zero because the training scaler was reused without refitting.

### Baseline

The baseline predicted the training-set mean of 57.7493 hours for every validation shipment.

Its validation R² was -0.0212. This means it performed slightly worse than predicting the validation set's own mean. The baseline provides a simple reference for assessing whether the models learn useful relationships.

### Validation comparison

| Model | MAE (hours) | MSE (hours²) | RMSE (hours) | R² |
|---|---:|---:|---:|---:|
| Mean baseline | 15.7952 | 355.5204 | 18.8552 | -0.0212 |
| Linear Regression | 3.9222 | 23.9235 | 4.8912 | 0.9313 |
| Decision Tree | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| Random Forest | 4.2794 | 30.3153 | 5.5059 | 0.9129 |
| Gradient Boosting | 3.9166 | 24.1722 | 4.9165 | 0.9306 |

Linear Regression was selected because it had the lowest validation RMSE: 4.8912 hours.

Gradient Boosting had slightly lower MAE, but the required selection criterion was RMSE.

Random Forest used 100 trees and `random_state=42`. The Gradient Boosting candidate used 100 trees, `learning_rate=0.1` and `random_state=42`.

### Decision Tree depth and overfitting

Each tree used `random_state=42`.

| Maximum depth | Training RMSE | Validation RMSE |
|---|---:|---:|
| 1 | 11.6334 | 12.1324 |
| 2 | 9.3687 | 10.4114 |
| 3 | 7.5295 | 8.7033 |
| 4 | 6.0999 | 7.3896 |
| 6 | 4.7114 | 6.2796 |
| 8 | 3.3360 | 6.7901 |
| 12 | 1.3613 | 7.3540 |
| Unlimited | 0.0000 | 7.8368 |

Depths 1–2 showed high bias: both training and validation errors were relatively high.

Depth 6 achieved the lowest validation RMSE and was selected for the Decision Tree candidate.

Depths 8, 12 and unlimited showed increasing overfitting. Training error decreased while validation error increased. The unlimited tree achieved zero training error but had validation RMSE of 7.8368 hours.

### Gradient Boosting learning-rate experiment

All runs used 100 trees and `random_state=42`.

| Learning rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

Learning rate 0.1 performed best.

At 0.01, updates were too small to learn sufficiently within 100 trees. At 1.0, updates were aggressive and validation performance worsened, consistent with overshooting or fitting noise.

### Cross-validation and final selection

The two candidates with the lowest validation RMSE were evaluated on the combined 1,020 training-plus-validation rows.

Both used the same `KFold(n_splits=5, shuffle=True, random_state=42)` splits. The scoring metric was R². Linear Regression's scaler was fitted separately within each fold through its pipeline.

| Model | Mean R² | Standard deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9353 | 0.0045 |

Cross-validation favored Gradient Boosting, while the original validation split favored Linear Regression. Both had consistent scores across folds.

For this submission, I would ship Linear Regression because it won under the prescribed validation-RMSE rule and provides a simpler model. However, Gradient Boosting's stronger cross-validation result makes it a credible alternative. I would compare both on additional representative data before a real operational deployment.

### Final test evaluation

The selected Linear Regression model was evaluated on the 180 test shipments using its original fit on the 840 training rows. It was not refitted on the combined training and validation data for this evaluation.

| Metric | Test result |
|---|---:|
| MAE | 3.5671 hours |
| MSE | 20.4624 hours² |
| RMSE | 4.5235 hours |
| R² | 0.9493 |

The average absolute prediction error was approximately 3.57 hours. Test results were not used to change the model or its settings.

### Part B outputs

- `outputs/part_b_checks.csv`
- `outputs/part_b_tree_depth.csv`
- `outputs/part_b_metrics.csv`
- `outputs/part_b_gb_learning_rate.csv`
- `outputs/part_b_cv.csv`

## Model results (classification)

### Target and data split

The target is `is_delayed`: 1 means delayed and 0 means on time. Initial models used the same 17 input features.

Both splits used `random_state=42` and stratification to preserve approximately the same proportion of delayed parcels.

| Split | Shipments | Delayed | On time |
|---|---:|---:|---:|
| Training | 840 | 195 | 645 |
| Validation | 180 | 42 | 138 |
| Test | 180 | 41 | 139 |

Models and feature selectors were fitted using training data. Validation data was used for configuration and threshold selection.

### Baseline and initial model comparison

The blind baseline predicted every parcel as on time. It achieved 76.67% accuracy but missed all 42 delayed validation parcels, giving zero recall and F1.

This shows why accuracy alone is insufficient for identifying delays.

The initial trained models used all 17 features and their default class prediction rules.

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Blind baseline | 138 | 0 | 42 | 0 | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree, depth 6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Default Random Forest | 132 | 6 | 10 | 32 | 0.9111 | 0.8421 | 0.7619 | 0.8000 |

Logistic Regression used a StandardScaler pipeline. Decision Tree depth 6 achieved the highest validation F1 among depths 2, 3, 4, 5, 6, 8 and 10.

Logistic Regression detected more delayed parcels than the initial forest, while the forest produced fewer false alerts. Subsequent feature-selection and tuning experiments followed the project's required Random Forest workflow.

### Feature scores and noise columns

The seven highest-ranked features by training-data F-score were:

| Rank | Feature | F-score |
|---|---|---:|
| 1 | `distance_km` | 389.2131 |
| 2 | `hub_tier` | 155.3483 |
| 3 | `hub_daily_capacity` | 144.1347 |
| 4 | `n_stops` | 125.4798 |
| 5 | `promised_hours` | 84.4177 |
| 6 | `rain_flag` | 53.0321 |
| 7 | `sale_event_flag` | 36.7281 |

All seven had F-scores above 30. All five bookkeeping/noise columns ranked outside the top seven and had F-scores below 10.

The noise columns together received 22.52% of the forest's impurity-based importance. Nonzero tree importance does not establish useful predictive signal on unseen data.

Feature-selection F-scores measure individual feature association with the target; they are different from the F1 metric used to evaluate classifier predictions.

### Number of selected features

SelectKBest with `f_classif` was fitted on training data inside a pipeline with a 100-tree Random Forest.

| Selected features (k) | Validation F1 |
|---|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.8140 |
| 12 | 0.8049 |
| 17 | 0.8000 |

Seven features produced the highest validation F1. Five features omitted useful information, while adding features beyond seven did not improve this validation result.

All subsequent experiments used `k=7`, as required by the project.

### Random Forest tuning

The searches tested these parameter choices:

- `n_estimators`: 50, 100, 150
- `max_depth`: 4, 6, 8
- `min_samples_leaf`: 1, 5, 10

Following the prescribed procedure, the seven-feature selector was fitted on all 840 training rows before the searches, and the transformed training data was passed to cross-validation.

Both searches used five-fold cross-validation with default accuracy scoring. Randomized Search used `n_iter=9` and `random_state=42`.

| Configuration | Candidates | CV folds | Total fits | Best CV accuracy | Validation F1 |
|---|---:|---:|---:|---:|---:|
| Grid Search | 27 | 5 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 5 | 45 | 0.8905 | 0.7901 |
| Default forest, same seven features | 1 | 0 | 1 | Not calculated | 0.8333 |

Search fit counts follow the project convention and exclude the additional refit of the selected configuration. The default forest required one direct training fit.

Grid Search selected 100 trees, depth 6 and minimum leaf size 1. Randomized Search selected 50 trees, depth 6 and minimum leaf size 1.

Neither tuned candidate beat the default seven-feature forest on validation F1. Tuning optimized CV accuracy, whereas final selection used validation F1. The validation set contained only 180 shipments, so small differences should not be treated as a universal ranking.

The selected configuration was:

- `SelectKBest(f_classif, k=7)`
- `RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42)`

The complete final pipeline was fitted on the original 840 training rows.

### Alert-threshold selection

A parcel is flagged as delayed when its predicted delay probability is greater than or equal to the threshold.

Missing a delayed parcel is more costly than an unnecessary call. The required rule therefore selects the highest threshold with validation recall of at least 0.80.

Thresholds from 0.1 through 0.9 were evaluated.

| Threshold | False alerts (FP) | Missed delays (FN) | Precision | Recall |
|---|---:|---:|---:|---:|
| 0.1 | 44 | 1 | 0.4824 | 0.9762 |
| 0.2 | 29 | 3 | 0.5735 | 0.9286 |
| 0.3 | 17 | 3 | 0.6964 | 0.9286 |
| 0.4 | 13 | 5 | 0.7400 | 0.8810 |
| 0.5 | 9 | 7 | 0.7955 | 0.8333 |
| 0.6 | 4 | 10 | 0.8889 | 0.7619 |
| 0.7 | 4 | 15 | 0.8710 | 0.6429 |
| 0.8 | 4 | 23 | 0.8261 | 0.4524 |
| 0.9 | 2 | 30 | 0.8571 | 0.2857 |

The selected threshold was 0.5. It achieved 83.33% validation recall and 79.55% precision. At 0.6, recall fell below the required 80%.

Lower thresholds caught more delays but generated more false alerts.

At threshold 0.5, validation F1 was 0.8140. This differs from the earlier seven-feature default-prediction F1 of 0.8333 because the explicit `>= 0.5` rule flags probability ties as delayed, while the forest's default `predict()` resolves those ties to class 0.

### Final test results

The selected pipeline and threshold were evaluated on the held-out test set without further tuning.

| Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 125 | 14 | 10 | 31 | 0.8667 | 0.6889 | 0.7561 | 0.7209 |

The model detected 31 of the 41 delayed test parcels, missed 10 and generated 14 false alerts.

Recall decreased from 83.33% on validation to 75.61% on test. Precision decreased from 79.55% to 68.89%.

The 80% recall rule applied to validation-based threshold selection; it does not guarantee 80% recall on unseen data. With only 41 delayed test parcels, each additional missed parcel changes test recall by approximately 2.44 percentage points.

The test results were reported without changing the selected configuration or threshold.

### Part C outputs

- `outputs/part_c_models.csv`
- `outputs/part_c_feature_scores.csv`
- `outputs/part_c_selectk.csv`
- `outputs/part_c_tuning.csv`
- `outputs/part_c_thresholds.csv`
- `outputs/part_c_final_test.csv`
- `outputs/part_c_final_config.csv`

## Operations playbook

Pending Part D.

This section will describe hub segments and how predicted delay risk combines with hub playbooks to produce an action list.

## Production and reproducibility

The completed scripts read repository files and calculate their results in code. Original generated input files are preserved; cleaning operates on copies.

Shared preparation functions provide consistent transformations for historical and new shipments. New shipments use the historical imputation medians.

Library versions are pinned in `requirements.txt`. Fixed random states make splits and model experiments repeatable.

Part C saves the selected classifier configuration and alert threshold in `outputs/part_c_final_config.csv` for use during model packaging.

Model packaging, saved preprocessing information, reload verification and reproducible batch scoring are pending Part D.

## Limitations

- The dataset is synthetic. Results do not establish performance on real Delhivery operations.
- Validation and test sets contain only 180 shipments each, so rankings and error estimates may vary with the sample.
- Linear Regression and Gradient Boosting were close on validation, and their cross-validation ranking differed.
- Correlations and grouped delay rates describe associations, not causal effects.
- The regression model produces point estimates, not prediction intervals.
- Classification test recall was 75.61%, below validation recall of 83.33%; the alert rule still misses some delayed parcels.
- Following the prescribed tuning procedure, feature selection occurred before the inner cross-validation splits. Those CV scores therefore do not independently evaluate the feature-selection step. Validation and test rows were excluded from fitting the selector.
- Hub segmentation, model packaging and final operational scoring are not yet completed.