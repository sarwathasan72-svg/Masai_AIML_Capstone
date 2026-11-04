# Masai_AIML_Capstone

Masai Capstone — Delhivery Last-Mile Delay Intelligence Pipeline

## Project overview

This project uses synthetic shipment data to answer three operational questions:

1. How many hours will a parcel take to arrive?
2. Which parcels are likely to miss their promised delivery time?
3. Which hubs need different operational playbooks?

No real Delhivery shipment records are used.

### Implementation status

- Part A: Data cleaning, feature preparation and exploratory analysis completed.
- Part B: Delivery-hours regression and evaluation completed.
- Part C: Delay classification, feature selection, tuning, threshold selection and final test evaluation completed.
- Part D: Hub segmentation, model packaging, reload verification and new-shipment scoring completed.
- Repeatability: Two consecutive scoring runs produced byte-identical output.
- Final submission checks: Clean-clone execution, repository contents and GitHub synchronization still need final verification.

## How to run

Run commands from the repository root.

### Clone and set up the environment

The following commands are for macOS/Linux:

```bash
git clone https://github.com/sarwathasan72-svg/Masai_AIML_Capstone.git
cd Masai_AIML_Capstone
git switch main

python3 -m venv .venv
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

The three generated input CSVs are included under `data/`. To regenerate them using the provided generator:

```bash
python generate_data.py
```

Run the pipeline in this order:

```bash
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`package_model.py` regenerates `models/delay_model.pkl` using the configuration selected in Part C.

The committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`. When installed versions differ, the grader re-runs `package_model.py` instead of loading the committed pickle.

The final batch action list is written to:

```text
outputs/new_shipment_actions.csv
```

### Main files

| File | Purpose |
|---|---|
| `generate_data.py` | Provided synthetic-data generator |
| `data/hubs.csv` | Reference information about 24 hubs |
| `data/raw_shipments.csv` | Historical shipments before cleaning |
| `data/new_shipments.csv` | 48 new shipments to score |
| `src/shipment_prep.py` | Shared constants, cleaning and feature preparation |
| `part_a_clean_eda.py` | Creates the modelling table and Part A analysis outputs |
| `data/clean_shipments.csv` | Intermediate cleaned historical data |
| `data/modelling_table.csv` | Prepared historical data used by Parts B, C and D |
| `part_b_regression.py` | Compares delivery-hours regression models |
| `part_c_classification.py` | Compares classifiers, selects features, tunes the forest and selects an alert threshold |
| `part_d_segments.py` | Creates hub profiles, segments and tier-mismatch reports |
| `package_model.py` | Rebuilds and saves the selected classifier and verifies reload consistency |
| `predict_delay.py` | Scores new parcels and assigns operational actions |
| `inspect_data.py` | Optional helper for inspecting data |
| `requirements.txt` | Exact library versions |
| `models/` | Saved classifier, threshold, feature order and imputation medians |
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

Cleaning standardized service names from 12 spellings to three categories, filled missing payment modes with `Unknown`, converted declared values to numeric INR amounts and parsed dispatch timestamps.

Missing numeric values were filled with historical medians:

| Column | Median |
|---|---:|
| `distance_km` | 314.95 km |
| `parcel_weight_kg` | 1.83 kg |
| `declared_value_inr` | INR 1,121.50 |

Feature preparation joined hub information and created `promised_hours`, `is_cod`, `is_weekend`, `dispatch_hour` and `log_declared_value`.

The modelling table retained all 1,200 shipments and had no missing values in the required features and targets. There were 278 delayed shipments, representing 23.17% of the total.

A shipment was labelled delayed when actual delivery hours exceeded promised hours.

The same preparation functions were checked on new shipments using the historical medians. All 48 rows were preserved, with no missing model-feature values.

Historical checks are recorded in `outputs/part_a_checks.csv`.

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

| Distance-band label | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| 0-200 | 274 | 6 | 2.19% |
| 201-400 | 563 | 77 | 13.68% |
| 401-600 | 275 | 120 | 43.64% |
| 601+ | 88 | 75 | 85.23% |

The bin edges are 0, 200, 400, 600 and 2000, with right edges included. The labels follow the assignment; for example, `201-400` represents distances greater than 200 and up to 400 km.

Longer journeys were associated with a greater likelihood of missing the delivery promise.

Service, tier and distance results are recorded in `outputs/part_a_delay_by_group.csv`.

### Correlations

The three strongest absolute correlations with `delivery_hours`, excluding the target itself, were:

| Column | Pearson correlation |
|---|---:|
| `promised_hours` | +0.8372 |
| `distance_km` | +0.4013 |
| `n_stops` | +0.3247 |

Longer service promises, longer distances and more stops were associated with longer delivery times. Correlation does not prove causation.

Source: `outputs/part_a_correlations.csv`.

### Declared value and log transformation

After filling missing values, mean declared value was INR 1,533.05 and the median was INR 1,121.50. The higher mean suggests a right-skewed distribution.

After applying the natural logarithm, the mean was 7.0277 and the median was 7.0224. The transformation compresses large values and helps reduce skew. Log-transformed values are not amounts in INR.

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

The data was split using two calls to `train_test_split`, both with `random_state=42`: first a 70%/30% split, then an equal split of the temporary 30%.

| Dataset | Rows | Purpose |
|---|---:|---|
| Training | 840 | Fit models |
| Validation | 180 | Compare models and choose settings |
| Test | 180 | Evaluate the selected model |

Linear Regression used a pipeline containing `StandardScaler` followed by the regression model. The scaler was fitted only on training rows.

The largest absolute mean among scaled training columns was approximately `4.017950e-16`, effectively zero. For scaled test columns, it was approximately `0.2186886`. Test means need not be zero because the training scaler was reused without refitting.

Source: `outputs/part_b_checks.csv`.

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

Random Forest used 100 trees and `random_state=42`. Gradient Boosting used 100 trees, `learning_rate=0.1` and `random_state=42`.

Source: `outputs/part_b_metrics.csv`.

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

Depths 8, 12 and unlimited showed increasing overfitting. Training error decreased while validation error increased. The unlimited tree achieved zero training error but validation RMSE of 7.8368 hours.

Source: `outputs/part_b_tree_depth.csv`.

### Gradient Boosting learning-rate experiment

All runs used 100 trees and `random_state=42`.

| Learning rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

Learning rate 0.1 performed best.

At 0.01, updates were too small to learn sufficiently within 100 trees. At 1.0, updates were aggressive and validation performance worsened, consistent with overshooting or fitting noise.

Source: `outputs/part_b_gb_learning_rate.csv`.

### Cross-validation and final selection

The two candidates with the lowest validation RMSE were evaluated on the combined 1,020 training-plus-validation rows.

Both used the same `KFold(n_splits=5, shuffle=True, random_state=42)` splits. The scoring metric was R². Linear Regression's scaler was fitted separately within each fold through its pipeline.

| Model | Mean R² | Standard deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9353 | 0.0045 |

Cross-validation favored Gradient Boosting, while the original validation split favored Linear Regression. Both had consistent scores across folds.

For this submission, I would ship Linear Regression because it won under the prescribed validation-RMSE rule and provides a simpler model. Gradient Boosting's stronger cross-validation result makes it a credible alternative to assess on additional representative data before real deployment.

Source: `outputs/part_b_cv.csv`.

### Final test evaluation

The selected Linear Regression model was evaluated on the 180 test shipments using its original fit on the 840 training rows. It was not refitted on combined training and validation data for this evaluation.

| Metric | Test result |
|---|---:|
| MAE | 3.5671 hours |
| MSE | 20.4624 hours² |
| RMSE | 4.5235 hours |
| R² | 0.9493 |

The average absolute prediction error was approximately 3.57 hours. Test results were not used to change the model or its settings.

Source: the test row in `outputs/part_b_metrics.csv`.

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

Initial trained models used all 17 features and their default class prediction rules.

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Blind baseline | 138 | 0 | 42 | 0 | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree, depth 6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Default Random Forest | 132 | 6 | 10 | 32 | 0.9111 | 0.8421 | 0.7619 | 0.8000 |

Logistic Regression used a StandardScaler pipeline. Decision Tree depth 6 achieved the highest validation F1 among depths 2, 3, 4, 5, 6, 8 and 10.

Logistic Regression detected more delayed parcels than the initial forest, while the forest produced fewer false alerts. Subsequent experiments followed the required Random Forest feature-selection and tuning workflow.

Source: `outputs/part_c_models.csv`.

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

Source: `outputs/part_c_feature_scores.csv`.

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

All subsequent experiments used `k=7`, as required.

Source: `outputs/part_c_selectk.csv`.

### Random Forest tuning

The searches tested:

- `n_estimators`: 50, 100, 150
- `max_depth`: 4, 6, 8
- `min_samples_leaf`: 1, 5, 10

Following the prescribed procedure, the seven-feature selector was fitted on all 840 training rows before the searches.

Both searches used five-fold cross-validation with default accuracy scoring. Randomized Search used `n_iter=9` and `random_state=42`.

| Configuration | Candidates | CV folds | Total fits | Best CV accuracy | Validation F1 |
|---|---:|---:|---:|---:|---:|
| Grid Search | 27 | 5 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 5 | 45 | 0.8905 | 0.7901 |
| Default forest, same seven features | 1 | 0 | 1 | Not calculated | 0.8333 |

Search fit counts exclude the additional best-configuration refit. The default forest required one direct training fit.

Grid Search selected 100 trees, depth 6 and minimum leaf size 1. Randomized Search selected 50 trees, depth 6 and minimum leaf size 1.

Neither tuned candidate beat the default seven-feature forest on validation F1. The searches optimized CV accuracy, whereas final selection used validation F1. Their restricted search space also did not include unlimited tree depth.

The selected configuration was:

```python
Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
    )),
])
```

The final pipeline was fitted on the original 840 training rows.

Sources: `outputs/part_c_tuning.csv` and `outputs/part_c_final_config.csv`.

### Alert-threshold selection

A parcel is flagged when its predicted delay probability is greater than or equal to the threshold.

Because a missed delayed parcel costs more than an unnecessary call, the required rule selects the highest tested threshold with validation recall of at least 0.80.

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

The selected threshold was 0.5, with 83.33% validation recall and 79.55% precision. At 0.6, recall fell below 80%.

Lower thresholds caught more delays but generated more false alerts.

At threshold 0.5, validation F1 was 0.8140. This differs from the earlier seven-feature default-prediction F1 of 0.8333 because the explicit `>= 0.5` rule flags probability ties as delayed, while the forest's default `predict()` resolves those ties to class 0.

Source: `outputs/part_c_thresholds.csv`.

### Final test results

The selected pipeline and threshold were evaluated on the held-out test set without further tuning.

| Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 125 | 14 | 10 | 31 | 0.8667 | 0.6889 | 0.7561 | 0.7209 |

The model detected 31 of 41 delayed test parcels, missed 10 and generated 14 false alerts.

Recall decreased from 83.33% on validation to 75.61% on test. Precision decreased from 79.55% to 68.89%.

The 80% recall rule applied to validation-based threshold selection; it does not guarantee 80% recall on unseen data. With only 41 delayed test parcels, each additional missed parcel changes recall by approximately 2.44 percentage points.

Source: `outputs/part_c_final_test.csv`.

### Part C outputs

- `outputs/part_c_models.csv`
- `outputs/part_c_feature_scores.csv`
- `outputs/part_c_selectk.csv`
- `outputs/part_c_tuning.csv`
- `outputs/part_c_thresholds.csv`
- `outputs/part_c_final_test.csv`
- `outputs/part_c_final_config.csv`

## Hub segmentation and model packaging

### Hub profiles and clustering

I summarized the historical shipments into 24 hub profiles, with 50 shipments per hub.

Clustering used four hub-level measurements:

- Mean delivery hours.
- Delay rate.
- Mean distance.
- Mean number of stops.

The profiles were sorted by ascending `hub_id`, scaled using StandardScaler and clustered with:

```python
KMeans(n_clusters=3, random_state=42, n_init=10)
```

The existing hub tier was not a clustering input. This allowed observed behaviour to be compared with the tier labels afterwards.

Clusters were named by increasing average hub delay rate, rather than their arbitrary cluster numbers.

| Segment | Hubs | Mean delivery hours | Delay rate | Mean distance (km) | Mean stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.2396 | 0.0257 | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 0.1550 | 314.1762 | 70.7175 |
| Escalate | 9 | 64.6504 | 0.4600 | 470.4757 | 84.3444 |

Sources: `outputs/hub_profile.csv` and `outputs/segment_profile.csv`.

### Cluster separation

The separation ratio is the mean Euclidean distance between hubs in different clusters divided by the mean distance between hubs in the same cluster.

Distances were measured on the scaled profiles using all 276 unordered hub pairs.

| K | Across-to-within distance ratio | Cluster sizes |
|---|---:|---|
| 2 | 2.8940 | 15-9 |
| 3 | 5.3928 | 9-8-7 |
| 4 | 6.3047 | 8-7-5-4 |
| 5 | 6.8866 | 8-7-4-3-2 |

Source: `outputs/part_d_separation.csv`.

### Model packaging and reload verification

`package_model.py` reads the saved Part C configuration, recreates the same stratified split and fits the selected pipeline on the 840 training rows.

It writes:

| Artifact | Contents |
|---|---|
| `models/delay_model.pkl` | Fitted SelectKBest and Random Forest pipeline |
| `models/delay_threshold.csv` | Selected threshold |
| `models/feature_order.csv` | All 17 input names in their required order |
| `models/impute_values.csv` | Three historical cleaning medians |

The pipeline accepts all 17 inputs and performs seven-feature selection internally.

After reloading the saved pipeline, probabilities were compared against the in-memory pipeline on 180 test rows.

| Check | Result |
|---|---:|
| Test rows scored | 180 |
| Maximum absolute probability difference | 0.0 |
| Threshold | 0.5 |

This verifies save/load consistency, not additional model accuracy.

Source: `outputs/part_d_reload_check.csv`.

### Batch-scoring repeatability

`predict_delay.py` reuses the shared cleaning and feature-building functions, stored medians, saved feature order, classifier and threshold.

The output preserves input-file order. Flags are calculated using unrounded probabilities; reported probabilities are rounded to four decimals.

Two consecutive runs produced byte-identical `outputs/new_shipment_actions.csv` files.

The observed SHA-256 checksum was:

```text
a67611c016f731b930962d383a8ab77bc954ed01746646c01ab2833e042a349b
```

This confirms repeatability for the tested inputs and environment.

### Part D outputs

- `outputs/hub_profile.csv`
- `outputs/part_d_separation.csv`
- `outputs/hub_segments.csv`
- `outputs/segment_profile.csv`
- `outputs/tier_vs_segment_mismatch.csv`
- `outputs/part_d_reload_check.csv`
- `outputs/new_shipment_actions.csv`
- `models/delay_model.pkl`
- `models/delay_threshold.csv`
- `models/feature_order.csv`
- `models/impute_values.csv`

## Operations recommendation

### Three operating playbooks

- **Standard:** Continue routine monitoring and investigate unusual changes.
- **Watch:** Review route workload, distance and service mix regularly.
- **Escalate:** Prioritize hub-manager review of flagged parcels and investigate recurring capacity or route constraints.

These are proposed responses based on synthetic historical data, not proven interventions or measured cost savings.

I retained K=3 because operations requires three playbooks. Although the separation ratio increases for K=4 and K=5, both create clusters with fewer than six hubs, too small to support their own playbook under the project rule.

Source: `outputs/part_d_separation.csv`.

### Escalate hubs

| Hub | Historical delay rate |
|---|---:|
| HUB13 | 46% |
| HUB15 | 36% |
| HUB17 | 46% |
| HUB18 | 38% |
| HUB19 | 62% |
| HUB20 | 48% |
| HUB21 | 40% |
| HUB23 | 44% |
| HUB24 | 54% |

Sources: `outputs/hub_profile.csv` and `outputs/hub_segments.csv`.

### Existing tiers versus observed behaviour

| Hub | City | Existing tier | Observed segment | Recommendation |
|---|---|---:|---|---|
| HUB07 | Kolkata | 1 | Watch | Apply Watch monitoring despite its Tier 1 label. |
| HUB13 | Nagpur | 2 | Escalate | Prioritize flagged parcels for hub-manager review. |
| HUB15 | Kochi | 2 | Escalate | Prioritize flagged parcels for hub-manager review. |
| HUB22 | Dehradun | 3 | Watch | Use observed Watch behaviour alongside its Tier 3 label. |

Source: `outputs/tier_vs_segment_mismatch.csv`.

These findings do not overwrite the tier labels. They show why tier alone should not determine the operating response.

### Parcel alert policy

The selected alert threshold is **0.50**.

On the final test set:

- Precision was **0.6889**: about **6.9 of every 10 flagged parcels** were actually delayed.
- Recall was **0.7561**: about **7.6 of every 10 delayed parcels** were detected.
- There were **31 detected delays, 14 false alerts and 10 missed delays**.

Source: `outputs/part_c_final_test.csv`.

The threshold met the validation recall requirement but did not achieve 80% recall on the held-out test set. This limitation should be considered when planning operational reliance.

### Actions for the 48 new parcels

| Action | Parcels |
|---|---:|
| No action | 40 |
| Proactive customer call | 2 |
| Escalate to hub manager | 6 |
| Total flagged | 8 |

Source: `outputs/new_shipment_actions.csv`.

Actions follow these rules:

1. If a parcel is not flagged, assign `No action`.
2. If it is flagged and its hub segment is Escalate, assign `Escalate to hub manager`.
3. Otherwise, if it is flagged, assign `Proactive customer call`.

These are recommendations in the CSV, not calls or escalations already executed.

Actual outcomes for the new batch are unavailable, so its prediction accuracy cannot yet be measured.

## Production plan

### Readiness characteristics

| Characteristic | Concrete repository evidence |
|---|---|
| Accuracy | Held-out regression and classification metrics in `part_b_metrics.csv` and `part_c_final_test.csv`. |
| Reliability | Input/output assertions and reload verification in `package_model.py` and `predict_delay.py`. |
| Reproducibility | Fixed random seeds, pinned dependencies, documented execution order and repeatability check. |
| Maintainability | Shared functions and constants in `src/shipment_prep.py`. |
| Consistency | Training preparation and new-shipment scoring reuse the same functions, historical medians and feature order. |
| Monitorability | The action CSV records shipment ID, hub, segment, probability, flag and action. |

These artifacts support production readiness but do not constitute a deployed monitoring service. Runtime performance and operational reliability still require measurement on representative live workloads.

### Proposed monitoring triggers

The following are starting policies, not measured production results. They should be reviewed after a representative pilot.

| Quantity | Proposed numeric trigger | Response |
|---|---|---|
| Actual delayed share | More than 5 percentage points above the comparable historical baseline for two consecutive completed weekly cohorts. | Investigate hub, route and service-mix changes. |
| Share of parcels flagged | More than 5 percentage points above or below the comparable pilot baseline for two consecutive weeks. | Check data quality, input drift and alert workload. |
| Recall after outcomes become known | Below 0.70 over the latest 100 actually delayed parcels. | Review missed cases and evaluate a challenger model and threshold. |
| Missing rate in a required numeric input | Above 5% in a scoring batch, measured before filling. | Investigate the upstream feed and imputation impact. |
| Unknown hub IDs | Any unmatched hub ID in a batch. | Resolve reference-data coverage before issuing actions for affected rows. |

The pilot baseline should be established on representative operational batches, rather than treating the small synthetic new-shipment batch as a permanent baseline.

Outcome-based monitoring must wait for sufficiently mature delivery outcomes. Undelivered parcels should not automatically be treated as on time.

### Retraining and drift

I propose a monthly model review and retraining evaluation, with an earlier review if a monitoring trigger is breached.

A candidate model should replace the existing model only after validation against the agreed metrics and operational alert workload. Retraining should not automatically imply deployment.

For the 15–28 September sale window, monitor `sale_event_flag` and compare sale-event performance with comparable sale-event data, rather than ordinary weeks. If comparable event history is unavailable, record that limitation.

Data drift could appear as changes in distance, route stops, service mix, hub capacity, rain frequency or missing-value rates. Changes in how those inputs relate to actual delays could also reduce model performance.

Review hub profiles and segment assignments as operational conditions change.

Any replacement classifier should be packaged with its matching threshold, feature order and imputation values.

## Limitations

- The dataset is synthetic. Results do not establish performance on real Delhivery operations.
- Validation and test sets contain only 180 shipments each, so rankings and error estimates may vary with the sample.
- The classification test set contains only 41 delayed parcels. Recall is sensitive to a small number of additional missed cases.
- The threshold was selected on one validation split. Test recall was 75.61%, below validation recall of 83.33%.
- Model probability scores have not been independently calibrated as real-world probabilities.
- Linear Regression and Gradient Boosting were close on validation, and their cross-validation ranking differed.
- The regression model produces point estimates, not prediction intervals or guaranteed delivery-time bounds.
- Correlations, grouped delay rates and cluster membership describe associations, not causal effects.
- Hub profiles use only 50 historical shipments per hub and may change with season, service mix or network conditions.
- Following the brief, historical medians were computed before splitting. A stricter future evaluation would fit imputation within training data or each training fold.
- Following the prescribed tuning procedure, feature selection occurred before internal cross-validation. Those CV scores do not independently evaluate feature selection; outer validation and test rows were excluded from fitting the selector.
- New-shipment outcomes are unavailable, so the accuracy of the 48-parcel action list is unknown.
- Monitoring triggers and operational playbooks are proposals. Their costs and benefits have not been measured in production.