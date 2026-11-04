import pandas as pd
from sklearn.model_selection import train_test_split

from src.shipment_prep import FEATURE_COLUMNS

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import NOISE_COLUMNS

import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Load the prepared historical shipments.
modelling_df = pd.read_csv("data/modelling_table.csv")

# Inputs: the same 17 features used in Part B.
X = modelling_df[FEATURE_COLUMNS].copy()

# Target: 0 = on time, 1 = delayed.
y = modelling_df["is_delayed"].copy()

# Split 70% training and 30% temporary data.
# Stratify preserves approximately the same class proportions.
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split the remaining data equally into validation and test.
X_val, X_test, y_val, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=42,
    stratify=y_remaining
)

# Summarize each dataset.
split_summary = pd.DataFrame([
    {
        "split": "train",
        "rows": len(y_train),
        "delayed": int(y_train.sum()),
        "on_time": int((y_train == 0).sum())
    },
    {
        "split": "validation",
        "rows": len(y_val),
        "delayed": int(y_val.sum()),
        "on_time": int((y_val == 0).sum())
    },
    {
        "split": "test",
        "rows": len(y_test),
        "delayed": int(y_test.sum()),
        "on_time": int((y_test == 0).sum())
    }
])

print("Number of input features:", X.shape[1])
print("Target:", y.name)
print("\nClassification split:")
print(split_summary.to_string(index=False))


# Reusable evaluation function for all classification models.
# Positive class (1) means delayed.
def classification_metrics(model_name, actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    return {
        "model": model_name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        )
    }


# Predict 0 (on time) for every validation shipment.
blind_predictions = np.zeros(len(y_val), dtype=int)

classification_results = [
    classification_metrics(
        "blind_all_on_time",
        y_val,
        blind_predictions
    )
]

print("\nBlind baseline validation results:")
print(
    pd.DataFrame(classification_results)
    .round(4)
    .to_string(index=False)
)


# Scale features and train Logistic Regression.
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Learn using training data only.
logistic_model.fit(X_train, y_train)

# Predict classes for the validation shipments.
logistic_predictions = logistic_model.predict(X_val)

# Evaluate using the function we created earlier.
logistic_result = classification_metrics(
    "logistic_regression",
    y_val,
    logistic_predictions
)

classification_results.append(logistic_result)

print("\nClassification validation comparison:")
print(
    pd.DataFrame(classification_results)
    .round(4)
    .to_string(index=False)
)


# compare Decision Tree classifiers at the required depths, choosing the highest validation F1.
# In Part B we selected using RMSE; here we use F1 because we are classifying delays.

# Try the depths required for classification.
classification_depths = [2, 3, 4, 5, 6, 8, 10]

tree_depth_results = []
best_classifier_tree = None
best_classifier_depth = None
best_tree_f1 = -1.0

for depth in classification_depths:
    tree_classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree_classifier.fit(X_train, y_train)
    tree_predictions = tree_classifier.predict(X_val)

    val_f1 = f1_score(
        y_val,
        tree_predictions,
        zero_division=0
    )

    tree_depth_results.append({
        "max_depth": depth,
        "val_f1": val_f1
    })

    # Ascending depths and strict > keep the smaller
    # depth if two F1 scores tie.
    if val_f1 > best_tree_f1:
        best_tree_f1 = val_f1
        best_classifier_depth = depth
        best_classifier_tree = tree_classifier

# Add the selected tree to our model comparison.
best_tree_predictions = best_classifier_tree.predict(X_val)

classification_results.append(
    classification_metrics(
        f"decision_tree_depth_{best_classifier_depth}",
        y_val,
        best_tree_predictions
    )
)

print("\nDecision Tree classification depth comparison:")
print(
    pd.DataFrame(tree_depth_results)
    .round(4)
    .to_string(index=False)
)

print("\nSelected tree depth:", best_classifier_depth)

print("\nClassification validation comparison:")
print(
    pd.DataFrame(classification_results)
    .round(4)
    .to_string(index=False)
)


# add Random Forest and save the four-model comparison.
# Train the default forest on all 17 features.
forest_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_classifier.fit(X_train, y_train)

# Predict validation classes using the default decision rule.
forest_predictions = forest_classifier.predict(X_val)

classification_results.append(
    classification_metrics(
        "random_forest_default",
        y_val,
        forest_predictions
    )
)

# Save the baseline and three trained classifiers.
classification_models_df = pd.DataFrame(classification_results)

Path("outputs").mkdir(exist_ok=True)

classification_models_df.to_csv(
    "outputs/part_c_models.csv",
    index=False,
    float_format="%.4f"
)

print("\nClassification validation comparison:")
print(
    classification_models_df.round(4).to_string(index=False)
)
print("\nSaved: outputs/part_c_models.csv")



# Calculate statistical scores for all 17 features.
feature_selector = SelectKBest(
    score_func=f_classif,
    k="all"
)
feature_selector.fit(X_train, y_train)


# Next: identify which features carry useful information about delays. The project asks for two scores:
# - F-score: checks how strongly each feature separates delayed and on-time shipments. This is different from the model’s F1 score.
# - Random Forest importance: measures how much each feature contributed to reducing impurity in the trained trees.

# Reuse the forest already trained on all 17 features.
feature_scores_df = pd.DataFrame({
    "feature": X_train.columns,
    "f_score": feature_selector.scores_,
    "rf_importance": forest_classifier.feature_importances_
})

# Rank features: 1 = highest F-score.
feature_scores_df["f_rank"] = (
    feature_scores_df["f_score"]
    .rank(ascending=False, method="first")
    .astype(int)
)

# Identify the five supplied bookkeeping/noise columns.
feature_scores_df["is_noise"] = (
    feature_scores_df["feature"].isin(NOISE_COLUMNS)
)

# Arrange the required columns and display by rank.
feature_scores_df = feature_scores_df[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
].sort_values("f_rank")

feature_scores_df.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False,
    float_format="%.4f"
)

# Calculate the noise columns' combined forest importance.
noise_importance = feature_scores_df.loc[
    feature_scores_df["is_noise"],
    "rf_importance"
].sum()

print("\nFeature scores:")
print(feature_scores_df.round(4).to_string(index=False))

print(
    f"\nNoise columns' combined forest importance: "
    f"{noise_importance:.2%}"
)
print("Saved: outputs/part_c_feature_scores.csv")


# test whether using fewer features improves validation F1.

# Compare forests using different numbers of selected features.
selectk_results = []

for k in [5, 7, 10, 12, 17]:
    selected_forest = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    # Both feature selection and model fitting use training data only.
    selected_forest.fit(X_train, y_train)

    selected_predictions = selected_forest.predict(X_val)

    selectk_results.append({
        "k": k,
        "val_f1": f1_score(
            y_val,
            selected_predictions,
            zero_division=0
        )
    })

selectk_df = pd.DataFrame(selectk_results)

selectk_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False,
    float_format="%.4f"
)

print("\nFeature-count comparison:")
print(selectk_df.round(4).to_string(index=False))
print("Saved: outputs/part_c_selectk.csv")


# tune Random Forest using Grid Search and Randomized Search,
# then compare them with an untuned forest using the same seven features.

# Select seven features using training data only.
tuning_selector = SelectKBest(score_func=f_classif, k=7)
X_train_7 = tuning_selector.fit_transform(X_train, y_train)
X_val_7 = tuning_selector.transform(X_val)

# Required parameter choices: 3 × 3 × 3 = 27 combinations.
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# Default scoring for RandomForestClassifier is accuracy.
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

print("\nRunning Grid Search...")
grid_search.fit(X_train_7, y_train)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

print("Running Randomized Search...")
random_search.fit(X_train_7, y_train)

# Fair baseline: an untuned forest on the same seven features.
default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
default_forest_7.fit(X_train_7, y_train)

tuning_results = []

for search_name, search in [
    ("grid_search", grid_search),
    ("randomized_search", random_search)
]:
    predictions = search.predict(X_val_7)
    candidate_count = len(search.cv_results_["params"])

    tuning_results.append({
        "search": search_name,
        "candidates": candidate_count,
        "cv_folds": 5,
        "total_fits": candidate_count * 5,
        "best_params": json.dumps(
            search.best_params_, sort_keys=True
        ),
        "best_cv_accuracy": search.best_score_,
        "val_f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

default_predictions_7 = default_forest_7.predict(X_val_7)

tuning_results.append({
    "search": "default_forest_same_7_features",
    "candidates": 1,
    "cv_folds": 0,
    "total_fits": 1,
    "best_params": "defaults",
    "best_cv_accuracy": np.nan,
    "val_f1": f1_score(
        y_val, default_predictions_7, zero_division=0
    )
})

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False,
    float_format="%.4f"
)

print("\nRandom Forest tuning comparison:")
print(tuning_df.round(4).to_string(index=False))
print("Saved: outputs/part_c_tuning.csv")


# select the winning configuration in code and choose the alert threshold.

# Select the highest validation F1.
# Resolve exact ties in the project's required order.
selection_order = [
    "default_forest_same_7_features",
    "grid_search",
    "randomized_search"
]

ranked_tuning = tuning_df.set_index("search").loc[selection_order]
chosen_from = ranked_tuning["val_f1"].idxmax()

if chosen_from == "default_forest_same_7_features":
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }
elif chosen_from == "grid_search":
    chosen_params = grid_search.best_params_.copy()
else:
    chosen_params = random_search.best_params_.copy()

# Fit a complete pipeline on the original training data.
final_classifier = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42,
        **chosen_params
    ))
])

final_classifier.fit(X_train, y_train)

# Find the probability column corresponding to delayed = 1.
positive_column = list(final_classifier.classes_).index(1)
val_probabilities = final_classifier.predict_proba(X_val)[
    :, positive_column
]

# Use explicit decimal values to avoid floating-point range issues.
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_results = []

for threshold in thresholds:
    predictions = (val_probabilities >= threshold).astype(int)

    result = classification_metrics(
        chosen_from,
        y_val,
        predictions
    )
    result.pop("model")

    threshold_results.append({
        "threshold": threshold,
        **result
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False,
    float_format="%.4f"
)

# Select using full-precision recall values.
eligible_thresholds = threshold_df.loc[
    threshold_df["recall"] >= 0.80,
    "threshold"
]

chosen_threshold = (
    float(eligible_thresholds.max())
    if not eligible_thresholds.empty
    else 0.1
)

print("\nSelected configuration:", chosen_from)
print("Selected parameters:", chosen_params)

print("\nValidation threshold comparison:")
print(threshold_df.round(4).to_string(index=False))

print("\nChosen threshold:", chosen_threshold)
print("Saved: outputs/part_c_thresholds.csv")


# evaluate the fixed model and threshold on the test set, then save the configuration for Part D.
# Evaluate the final model on the test set.
# Keep the model fitted on the original training rows.
test_probabilities = final_classifier.predict_proba(X_test)[
    :, positive_column
]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_result = classification_metrics(
    chosen_from,
    y_test,
    test_predictions
)
test_result.pop("model")

final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    **test_result
}])

final_test_df.to_csv(
    "outputs/part_c_final_test.csv",
    index=False,
    float_format="%.4f"
)

# Save the actual selected configuration for model packaging.
final_parameters = final_classifier.named_steps["model"].get_params()

final_config_df = pd.DataFrame([{
    "selector_k": final_classifier.named_steps["selector"].k,
    "n_estimators": final_parameters["n_estimators"],
    "max_depth": (
        "unlimited"
        if final_parameters["max_depth"] is None
        else final_parameters["max_depth"]
    ),
    "min_samples_leaf": final_parameters["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config_df.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)

print("\nFinal classification test results:")
print(final_test_df.round(4).to_string(index=False))

print("\nFinal classification configuration:")
print(final_config_df.to_string(index=False))

print("\nSaved: outputs/part_c_final_test.csv")
print("Saved: outputs/part_c_final_config.csv")