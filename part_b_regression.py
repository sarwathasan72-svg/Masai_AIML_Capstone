import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURE_COLUMNS


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


from sklearn.model_selection import KFold, cross_val_score


# 1. Load the prepared historical shipments.
modelling_df = pd.read_csv("data/modelling_table.csv")

# 2. Separate model inputs (X) and the answer to learn (y).
X = modelling_df[FEATURE_COLUMNS].copy()
y = modelling_df["delivery_hours"].copy()

# 3. Keep 70% for training; set aside 30%.
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# 4. Divide the remaining 30% equally:
#    15% validation and 15% testing.
X_val, X_test, y_val, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=42
)

# 5. Check the sizes.
print("Total shipments:", len(modelling_df))
print("Number of input features:", X.shape[1])
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Target:", y.name)

# Baseline: predict the training-set mean for every validation row.
training_mean_hours = y_train.mean()

baseline_predictions = np.full(
    len(y_val),
    training_mean_hours
)

baseline_mae = mean_absolute_error(y_val, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_val, baseline_predictions)
)
baseline_r2 = r2_score(y_val, baseline_predictions)

print("\nBaseline validation results:")
print(f"Training mean delivery hours: {training_mean_hours:.4f}")
print(f"MAE: {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"R2: {baseline_r2:.4f}")


# Scale inputs, then train Linear Regression.
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Learn scaling and model parameters using training data only.
linear_model.fit(X_train, y_train)

# Predict delivery hours for validation shipments.
linear_predictions = linear_model.predict(X_val)

linear_mae = mean_absolute_error(y_val, linear_predictions)
linear_rmse = np.sqrt(
    mean_squared_error(y_val, linear_predictions)
)
linear_r2 = r2_score(y_val, linear_predictions)

print("\nLinear Regression validation results:")
print(f"MAE: {linear_mae:.4f}")
print(f"RMSE: {linear_rmse:.4f}")
print(f"R2: {linear_r2:.4f}")

print(
    f"RMSE improvement over baseline: "
    f"{baseline_rmse - linear_rmse:.4f} hours"
)


# Compare Decision Trees with different depth limits.
depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []
best_tree = None
best_depth = None
best_val_rmse = float("inf")

for depth in depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    # Learn from training data only.
    tree.fit(X_train, y_train)

    # Predict for training and validation shipments.
    train_predictions = tree.predict(X_train)
    val_predictions = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_predictions)
    )
    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_predictions)
    )

    tree_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

    # Depths are ordered smallest first.
    # Using < keeps the smaller depth if errors tie.
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_depth = depth
        best_tree = tree

# Save the required comparison table.
tree_depth_df = pd.DataFrame(tree_results)

Path("outputs").mkdir(exist_ok=True)
tree_depth_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False,
    float_format="%.4f"
)

print("\nDecision Tree depth comparison:")
print(tree_depth_df.to_string(index=False))

print(
    "\nSelected depth:",
    "unlimited" if best_depth is None else best_depth
)
print(f"Selected validation RMSE: {best_val_rmse:.4f}")
print("Saved: outputs/part_b_tree_depth.csv")

# Train a Random Forest model with the best depth found above.

# Train Random Forest using the required settings.
forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
forest_model.fit(X_train, y_train)

# Train Gradient Boosting using the required settings.
boosting_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
boosting_model.fit(X_train, y_train)

# Keep all four trained candidates together.
# linear_model and best_tree were fitted earlier.
candidate_models = {
    "linear_regression": linear_model,
    "decision_tree": best_tree,
    "random_forest": forest_model,
    "gradient_boosting": boosting_model,
}

# A reusable function to calculate all required metrics.
def regression_metrics(model_name, actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "model": model_name,
        "split": "val",
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }


# Include the baseline for comparison.
metrics_rows = [
    regression_metrics(
        "mean_baseline",
        y_val,
        baseline_predictions
    )
]

# Evaluate each trained model on the same validation set.
for model_name, model in candidate_models.items():
    predictions = model.predict(X_val)

    metrics_rows.append(
        regression_metrics(model_name, y_val, predictions)
    )

metrics_df = pd.DataFrame(metrics_rows)

Path("outputs").mkdir(exist_ok=True)
metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
    float_format="%.4f"
)

print("\nValidation model comparison:")
print(metrics_df.round(4).to_string(index=False))

# Rank the four candidates, excluding the baseline.
candidate_ranking = (
    metrics_df[metrics_df["model"] != "mean_baseline"]
    .sort_values("RMSE", kind="stable")
    .reset_index(drop=True)
)

winner_name = candidate_ranking.iloc[0]["model"]

print("\nBest model on validation:", winner_name)
print("Saved: outputs/part_b_metrics.csv")


#---------------------

# Compare the three learning rates required by the project.
learning_rate_results = []

for rate in [0.01, 0.1, 1.0]:
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )

    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_val)

    learning_rate_results.append({
        "learning_rate": rate,
        "val_rmse": np.sqrt(
            mean_squared_error(y_val, gb_predictions)
        ),
        "val_r2": r2_score(y_val, gb_predictions)
    })

learning_rate_df = pd.DataFrame(learning_rate_results)

learning_rate_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False,
    float_format="%.4f"
)

print("\nGradient Boosting learning-rate comparison:")
print(learning_rate_df.round(4).to_string(index=False))
print("Saved: outputs/part_b_gb_learning_rate.csv")


# Combine training and validation data for cross-validation.
X_train_val = pd.concat([X_train, X_val], ignore_index=True)
y_train_val = pd.concat([y_train, y_val], ignore_index=True)

# Use the same folds for both models for a fair comparison.
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Select the two best candidates from our validation ranking.
top_two_names = candidate_ranking.head(2)["model"].tolist()

cv_results = []

for model_name in top_two_names:
    scores = cross_val_score(
        candidate_models[model_name],
        X_train_val,
        y_train_val,
        cv=cv
    )

    # Default scoring for these regressors is R².
    cv_results.append({
        "model": model_name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

    print(f"\n{model_name} — five fold R2 scores:")
    print(np.round(scores, 4))

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False,
    float_format="%.4f"
)

print("\nCross-validation comparison:")
print(cv_df.round(4).to_string(index=False))
print("Saved: outputs/part_b_cv.csv")


# scaling checks and final test evaluation.

# Record scaling checks using the scaler already fitted
# on the 840 training rows inside the Linear Regression pipeline.
fitted_scaler = linear_model.named_steps["scaler"]

X_train_scaled = fitted_scaler.transform(X_train)
X_test_scaled = fitted_scaler.transform(X_test)

checks_df = pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_val)},
    {"metric": "test_rows", "value": len(X_test)},
    {
        "metric": "train_scaled_max_abs_mean",
        "value": np.abs(X_train_scaled.mean(axis=0)).max()
    },
    {
        "metric": "test_scaled_max_abs_mean",
        "value": np.abs(X_test_scaled.mean(axis=0)).max()
    },
])

checks_df.to_csv(
    "outputs/part_b_checks.csv",
    index=False
)

print("\nPart B checks:")
print(checks_df.to_string(index=False))

# Evaluate the validation winner on the test set.
# Keep its original fit on the 840 training rows.
final_model = candidate_models[winner_name]
test_predictions = final_model.predict(X_test)

test_mse = mean_squared_error(y_test, test_predictions)

test_row = {
    "model": winner_name,
    "split": "test",
    "MAE": mean_absolute_error(y_test, test_predictions),
    "MSE": test_mse,
    "RMSE": np.sqrt(test_mse),
    "R2": r2_score(y_test, test_predictions),
}

# Save five validation rows plus one final test row.
final_metrics_df = pd.concat(
    [metrics_df, pd.DataFrame([test_row])],
    ignore_index=True
)

final_metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
    float_format="%.4f"
)

print("\nFinal test results:")
print(pd.DataFrame([test_row]).round(4).to_string(index=False))
print("\nSaved: outputs/part_b_checks.csv")
print("Updated: outputs/part_b_metrics.csv")