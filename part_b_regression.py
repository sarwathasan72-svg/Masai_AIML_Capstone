import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURE_COLUMNS


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



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