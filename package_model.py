# package the Part C classifier so another script can load it and predict new shipments.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURE_COLUMNS, clean_shipments


# 1. Create the output folders if needed.
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# 2. Read the configuration selected in Part C.
config_df = pd.read_csv("outputs/part_c_final_config.csv")

assert len(config_df) == 1, "Expected one final configuration."
config = config_df.iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

# Convert the CSV representation back to Python's None.
depth_text = str(config["max_depth"]).strip().lower()
max_depth = (
    None
    if depth_text == "unlimited"
    else int(float(depth_text))
)

# 3. Read the historical modelling table.
modelling_df = pd.read_csv("data/modelling_table.csv")

X = modelling_df[FEATURE_COLUMNS].copy()
y = modelling_df["is_delayed"]

# 4. Recreate exactly the same stratified split as Part C.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

# 5. Rebuild the chosen pipeline and fit on training rows only.
delay_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )),
])

delay_model.fit(X_train, y_train)

# 6. Save the fitted pipeline.
joblib.dump(delay_model, "models/delay_model.pkl")

# 7. Save the chosen threshold.
pd.DataFrame({
    "threshold": [threshold],
}).to_csv(
    "models/delay_threshold.csv",
    index=False,
    float_format="%.4f",
)

# 8. Save all 17 input names in their required order.
# The pipeline performs its seven-feature selection internally.
pd.DataFrame({
    "feature": FEATURE_COLUMNS,
}).to_csv(
    "models/feature_order.csv",
    index=False,
)

# 9. Recompute the historical cleaning medians using shared logic.
raw_shipments = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw_shipments)

pd.DataFrame(
    list(medians.items()),
    columns=["column", "median"],
).to_csv(
    "models/impute_values.csv",
    index=False,
    float_format="%.4f",
)

# 10. Reload the saved pipeline.
reloaded_model = joblib.load("models/delay_model.pkl")

# Select the probability column for the delayed class, labelled 1.
original_positive_column = list(delay_model.classes_).index(1)
reloaded_positive_column = list(reloaded_model.classes_).index(1)

original_probabilities = delay_model.predict_proba(X_test)[
    :, original_positive_column
]

reloaded_probabilities = reloaded_model.predict_proba(X_test)[
    :, reloaded_positive_column
]

# 11. Verify that saving/loading did not change predictions.
max_probability_difference = float(
    np.max(
        np.abs(original_probabilities - reloaded_probabilities)
    )
)

assert len(X_test) == 180, "Expected 180 test rows."
assert max_probability_difference <= 1e-9, (
    "Reloaded model predictions differ from the original."
)

reload_checks = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test),
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_probability_difference,
    },
    {
        "metric": "threshold",
        "value": threshold,
    },
])

reload_checks.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False,
    float_format="%.4f",
)

# 12. Display the results.
print("Training rows:", len(X_train))
print("Selected features:", selector_k)
print("Number of trees:", n_estimators)
print("Maximum depth:", max_depth)
print("Minimum samples per leaf:", min_samples_leaf)
print("Alert threshold:", threshold)

print("\nModel reload check:")
print(reload_checks.to_string(index=False))

print("\nSaved: models/delay_model.pkl")
print("Saved: models/delay_threshold.csv")
print("Saved: models/feature_order.csv")
print("Saved: models/impute_values.csv")
print("Saved: outputs/part_d_reload_check.csv")