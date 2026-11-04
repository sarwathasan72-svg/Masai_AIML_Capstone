
# Now we’ll use it to score the 48 new parcels and assign actions.
# Prediction	Hub segment	  Action
# Not flagged	Any segment	  No action
# Flagged	   Standard or Watch	Proactive customer call
# Flagged	   Escalate	Escalate to hub manager



from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# 1. Load the saved model and its supporting information.
delay_model = joblib.load("models/delay_model.pkl")

threshold = float(
    pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0]
)

feature_order = (
    pd.read_csv("models/feature_order.csv")["feature"].tolist()
)

impute_values = pd.read_csv("models/impute_values.csv")

stored_medians = dict(
    zip(impute_values["column"], impute_values["median"])
)

# 2. Read new shipments, hub information and hub segments.
new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
hub_segments = pd.read_csv("outputs/hub_segments.csv")

assert new_shipments["shipment_id"].is_unique, (
    "New shipment IDs must be unique."
)

# Remember the original order for the final output.
original_ids = new_shipments["shipment_id"].tolist()

# 3. Reuse the shared cleaning and feature-building functions.
# Fill missing values using historical medians, not new-batch medians.
clean_new, _ = clean_shipments(
    new_shipments,
    medians=stored_medians,
)

prepared_new = build_features(clean_new, hubs)

assert len(prepared_new) == len(new_shipments), (
    "Preparation changed the number of new shipments."
)

# 4. Arrange all 17 inputs in the saved order.
# The loaded pipeline selects its seven features internally.
X_new = prepared_new[feature_order].copy()

assert not X_new.isna().any().any(), (
    "Missing values found in model inputs."
)

# 5. Predict the probability of the delayed class, labelled 1.
positive_column = list(delay_model.classes_).index(1)

probabilities = delay_model.predict_proba(X_new)[
    :, positive_column
]

# Decide flags BEFORE rounding probabilities for display.
flagged = (probabilities >= threshold).astype(int)

# 6. Attach predictions to shipment identifiers.
actions = prepared_new[["shipment_id", "hub_id"]].copy()

actions["delay_probability"] = probabilities
actions["flagged"] = flagged

# 7. Attach the segment assigned to each hub.
actions = actions.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one",
    sort=False,
)

assert actions["segment"].notna().all(), (
    "Some shipments have no matching hub segment."
)

# 8. Apply the operations action rules.
actions["action"] = np.select(
    [
        (actions["flagged"] == 1)
        & (actions["segment"] == "Escalate"),

        actions["flagged"] == 1,
    ],
    [
        "Escalate to hub manager",
        "Proactive customer call",
    ],
    default="No action",
)

# 9. Restore the exact input-file order.
actions = (
    actions.set_index("shipment_id")
    .loc[original_ids]
    .reset_index()
)

# Keep only the required columns, in the required order.
actions = actions[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
].copy()

# Round only the reported probability.
actions["delay_probability"] = (
    actions["delay_probability"].round(4)
)

# 10. Validate the final output.
assert len(actions) == 48, "Expected 48 new shipment results."
assert actions["shipment_id"].is_unique, (
    "Duplicate shipment IDs in output."
)
assert not actions.isna().any().any(), (
    "Missing values found in the action list."
)
assert actions["shipment_id"].tolist() == original_ids, (
    "Output order differs from input order."
)

# Confirm that the exported probabilities agree with the flags.
assert np.array_equal(
    actions["flagged"].to_numpy(),
    (
        actions["delay_probability"].to_numpy() >= threshold
    ).astype(int),
), "Rounded probabilities disagree with the alert flags."

# 11. Save the final action list.
Path("outputs").mkdir(exist_ok=True)

actions.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False,
    float_format="%.4f",
)

# 12. Display the results.
print("New shipments scored:", len(actions))
print("Alert threshold:", threshold)
print("Flagged shipments:", int(actions["flagged"].sum()))

print(
    "Hub-manager escalations:",
    int(
        (actions["action"] == "Escalate to hub manager").sum()
    ),
)

print("\nAction counts:")
print(actions["action"].value_counts().to_string())

print("\nAction list preview:")
print(actions.head(10).to_string(index=False))

print("\nSaved: outputs/new_shipment_actions.csv")