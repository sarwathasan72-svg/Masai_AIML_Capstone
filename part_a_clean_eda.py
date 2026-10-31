import pandas as pd
# from src.shipment_prep import clean_shipments, build_features

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURE_COLUMNS,
)

# 1. Read the original shipment file.
raw_ship_df = pd.read_csv("data/raw_shipments.csv")

# 2. Run the cleaning function.
# It returns the cleaned DataFrame and the medians used.
clean_ship_df, medians = clean_shipments(raw_ship_df)

# 3. Save the cleaned data to a SEPARATE file.
clean_ship_df.to_csv("data/clean_shipments.csv", index=False)

# 4. Display a summary.
print("Original rows:", len(raw_ship_df))
print("Cleaned rows:", len(clean_ship_df))
print("Unique service types:", clean_ship_df["service_type"].unique())
print("Medians used:", medians)
print("Cleaned data saved to data/clean_shipments.csv")
print("Missing values in 'payment_mode':", clean_ship_df["payment_mode"].isna().sum())
print("Payment mode counts:", clean_ship_df["payment_mode"].value_counts())
print("Declared value statistics:", clean_ship_df["declared_value_inr"].describe())

# 5. Read hub information.
hubs_df = pd.read_csv("data/hubs.csv")

# 6. Add hub details and prepare features.
modelling_df = build_features(clean_ship_df, hubs_df)

# 7. Create the historical delay target.
modelling_df["is_delayed"] = (
    modelling_df["delivery_hours"] > modelling_df["promised_hours"]
).astype(int)

# 8. Save the prepared modelling table.
modelling_df.to_csv("data/modelling_table.csv", index=False)

# 9. Display the results.
print("\nTotal shipments:", len(modelling_df))
print("Delayed shipments:", modelling_df["is_delayed"].sum())
print("Modelling table saved to data/modelling_table.csv")

modelling_table_df = pd.read_csv("data/modelling_table.csv")
print(modelling_table_df.info())
print("\nFirst 10 rows for Modelling:")
print(modelling_table_df.head(10))
print(modelling_table_df.describe())
print(modelling_table_df.shape)

from pathlib import Path

# Create a folder for analysis results.
Path("outputs").mkdir(exist_ok=True)

# Use a separate copy for analysis.
analysis_df = modelling_df.copy()

# Divide distances into the required groups.
analysis_df["distance_band"] = pd.cut(
    analysis_df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True
)

# Collect the summary for each grouping.
group_results = []

for column in ["service_type", "hub_tier", "distance_band"]:
    summary = (
        analysis_df.groupby(column, observed=True)
        .agg(
            shipments=("shipment_id", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean")
        )
        .reset_index()
    )

    summary = summary.rename(columns={column: "group"})
    summary.insert(0, "dimension", column)

    group_results.append(summary)

# Combine the summaries into one DataFrame.
delay_by_group = pd.concat(group_results, ignore_index=True)

# Save the required CSV.
delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False,
    float_format="%.4f"
)

print("\nDelay comparison:")
print(delay_by_group.to_string(index=False))
print("\nSaved: outputs/delay_by_group.csv")

# Prepare a separate table for "before cleaning" counts.
# The brief requires these counts AFTER removing duplicates.
before_cleaning = raw_ship_df.drop_duplicates().copy()

# Calculate delivery-hour quartiles and outlier boundaries.
hours = modelling_df["delivery_hours"]

hours_q1 = hours.quantile(0.25)
hours_q3 = hours.quantile(0.75)
hours_iqr = hours_q3 - hours_q1

hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr

hours_outliers = (
    (hours < hours_lower_bound)
    | (hours > hours_upper_bound)
)

# Columns that must have no missing values.
required_columns = FEATURE_COLUMNS + [
    "delivery_hours",
    "is_delayed",
]

# Compute every check directly from the data.
checks = {
    "raw_rows": len(raw_ship_df),

    "duplicate_rows_removed": (
        len(raw_ship_df) - len(before_cleaning)
    ),

    "rows_after_dedup": len(before_cleaning),

    "missing_distance_km_before": (
        before_cleaning["distance_km"].isna().sum()
    ),

    "missing_parcel_weight_before": (
        before_cleaning["parcel_weight_kg"].isna().sum()
    ),

    "sentinel_weight_rows": (
        before_cleaning["parcel_weight_kg"] > 100
    ).sum(),

    "missing_declared_value_before": (
        before_cleaning["declared_value"].isna().sum()
    ),

    "missing_payment_mode_before": (
        before_cleaning["payment_mode"].isna().sum()
    ),

    "service_type_spellings_before": (
        before_cleaning["service_type"].nunique()
    ),

    "service_type_spellings_after": (
        clean_ship_df["service_type"].nunique()
    ),

    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],

    "rows_after_hub_merge": len(modelling_df),

    "nulls_in_features_and_target": (
        modelling_df[required_columns].isna().sum().sum()
    ),

    "delayed_count": modelling_df["is_delayed"].sum(),
    "delay_rate": modelling_df["is_delayed"].mean(),

    "hours_q1": hours_q1,
    "hours_q3": hours_q3,
    "hours_lower_bound": hours_lower_bound,
    "hours_upper_bound": hours_upper_bound,
    "hours_outlier_count": hours_outliers.sum(),

    "declared_value_mean_inr": (
        modelling_df["declared_value_inr"].mean()
    ),

    "declared_value_median_inr": (
        modelling_df["declared_value_inr"].median()
    ),

    "log_declared_value_mean": (
        modelling_df["log_declared_value"].mean()
    ),

    "log_declared_value_median": (
        modelling_df["log_declared_value"].median()
    ),
}

# Convert the dictionary into the required two-column table.
checks_df = pd.DataFrame(
    checks.items(),
    columns=["metric", "value"]
)

Path("outputs").mkdir(exist_ok=True)

checks_df.to_csv(
    "outputs/part_a_checks.csv",
    index=False,
    float_format="%.4f"
)

print("\nPart A checks:")
print(checks_df.to_string(index=False))
print("\nSaved: outputs/part_a_checks.csv")

# Calculate correlations between numeric columns.
correlations_df = modelling_df.select_dtypes(
    include="number"
).corr()

# Keep row labels so each correlation can be identified.
correlations_df.to_csv(
    "outputs/part_a_correlations.csv",
    index=True,
    float_format="%.4f"
)

# Show the three strongest correlations with delivery hours,
# ranked by absolute strength while preserving their signs.
hours_correlations = correlations_df["delivery_hours"].drop(
    "delivery_hours"
)

top_three = hours_correlations.loc[
    hours_correlations.abs().sort_values(
        ascending=False
    ).head(3).index
]

print("\nTop three correlations with delivery_hours:")
print(top_three.round(4))
print("\nSaved: outputs/part_a_correlations.csv")

# Check preparation of new shipments using historical medians.
new_ship_df = pd.read_csv("data/new_shipments.csv")

new_clean_df, _ = clean_shipments(
    new_ship_df,
    medians=medians
)

new_features_df = build_features(new_clean_df, hubs_df)

missing_features = (
    new_features_df[FEATURE_COLUMNS].isna().sum().sum()
)

print("\nNew shipment preparation check:")
print("Input rows:", len(new_ship_df))
print("Prepared rows:", len(new_features_df))
print("Missing feature values:", missing_features)

assert len(new_features_df) == len(new_ship_df), (
    "The shipment count changed during preparation."
)
assert missing_features == 0, (
    "Some model features still contain missing values."
)