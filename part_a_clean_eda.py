import pandas as pd
from src.shipment_prep import clean_shipments, build_features


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
    "outputs/delay_by_group.csv",
    index=False,
    float_format="%.4f"
)

print("\nDelay comparison:")
print(delay_by_group.to_string(index=False))
print("\nSaved: outputs/delay_by_group.csv")