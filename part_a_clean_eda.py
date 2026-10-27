import pandas as pd
from src.shipment_prep import clean_shipments

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
