import pandas as pd
import numpy as np


def clean_shipments(raw, medians=None):
    # Work on a copy. Keep the original DataFrame unchanged.
    clean_df = raw.copy()

    # 1. Remove exact duplicate rows.
    clean_df = clean_df.drop_duplicates()

    # 2. Standardize service names.
    # Example: " EXPRESS " becomes "Express".
    clean_df["service_type"] = (
         clean_df["service_type"].str.strip().str.title()
     )

    # 3. Fill missing payment modes.
    clean_df["payment_mode"] = (
         clean_df["payment_mode"].fillna("Unknown")
     )

    # 4. Convert money stored as text into numbers.
    # Example: "₹1,250.50" becomes 1250.50.
    amount_text = (
         clean_df["declared_value"]
         .str.replace("₹", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.strip()
     )

    clean_df["declared_value_inr"] = pd.to_numeric(amount_text)

    # Remove the old text column.
    clean_df = clean_df.drop(columns=["declared_value"])

    # 5. Convert dispatch timestamps into datetime values.
    clean_df["dispatch_time"] = pd.to_datetime(
         clean_df["dispatch_time"]
     )

    # 6. Treat weights above 100 kg as missing,
    # as required by this project's instructions.
    clean_df.loc[
         clean_df["parcel_weight_kg"] > 100,
         "parcel_weight_kg"
     ] = np.nan

    # 7. Calculate medians only if none were supplied.
    if medians is None:
         medians = {
             "distance_km": float(
                 clean_df["distance_km"].median()
             ),
             "parcel_weight_kg": float(
                 clean_df["parcel_weight_kg"].median()
             ),
             "declared_value_inr": float(
                 clean_df["declared_value_inr"].median()
             ),
         }

    # Fill missing values using these medians.
    clean_df = clean_df.fillna(medians)

    # Return both the cleaned data and the medians used.
    return clean_df, medians

#  Build features for modeling. This function is used in part_b_model.py.
def build_features(clean, hubs):
    # Attach hub information to each shipment using hub_id.
    feature_df = clean.merge(
        hubs,
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    # Convert service type into promised delivery hours.
    promised_hours = {
        "Express": 36,
        "Standard": 60,
        "Economy": 96
    }

    feature_df["promised_hours"] = (
        feature_df["service_type"].map(promised_hours).astype(int)
    )

    # COD = 1; all other payment modes = 0.
    feature_df["is_cod"] = (
        feature_df["payment_mode"] == "COD"
    ).astype(int)

    # Monday = 0, ..., Saturday = 5, Sunday = 6.
    feature_df["is_weekend"] = (
        feature_df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # Extract the dispatch hour: 0 to 23.
    feature_df["dispatch_hour"] = (
        feature_df["dispatch_time"].dt.hour
    )

    # Apply the natural logarithm to the declared value.
    feature_df["log_declared_value"] = np.log(
        feature_df["declared_value_inr"]
    )

    return feature_df