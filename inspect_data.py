import numpy as np
import pandas as pd

# Read the new shipments file
# new_ship_df = pd.read_csv("data/new_shipments.csv")
# hub_df = pd.read_csv("data/hubs.csv")

# Display the first 5 rows
# print(new_ship_df.head())
# print(hub_df.head(10))
# print(hub_df.shape)
# raw_ship_df = pd.read_csv("data/raw_shipments.csv")
# print(raw_ship_df.columns)
# print(raw_ship_df.head())
# print(raw_ship_df.shape)
# print(len(raw_ship_df))
# print(raw_ship_df.sample(5, random_state=7))

# print(raw_ship_df.isnull().sum())
# print(raw_ship_df.info())
# print(raw_ship_df.columns.to_list())
# disonly only those column having missing values
# missing_cols = raw_ship_df.columns[raw_ship_df.isnull().any()].to_list()
# missing_count = raw_ship_df.isna().sum()
# print(missing_count[missing_count > 0])
# find total oder in each service type
# service_type_counts = raw_ship_df['service_type'].value_counts()
# print(service_type_counts)
# find % contribution of each service type
# service_type_percentage = raw_ship_df['service_type'].value_counts(normalize=True) * 100
# print(service_type_percentage)  
# payment_mode: replace a missing value with the placeholder "Unknown";
# payment_mode_missing_count = raw_ship_df['payment_mode'].isnull().sum()
# print(f"Missing values in 'payment_mode': {payment_mode_missing_count}")
# raw_ship_df['payment_mode'].fillna('Unknown', inplace=True)  
#  
#  Temp File 
# raw_ship_df = pd.read_csv("data/raw_shipments.csv")
# print(raw_ship_df.shape)

clean_shipments_df = pd.read_csv("data/clean_shipments.csv")
print(clean_shipments_df.shape)
print(clean_shipments_df.duplicated().sum())
print(clean_shipments_df["service_type"].unique())
print(clean_shipments_df["service_type"].value_counts())
#What is the % contribution of each service type?
print(clean_shipments_df["service_type"].value_counts(normalize=True) * 100)
print(clean_shipments_df["payment_mode"].isna().sum())
print(clean_shipments_df["payment_mode"].value_counts())
print("Pandas:",pd.__version__)
print("Numpy:",np.__version__)
print(clean_shipments_df.head(10))

