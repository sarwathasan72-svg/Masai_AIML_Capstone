# Part D: Hub Segments, Packaging and Operations Playbook.

# Part D has three stages:
# Stage	File	Purpose
# 1. HHub segmentation	part_d_segments.py	Summarize the 24 hubs and use K-Means to group them into Standard, Watch and Escalate.
# 2. Model packaging	package_model.py	Save the selected Part C classifier and verify that reloading it gives identical predictions.
# 3. New-parcel scoring	predict_delay.py	Predict risk for the 48 new parcels and assign actions using their hub segments.


from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import numpy as np

# 1. Read the historical modelling table created in Part A.
modelling_df = pd.read_csv("data/modelling_table.csv")

# 2. Ensure the output folder exists.
Path("outputs").mkdir(exist_ok=True)

# 3. Summarize shipments into one row per hub.
hub_profile = (
    modelling_df
    .groupby("hub_id")
    .agg(
        shipments=("hub_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)

# 4. Check the expected structure.
assert len(hub_profile) == 24, "Expected exactly 24 hubs."
assert hub_profile["shipments"].eq(50).all(), (
    "Expected exactly 50 shipments per hub."
)

# 5. Save the required hub profile.
hub_profile.to_csv(
    "outputs/hub_profile.csv",
    index=False,
    float_format="%.4f",
)

# 6. Display a summary.
print("Number of hubs:", len(hub_profile))
print("Total shipments:", hub_profile["shipments"].sum())

print("\nHub profile preview:")
print(hub_profile.head().to_string(index=False))

print("\nSaved: outputs/hub_profile.csv")

# group hubs with similar operating behaviour using K-Means.
# K-Means discovers groups based on similarities—this is unsupervised learning.

# 7. Select the four hub-performance columns for clustering.
PROFILE_COLUMNS = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

# Keep hubs in the order required by the project.
hub_profile = (
    hub_profile
    .sort_values("hub_id")
    .reset_index(drop=True)
)

# 8. Scale the columns so different units are comparable.
scaler = StandardScaler()

scaled_profiles = scaler.fit_transform(
    hub_profile[PROFILE_COLUMNS]
)

# 9. Group the hubs into three clusters.
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

hub_profile["cluster"] = kmeans.fit_predict(scaled_profiles)

# 10. Inspect the groups.
print("\nHub cluster assignments:")
print(
    hub_profile[
        ["hub_id", "delay_rate", "cluster"]
    ].to_string(index=False)
)

cluster_sizes = (
    hub_profile.groupby("cluster")
    .size()
    .rename("hubs")
)

print("\nNumber of hubs per cluster:")
print(cluster_sizes.to_string())

print(
    "\nCluster sizes, largest first:",
    sorted(cluster_sizes.tolist(), reverse=True),
)



# 11. Identify every unique pair of hubs.
# For 24 hubs: 24 * 23 / 2 = 276 pairs.
pair_i, pair_j = np.triu_indices(
    len(scaled_profiles),
    k=1,
)

# Euclidean distance between each pair's scaled profiles.
pair_distances = np.linalg.norm(
    scaled_profiles[pair_i] - scaled_profiles[pair_j],
    axis=1,
)

print("\nNumber of unique hub pairs:", len(pair_distances))

# 12. Compare separation for different numbers of clusters.
separation_rows = []

for k in [2, 3, 4, 5]:
    comparison_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    comparison_labels = comparison_model.fit_predict(
        scaled_profiles
    )

    # True when both hubs belong to the same cluster.
    same_cluster = (
        comparison_labels[pair_i]
        == comparison_labels[pair_j]
    )

    mean_within = pair_distances[same_cluster].mean()
    mean_across = pair_distances[~same_cluster].mean()

    separation_ratio = mean_across / mean_within

    # Format cluster sizes from largest to smallest.
    sizes = (
        pd.Series(comparison_labels)
        .value_counts()
        .sort_values(ascending=False)
        .tolist()
    )

    separation_rows.append({
        "K": k,
        "across_over_within": separation_ratio,
        "cluster_sizes": "-".join(str(size) for size in sizes),
    })

# 13. Save the required comparison.
separation_df = pd.DataFrame(separation_rows)

separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False,
    float_format="%.4f",
)

print("\nCluster separation comparison:")
print(separation_df.round(4).to_string(index=False))

print("\nSaved: outputs/part_d_separation.csv")


# 14. Rank the original three clusters by average delay rate.
# Use the K=3 assignments already stored in hub_profile["cluster"].
cluster_delay_rates = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

segment_order = ["Standard", "Watch", "Escalate"]

# Build the mapping from calculated results, not hard-coded cluster IDs.
cluster_to_segment = dict(
    zip(cluster_delay_rates.index, segment_order)
)

hub_profile["segment"] = (
    hub_profile["cluster"].map(cluster_to_segment)
)

# 15. Attach hub city and existing tier.
hubs = pd.read_csv("data/hubs.csv")

hub_segments = (
    hub_profile[["hub_id", "segment"]]
    .merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one",
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

assert len(hub_segments) == 24, "Expected 24 hub assignments."
assert not hub_segments.isna().any().any(), (
    "Missing values found in hub assignments."
)

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False,
)

# 16. Summarize each segment using the hub-level values.
segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "size"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
    .reindex(segment_order)
    .reset_index()
)

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False,
    float_format="%.4f",
)

# 17. Compare observed segments with what the tier labels imply.
tier_to_expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}

expected_segments = (
    hub_segments["hub_tier"].map(tier_to_expected_segment)
)

tier_mismatches = hub_segments.loc[
    hub_segments["segment"] != expected_segments
].copy()

tier_mismatches.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False,
)

# 18. Display the results.
print("\nCluster-to-segment mapping:")
print(cluster_to_segment)

print("\nSegment profile:")
print(segment_profile.round(4).to_string(index=False))

print("\nHubs whose tier and segment differ:")
print(tier_mismatches.to_string(index=False))

print("\nSaved: outputs/hub_segments.csv")
print("Saved: outputs/segment_profile.csv")
print("Saved: outputs/tier_vs_segment_mismatch.csv")