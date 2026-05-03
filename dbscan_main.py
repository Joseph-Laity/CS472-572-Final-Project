import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# =========================
# 1. Load data
# =========================
DATASET = 1
df = pd.read_csv(f"psg_models{DATASET}.csv")

earth_data = {
    "H2O": 0.010000,
    "CO2": 0.000420,
    "O2": 0.207000,
    "N2": 0.772000,
    "CH4": 0.0000019,
    "N2O": 0.0000003,
    "CO": 0.0000001,
    "O3": 0.00000007,
    "SO2": 0.00000001,
    "NH3": 0.000000001,
    "C2H6": 0.000000002,
    "NO2": 0.00000002
}

# Feature selection
df = df.loc[:, "H2O":"NO2"]
feature_names = df.columns.tolist()

# Convert to numpy
X = df.to_numpy()

# Add Earth
earth_vector = np.array([earth_data[col] for col in feature_names])
X = np.vstack([X, earth_vector])

earth_index = len(X) - 1

# =========================
# 2. Scale features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. Run DBSCAN
# =========================
clusterer = DBSCAN(
    eps=1.1,        # TUNE THIS
    min_samples=2
)

labels = clusterer.fit_predict(X_scaled)

# =========================
# 4. Assign Earth to nearest cluster (OPTION 1)
# =========================
cluster_mask = labels != -1

cluster_points = X_scaled[cluster_mask]
cluster_labels = labels[cluster_mask]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(cluster_points)

dist, idx = nn.kneighbors([X_scaled[earth_index]])

earth_label = cluster_labels[idx[0][0]]

# =========================
# 5. Remove noise for plotting
# =========================
mask = labels != -1

X_plot_raw = X[mask]
X_plot_scaled = X_scaled[mask]
labels_plot = labels[mask]

# Label mapping
unique_labels = np.unique(labels_plot)
label_map = {label: i for i, label in enumerate(unique_labels)}
labels_mapped = np.array([label_map[l] for l in labels_plot])

# Earth color
earth_color = plt.get_cmap("tab20", len(unique_labels))(label_map[earth_label])

print("Cluster labels (no noise):")
print(unique_labels)

print(f"\nNumber of clusters: {len(unique_labels)}")
print(f"Number of noise points removed: {np.sum(labels == -1)}")
print(f"Earth cluster: {earth_label}")

# =========================
# 6. Pairwise plots
# =========================
output_dir = "feature_plots"
os.makedirs(output_dir, exist_ok=True)

n_features = len(feature_names)

for i in range(n_features):
    for j in range(i + 1, n_features):

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            X_plot_raw[:, i],
            X_plot_raw[:, j],
            c=labels_mapped,
            cmap="tab20",
            alpha=0.7
        )

        # Earth overlay (RAW)
        plt.scatter(
            earth_vector[i],
            earth_vector[j],
            s=250,
            c=[earth_color],
            edgecolors='blue',
            linewidths=2.5,
            marker='o',
            zorder=10
        )

        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.title(f"{feature_names[i]} vs {feature_names[j]}")

        cbar = plt.colorbar(scatter)
        cbar.set_label("Cluster ID")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dbscan_{feature_names[i]}_vs_{feature_names[j]}.png"))
        plt.close()

print(f"Saved pairwise plots to {output_dir}")

# =========================
# 7. PCA
# =========================
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Earth PCA projection
earth_2d = X_2d[earth_index]

# =========================
# 8. PCA plot
# =========================
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_2d[mask, 0],
    X_2d[mask, 1],
    c=labels_mapped,
    cmap="tab20",
    alpha=0.7
)

plt.scatter(
    earth_2d[0],
    earth_2d[1],
    s=300,
    c=[earth_color],
    edgecolors='blue',
    linewidths=2.5,
    marker='o',
    zorder=10
)

plt.title("DBSCAN Clusters (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter).set_label("Cluster ID")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dbscan_PCA_projection.png"))
plt.close()

# =========================
# 9. Export clusters as index lists (NO EARTH, NO CLUSTER COLUMN)
# =========================

export_dir = "cluster_outputs"
os.makedirs(export_dir, exist_ok=True)

# Remove Earth before indexing
labels_no_earth = labels[:-1]

# Original dataframe (no Earth included)
df_no_earth = df.copy()

# Add original index as a column (this is what you want to preserve)
df_no_earth["original_index"] = df_no_earth.index

# Export indices per cluster
for cluster_id in np.unique(labels_no_earth):

    if cluster_id == -1:
        continue

    # Get indices of rows in this cluster
    cluster_indices = df_no_earth.index[labels_no_earth == cluster_id].to_numpy()

    # Save just indices
    out_path = os.path.join(export_dir, f"cluster_{cluster_id}_indices.csv")
    
    pd.DataFrame({"index": cluster_indices}).to_csv(out_path, index=False)

print(f"Saved cluster index files to '{export_dir}'")
