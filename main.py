import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1. Load data
# =========================
df = pd.read_csv("psg_models.csv")

# Select column range (inclusive)
df = df.loc[:, "H2O":"NO2"]

feature_names = df.columns.tolist()

# Convert to numpy
X = df.to_numpy()

# =========================
# 2. Scale features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. Run HDBSCAN
# =========================
clusterer = HDBSCAN(
    min_cluster_size=5,
    min_samples=1,
    cluster_selection_method="leaf"
)

labels = clusterer.fit_predict(X_scaled)

# =========================
# 4. Remove noise
# =========================
mask = labels != -1
X_plot = X_scaled[mask]
labels_plot = labels[mask]

# =========================
# 5. Normalize labels for consistent coloring
# =========================
unique_labels = np.unique(labels_plot)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
labels_mapped = np.array([label_map[l] for l in labels_plot])

num_clusters = len(unique_labels)

print("Cluster labels (no noise):")
print(unique_labels)

print(f"\nNumber of clusters: {num_clusters}")
print(f"Number of noise points removed: {np.sum(labels == -1)}")

# =========================
# 6. Create output folder
# =========================
output_dir = "feature_plots"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 7. Colormap
# =========================
cmap = plt.get_cmap("tab20", num_clusters)

# =========================
# 8. Pairwise feature plots
# =========================
n_features = len(feature_names)

for i in range(n_features):
    for j in range(i + 1, n_features):

        plt.figure()

        scatter = plt.scatter(
            X_plot[:, i],
            X_plot[:, j],
            c=labels_mapped,
            cmap=cmap
        )

        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.title(f"{feature_names[i]} vs {feature_names[j]} (HDBSCAN)")

        cbar = plt.colorbar(scatter)
        cbar.set_label("Cluster ID")

        filename = f"{feature_names[i]}_vs_{feature_names[j]}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath)
        plt.close()

print(f"\nSaved pairwise plots to '{output_dir}'")

# =========================
# 9. PCA for visualization
# =========================
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_plot)

# =========================
# 10. PCA Plot
# =========================
plt.figure()

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=labels_mapped,
    cmap=cmap
)

plt.title("HDBSCAN Clusters (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

cbar = plt.colorbar(scatter)
cbar.set_label("Cluster ID")

plt.savefig(os.path.join(output_dir, "PCA_projection.png"))
plt.close()