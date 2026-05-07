# Exoplanet Atmosphere Clustering

This project clusters exoplanet atmospheric compositions using HDBSCAN and visualizes the results with pairwise plots and PCA.

Earth is included as a reference point in the visualizations.

---

## Requirements

Download the requirements.txt file.
Install the required packages:

```bash
pip install -r requirements.txt
```

--- 

## How to Run

Download the main.py, dbscan_main.py, and psg_models0.csv and place them all into one folder.

Run ```python main.py``` or ```python dbscan_main.py``` to run the code.

--- 

## Outputs

2 directories will be created in the folder the code is located, feature_plots and cluster_outputs.

feature_plots will contain all created scatterplots, and cluster_outputs will contain csv files with the input data for each cluster.
