# innate-potato
These scripts intend to provide a way to efficiently analyse high dimensional mass cytometry data.
The essence of the project is to combine dimentionality reduction (using UMAP) and density-based clustering (using HDBSCAN). 
The scripts provide:
* cluster visualization on a 2D space
* marker intensity on a 2D space
* clustering quality control
* visualization of the clusters' sizes over time

Different approaches can be used:

## Reducing the dimensionality of the data before clustering on 2 dimensions. 
NOTE: reducing the dimensionality before clustering induces a loss of information and might diminish the efficiency and precision of the clustering.
* using ***UMAP_to_clusters.py***

## Clustering on a high dimensional dataset and using dimensionality reduction for visualization purposes.
NOTE: Without reducing the dimensonality of the data before clustering, HDBSCAN might not be able to detect fine overdensities, this is why this approach should be used on large datasets 
* using ***clusters_to_UMAP.py***
