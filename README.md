# Cytof deep phenotyping using dimensionality reduction and density-based clustering
# Overview
These scripts intend to provide a way to efficiently analyse high dimensional mass cytometry data.
The essence of the project is to combine dimentionality reduction (using UMAP) and density-based clustering (using HDBSCAN). 
The scripts provide:
* cluster visualization in a 2D space
* marker intensity in a 2D space
* clustering quality control
* visualization of the clusters' sizes over time

Different approaches can be used:

## Reducing the dimensionality of the data before clustering on 2 dimensions. 
NOTE: This approach should be used on smaller datasets.
* using ***Serial.py***

![image](https://user-images.githubusercontent.com/66411147/137888387-30fc2a02-c250-4d10-9d19-76459a2be03f.png)


## Clustering on a high dimensional dataset and using dimensionality reduction for visualization purposes.
NOTE: This approach should be used on larger datasets .
* using ***parallel.py***

![image](https://user-images.githubusercontent.com/66411147/137888441-6d5ba92e-5604-4203-add8-e9fdeb71ed63.png)


## Control samples automated quality check
* using QC.py

![image](https://user-images.githubusercontent.com/66411147/137936127-a5d9b6b1-3eb1-4e73-b2f3-6e3e7e1ede7b.png)


## Outputs

### Cluster visualization

![D28](https://user-images.githubusercontent.com/66411147/144065154-8708517f-521d-4b5e-922b-faf74a599ec7.png)

Such outputs are generated for each timepoint

### Cell population identification

![image](https://user-images.githubusercontent.com/66411147/144065851-8c9566e3-4896-4f6f-81c6-163ea755881b.png)

Allows manual identification of cell populations

### Evolution of cluster sizes along timepoints

![cluster_composition](https://user-images.githubusercontent.com/66411147/144067963-0e05b9b9-9e1b-4423-8095-184c2b8dec84.png)

All timepoints were clustered together. Their sizes vary with timepoints. 

### Cluster quality control visualization

![image](https://user-images.githubusercontent.com/66411147/137884372-824352bd-a2a6-46e4-b7ab-fd3cb0a03830.png)
![0QC](https://user-images.githubusercontent.com/66411147/137885696-435629e3-9b87-4a6f-9b80-6cb5840cf813.png)













