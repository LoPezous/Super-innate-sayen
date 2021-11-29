# Cytof deep phenotyping and exploration of trained immunity mechanisms
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

![D28](https://user-images.githubusercontent.com/66411147/143853892-cb4d2173-afbe-4f59-9f33-1c9404c487f0.png)

Such outputs are generated for each timepoint

### Cell population identification

![image](https://user-images.githubusercontent.com/66411147/143854342-07b24d79-b758-4fed-ada2-4b2aca375c25.png)

Allows manual identification of cell populations

### Evolution of clustert sizes along timeline

![image](https://user-images.githubusercontent.com/66411147/137884285-32f91434-9e85-40d5-ac41-b4c94151f49d.png)

All timepoints were clustered together. Their sizes vary with timepoints. 

### Cluster quality control visualization

![image](https://user-images.githubusercontent.com/66411147/137884372-824352bd-a2a6-46e4-b7ab-fd3cb0a03830.png)
![0QC](https://user-images.githubusercontent.com/66411147/137885696-435629e3-9b87-4a6f-9b80-6cb5840cf813.png)


Such outputs are generated for each cluster

### Marker quality control file 

QC_list.txt 

## NOTE

control samples quality controls (***QC.py***) and clusters quality controls (inside ***Serial.py*** and ***Parallel.py***) are DIFFERENT. ***QC.py*** checks the validity of the markers while cluster quality control check the validity of the clustering





