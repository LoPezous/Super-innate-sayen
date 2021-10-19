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
NOTE: reducing the dimensionality before clustering induces a loss of information and might diminish the efficiency and precision of the clustering.
* using ***Serial.py***

![image](https://user-images.githubusercontent.com/66411147/137888387-30fc2a02-c250-4d10-9d19-76459a2be03f.png)


## Clustering on a high dimensional dataset and using dimensionality reduction for visualization purposes.
NOTE: Without reducing the dimensonality of the data before clustering, HDBSCAN might not be able to detect fine overdensities, this is why this approach should be used on large datasets 
* using ***parallel.py***

![image](https://user-images.githubusercontent.com/66411147/137888441-6d5ba92e-5604-4203-add8-e9fdeb71ed63.png)


## Control samples automated quality check
* using QC.py

![image](https://user-images.githubusercontent.com/66411147/137936127-a5d9b6b1-3eb1-4e73-b2f3-6e3e7e1ede7b.png)


## Outputs

### Cluster visualization

![image](https://user-images.githubusercontent.com/66411147/137884030-9ee6f83d-b440-485e-bfa1-92d58392adda.png)

clusters were established both on epigenetic and phenotipic dimensions
Such outputs are generated for each timepoints

### Cell population identification

![image](https://user-images.githubusercontent.com/66411147/137884190-304e7faf-ea32-435c-9764-05def11cf8c6.png)

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





