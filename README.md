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


Such outputs are generated for each cluster


# Installation and usage
## installation
1. Download miniconda:   
Windows 64-bit --> https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe  
Windows 32-bit --> https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86.exe
2. Download this repository
3. Install the dependencies:  
In the miniconda prompt, type: **pip install -r *path*/requirements.txt**  
(replace ***path*** by the path to the folder where you downloaded the repository)
## usage
1. Store your .fcs files in a folder named **files** inside the main folder
2. The files' names must contain some tags:  
BL for baseline timepoint  
DXX for other timepoints (replace XX by the number of days)  
The animal tag  
file name example: *D28_CDF059_pet_et_répète_sont_dans_un_bateau.fcs*
4. Modify the script:
![image](https://user-images.githubusercontent.com/66411147/144062796-da6078d3-69cc-4d09-8869-6c12a55b0d6d.png)  
  
**channels_to_drop_**: The channels that do not correspond to any markers (e.g., beads)  
  
**panel_**: your marker panel /!\ MUST BE ORDERED AS IN THE FCS FILE  
  
**markers_to_drop_**: markers you do not wish to use for analysis
  
**animals**: the animal tags you provided in the filenames  
  
**cells**: downsample size for each timepoint  
  
**neighbor**: UMAP parameter  
  
**metric**: UMAP parameter  
  
**min_sample**: HDBSCAN parameter  
  
**min_size**: HDBSCAN parameter  











