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


## Installation and usage
### installation
1. Download miniconda:   
Windows 64-bit --> https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe  
Windows 32-bit --> https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86.exe
2. Download this repository
3. Install the dependencies:  
In the miniconda prompt, type: **pip install -r *path*/requirements.txt**  
(replace ***path*** by the path to the folder where you downloaded the repository)
### usage
1. Store your .fcs files in a folder named **files** inside the main folder
2. The files' names must contain some tags:  
BL for baseline timepoint  
DXX for other timepoints (replace XX by the number of days)  
The animal names 
4. Modify the script:
![image](https://user-images.githubusercontent.com/66411147/144062796-da6078d3-69cc-4d09-8869-6c12a55b0d6d.png)  
  
**channels_to_drop_**: The channels that do not correspond to any markers (e.g., beads)  
  
**panel_**: your marker panel /!\ MUST BE ORDERED AS IN THE FCS FILE  
  
**markers_to_drop_**: markers you do not wish to use for analysis
  
**animals**: the animal tags you provided in the filenames  
  
**cells**: the number of cells you wish to analyze per timepoint  
  
**neighbor**: UMAP parameter  
  
**metric**: UMAP parameter  
  
**min_sample**: HDBSCAN parameter  
  
**min_size**: HDBSCAN parameter  











