## Result

Whole repo is at [MBP-BrainIAC](https://github.com/TyBruceChen/MBP-BrainIAC). These are just results analysis.

To generate the optical flow visualization, notebook needs either to modify the path of optical_flow.py file or clone from the whole repo mentioned above. 

Files:

- features.csv: raw features extracted from BrainIAC ViT
- features_pca.csv: merged features from PCA (PC1 - PC10)
- pca_lr_knn_kmeans_opticalflow.ipynb: LR, KNN, K-means used to fitting / clustering the data, and generate optical flow visualization
- output/*html: optical flow visualization compared different K-means centered points between different clusters
