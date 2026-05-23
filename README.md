## Result

Home repo is at [MBP-BrainIAC](https://github.com/TyBruceChen/MBP-BrainIAC/tree/main). These are just results analysis.

Files:

- features.csv: raw features extracted from BrainIAC ViT
- features_pca.csv: merged features from PCA (PC1 - PC10)
- pca_lr_knn_kmeans_opticalflow.ipynb: LR, KNN, K-means used to fitting / clustering the data, and generate optical flow visualization
- output/*html: optical flow visualization compared different K-means centered points between different clusters