K-Means Clustering From Scratch 

A complete implementation of the K-Means clustering algorithm using only NumPy, without scikit-learn’s KMeans.
This project includes full clustering logic, dataset generation, visualization, Elbow Method, Silhouette Analysis, and a final evaluation report.

Overview

This project focuses on implementing the K-Means unsupervised learning algorithm from scratch, with the purpose of understanding:

How centroids are initialized

How distances are calculated

How data points are reassigned during iterations

How convergence is reached

How to evaluate clustering quality using

Elbow Method (SSE)

Silhouette Score (NumPy custom implementation)

The entire workflow — from dataset generation to evaluation and visualization — is done in one Python file for simplicity and accuracy.

 Objectives

✔ Implement K-Means algorithm using NumPy (no ML libraries)
✔ Generate well-separated synthetic data (Gaussian blobs)
✔ Determine the optimal number of clusters using:

Elbow Method

Silhouette Analysis
✔ Visualize final clusters & centroids
✔ Provide summary, interpretation, and full analysisMethodology
1. Data Generation

We use sklearn.datasets.make_blobs() ONLY for dataset creation.
The dataset includes:

600 samples

2 numeric features

4 true cluster centers

Controlled separation

This produces clean data suitable for visual clustering.

2. K-Means Implementation (Pure NumPy)

The algorithm was implemented manually with the following components:

✔ Random Centroid Initialization

Random data points chosen as initial cluster centers.

✔ Distance Calculation

Euclidean distance (L2 norm) computed between every point and centroid.

✔ Cluster Assignment

Each sample is assigned to its nearest centroid.

✔ Centroid Update

New centroid = mean of all points assigned to that cluster.

✔ Convergence

Algorithm stops when centroid movement < tolerance threshold.

This ensures a fully working K-Means algorithm without relying on scikit-learn.

📈 3. Evaluation Metrics

To determine the best value of K, two independent methods were applied:

🔹 A. Elbow Method (Sum of Squared Errors — SSE)

SSE is computed for K = 1 to 10.
The “elbow point” is where reduction in error slows down sharply.

🟢 Optimal K from Elbow Method = 4

🔹 B. Silhouette Score (Custom NumPy Implementation)

For each value of K (2 to 10):

a(i) = Mean distance to points in same cluster

b(i) = Mean distance to closest neighboring cluster

Silhouette = (b − a) / max(a, b)

Higher silhouette score → better cluster quality.

🟢 Optimal K from Silhouette Score = 4

📌 4. Final Results

Both evaluation methods independently selected the same value:

Method	Chosen K
Elbow Method	4
Silhouette Score	4

A final K-Means model with K=4 was trained and visualized.
Clusters are clearly separated, and centroids are correctly positioned.
