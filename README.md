# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Choose the number of clusters (K): 
Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2)Initialize cluster centroids: 
Randomly select K data points from your dataset as the initial centroids of the clusters.

3)Assign data points to clusters: 
Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically  done using Euclidean distance, but other distance metrics can also be used.

4)Update cluster centroids: 
Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5)Repeat steps 3 and 4: 
Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6)Evaluate the clustering results: 
Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

7)Select the best clustering solution: 
If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Mohamed Anas O.I
RegisterNumber:  212223110028
*/

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()
```
### Output
![Screenshot 2024-11-10 192538](https://github.com/user-attachments/assets/4928905d-7ac5-4a92-8f13-953fc9f00f1a)

```
data.info()
```
### Output

![Screenshot 2024-11-10 192543](https://github.com/user-attachments/assets/d7842d1b-e473-434d-93bc-94154fe5b758)

```
data.isnull().sum()
```
### Output
![Screenshot 2024-11-10 192548](https://github.com/user-attachments/assets/f28f84fd-8d3a-46f6-8710-34c9a5d279d8)

```
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++")
  kmeans.fit(data.iloc[:, 3:])
  wcss.append(kmeans.inertia_)
  
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])
y_pred
```
### Output

![Screenshot 2024-11-10 192605](https://github.com/user-attachments/assets/0b0819c3-0db6-4d34-abe7-272462ecff7d)

![Screenshot 2024-11-10 192613](https://github.com/user-attachments/assets/19aa3b75-1bee-44f1-9cb9-87384ab6b229)

```
data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")
```
### Output

![Screenshot 2024-11-10 192621](https://github.com/user-attachments/assets/3ead5d85-17bd-4c50-956e-a0c88727a1a0)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
