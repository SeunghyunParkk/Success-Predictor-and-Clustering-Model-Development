

import numpy as np
import pandas as pd

df = pd.read_csv("cereals.CSV")

#Task 0
col_name = ['Name', 'Calories', 'Protein', 'Fat', 'Fiber','Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins']
df_new = df.dropna(subset=col_name)

#Task 1
X = df_new[col_name[1:]]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='euclidean')
cluster.fit_predict(X_std)

labels, counts = np.unique(cluster.labels_, return_counts=True)

for label, count in zip(labels, counts):
    print(f"Cluster {label} has {count} cereals.")

#Task 2
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=815)
model = kmeans.fit(X_std)
labels = model.predict(X_std)
kmean_cluster = pd.DataFrame(list(zip(df_new['Name'],np.transpose(labels))), columns = ['Name','Cluster label'])
cluster_counts = kmean_cluster['Cluster label'].value_counts()
print(cluster_counts[0])
print(cluster_counts[1])

#Task 3
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=col_name[1:])
centroids_df = centroids_df.T
centroids_df.columns = ['Cluster 0', 'Cluster 1']
print(centroids_df)