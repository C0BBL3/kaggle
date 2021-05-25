import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as sklearnKmeans
from k_means_clustering import KMeans

dataframe = pd.read_csv(os.getcwd() + '\\titanic\\processed_dataset_of_knowns.csv')

min_max_dict = {}
for column in dataframe[["Sex", "Pclass", "Fare", "Age", "SibSp"]].columns:
    min_max_dict[column] = list(dataframe[column].apply(lambda element: (element -  min(dataframe[column])) / (max(dataframe[column]) -  min(dataframe[column]))))

min_max_dataframe = pd.DataFrame(min_max_dict)
data = list([list(row) for row in min_max_dataframe.values])

error = []
for k in range(1,26):
    initial_clusters = {}
    for i in range(1, k + 1):
        temp = []
        for index, _ in enumerate(data):
            if index % (i) == 0:
                temp.append(index)
        initial_clusters[i] = temp
    kmeans = KMeans(initial_clusters, data)
    kmeans.run()
    error.append(kmeans.error())

plt.plot(list(range(1,26)), error)

error = []
np_array = np.array(data)
for k in range(1,26):
    kmeans = sklearnKmeans(n_clusters=k, random_state=0).fit(np_array)
    total_error = 0
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    unique_lables = sorted(set(kmeans.labels_))
    clusters = {label + 1: [] for label in unique_lables}
    for index, label in enumerate(kmeans.labels_):
        clusters[label + 1].append(index)
    for k, indices in clusters.items():
        current_data = [data[index] for index in indices]
        for row in current_data:
            total_error += sum([(center_coord - element) ** 2 for element, center_coord in zip(row, centroids[k - 1])])
    error.append(total_error)
  
plt.plot(list(range(1,26)), error)

plt.legend(['mine', 'sklearns'])
plt.xlabel('k')
plt.ylabel('sum squared error')
plt.title('Elbow Method')
plt.savefig('129.png')