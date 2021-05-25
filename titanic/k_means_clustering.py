import math
import random

class KMeans:
    def __init__(self, initial_clusters, data):
        self.prev_iteration_clusters = {}
        self.clusters = initial_clusters
        self.data = data
        self.cols = list(range(len(self.data[0])))

    def run(self):
        self.centroids = self.get_centroids()
        iteration = 0
        while self.prev_iteration_clusters != self.clusters and iteration < len(self.data):
            new_clusters = {key: [] for key, _ in self.clusters.items()}
            self.centroids = self.get_centroids()
            for row in self.data:
                distances = []
                for center in self.centroids.values():
                    distances.append(math.sqrt(sum([(center_coord - element) ** 2 for element, center_coord in zip(row, center)])))
                new_clusters[distances.index(min(distances)) + 1].append(self.data.index(row))
            prev_iteration_clusters = {key: value for key, value in self.clusters.items()}
            self.clusters = {key: value for key, value in new_clusters.items()}
            self.prev_iteration_clusters = {key: value for key, value in prev_iteration_clusters.items()}
            self.current_error = self.error()
            del prev_iteration_clusters
            iteration += 1

    def get_centroids(self):
        centroids = {}
        for k, indices in self.clusters.items():
            if indices != []:
                current_data = [self.data[index] for index in indices]
                transposed_data = [[row[i] for row in current_data] for i in self.cols]
                centroids[k] = [sum(transposed_data[i]) / len(transposed_data[i]) for i in self.cols]
            else:
                centroids[k] = self.centroids[k]
        return centroids

    def error(self):
        total_error = 0
        centroids = self.get_centroids()
        for k, indices in self.clusters.items():
            current_data = [self.data[index] for index in indices]
            for row in current_data:
                total_error += sum([(center_coord - element) ** 2 for element, center_coord in zip(row, centroids[k])])
        return total_error