from random import sample
from math import sqrt
import numpy as np
import copy
import random
from plot import plot_silhouette, plot_distance
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from preprocessing import preprocess
from sklearn.mixture import GaussianMixture


class GaussMix:
    def __init__(self, n):
        self.gauss = GaussianMixture(n_components=n, random_state=42)
        self.labels_ = None

    def fit_predict(self, data):
        self.gauss.fit(data)
        labels = self.gauss.predict(data)
        self.labels_ = labels
        return labels
    

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters  
        self.max_iter = max_iter  
        self.tol = tol                
        self.centroids = None         
        self.labels = None            

    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        self.predict(X)


class DBSCAN:
    def __init__(self, eps=2, min_samples=5):
        self.X = None
        self.eps = eps
        self.min_samples = min_samples
        self.neighbor_list = []  # [set()]
        self.omega_set = set()  # 核心点
        self.gama = set()  # 未访问的点
        self.labels=[]

    def find_neighbor(self, p):
        N = list() 
        temp = np.sum((self.X-self.X[p])**2, axis=1)**0.5 
        N = np.argwhere(temp <= self.eps).flatten().tolist() 
        return set(N)

    def fit_predict(self, data):
        self.X = data
        k = -1

        for xi in range(len(self.X)):
            self.gama.add(xi)
            self.labels.append(-1)
            self.neighbor_list.append(self.find_neighbor(xi))
            if(len(self.neighbor_list[-1])>=self.min_samples):
                self.omega_set.add(xi)

        while(len(self.omega_set)>0):
            gama_copy = copy.deepcopy(self.gama)
            p = random.choice(list(self.omega_set))
            k += 1
            C = []
            C.append(p)
            self.gama.remove(p)
            while(len(C)>0):
                c = C[0]
                C.remove(c)
                if(len(self.neighbor_list[c])>self.min_samples):
                    delta = self.neighbor_list[c] & self.gama
                    deltalist = list(delta)
                    for i in range(len(delta)):
                        C.append(deltalist[i])
                        self.gama = self.gama - delta
            Ck = gama_copy - self.gama
            for i in range(len(Ck)):
                self.labels[list(Ck)[i]] = k
            self.omega_set = self.omega_set - Ck
        return self.labels

def best_k_clusters(kmax, data):
    sil = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(k)
        kmeans.fit(data)
        sil.append(silhouette_score(data, kmeans.predict(data)))
    plot_silhouette(sil, "k_means")

def best_n_components(nmax, data):
    sil = []
    for n in range(2, nmax + 1):
        gaussmix = GaussMix(n)
        sil.append(silhouette_score(data, gaussmix.fit_predict(data)))
    plot_silhouette(sil, "gauss_mix")

def best_dbscan_eps(data):
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, -1])
    plot_distance(distances)

if __name__ == "__main__":
    data = preprocess("data.txt")
    best_k_clusters(kmax=3, data=data)
    best_n_components(nmax=3, data=data)
    best_dbscan_eps(data)