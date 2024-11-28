# from sklearn.cluster import DBSCAN
# from sklearn.cluster import KMeans
from models import KMeans, DBSCAN, GaussMix
from sklearn.metrics import silhouette_score
import numpy as np
from plot import plot_silhouette, plot_clusters
import time 
import argparse
from preprocessing import preprocess

kmax = 10


def train(data, args):
    kmeans = KMeans(args.clusters)
    labels = kmeans.fit_predict(data)
    plot_clusters(data, labels, 'kmeans')

    dbscan = DBSCAN(eps=args.eps, min_samples=args.num_samples) 
    labels = dbscan.fit_predict(data)
    plot_clusters(data, labels, 'DBSCAN')

    gaussmix = GaussMix(n=args.n_components)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cluster')
    parser.add_argument("--data-path", type=str, default="data.txt")
    parser.add_argument("--clusters", "-c", type=int, help="the clusters of kmens", required=True)
    parser.add_argument("--eps", "-e", type=float, help="the distance for DBSCAN cluster", required=True)
    parser.add_argument("--num-samples", "-n", type=int, help="min samples for a point in dbscan to be center", required=True)
    parser.add_argument("--n-components", "-nc", type=int, help="number of gaussian distribution in gauss mix", required=True)
    parser.add_argument('--noplot', action="store_true", help='plot')
    args = parser.parse_args()

    data = preprocess(args.data_path)
    train(data, args)
