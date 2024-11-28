import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

def plot_distance(distances):
    plt.plot(distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel("4th Nearest Neighbor Distance")
    plt.title("k-distance Graph")
    plt.savefig('fig/eps_dbscan.png')

def plot_silhouette(sil, name):
    fig, ax = plt.subplots()
    
    ax.set_xticks(range(len(sil)))
    ax.set_xticklabels(range(2, 2 + len(sil)))
    ax.plot(sil,'gx-')
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    plt.savefig(f'fig/{name}-silhouette.png')


def plot_clusters(X, labels, title):
    fig, ax = plt.subplots()
    distance = pairwise_distances(X, metric='cosine')
    pca = PCA(2)
    X_pca = pca.fit_transform(distance)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=2, c=labels)
    ax.set_title(title)
    plt.savefig(f'fig/{title}.png')