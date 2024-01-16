import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('CreditCardClients.csv')
#df.dropna()  # Drop NA
df = df[df['Y'] == 0]
print(df)
X = df.values
plt.figure(figsize=(5, 4));
plt.scatter(X[:, 2], X[:, 5], s=150)  # 6th and 7th column
plt.xlabel('x1');
plt.ylabel('x2');
plt.title('Unclustered data points')
plt.show()


def k_means_clustering(X, k):
    centroids = {}
    max_iter = 5
    for i in range(k):
        centroids[i] = X[i]
    for i in range(max_iter):
        classifications = {}
        for i in range(k):
            classifications[i] = []
        for featureset in X:
            distances = [np.linalg.norm(featureset - centroids[centroid])
                         for centroid in centroids]
            classification = distances.index(min(distances))
            classifications[classification].append(featureset)
        # prev_centroids = dict(centroids)
        for classification in classifications:
            centroids[classification] = \
                np.average(classifications[classification], axis=0)
    return centroids, classifications


def plot_centroids_cluster(centroids, classifications, K):
    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1],
                    marker="o", color="k", s=100, linewidths=5)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for classification in classifications:
        color = colors[classification]
        for featureset in classifications[classification]:
            plt.scatter(featureset[0], featureset[1],
                        color=color, s=100, linewidths=5)
    plt.xlabel('x1');
    plt.ylabel('x2');
    plt.title('Clustering, K = ' + str(K))
    plt.show()


K = 3
ctrds, clf = k_means_clustering(X, K)
plt.figure(figsize=(5, 4))
plot_centroids_cluster(ctrds, clf, K)

WCSS_array = np.array([])
for K in range(1, 7):
    centroids, classifications = k_means_clustering(X, K)
    plt.figure(figsize=(5, 4))
    plot_centroids_cluster(centroids, classifications, K)
    wcss = 0
    for k in range(K):
        distances = [np.linalg.norm(m - centroids[k])
                     for m in classifications[k]]
        wcss += np.sum(distances)
    WCSS_array = np.append(WCSS_array, wcss)

K_array = np.arange(1, 7, 1)
plt.figure(figsize=(5, 4));
plt.plot(K_array, WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Elbow method to determine optimum number of clusters')
plt.show()

sampleDataCluster = pd.DataFrame()




