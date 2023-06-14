import numpy as np
import os
import pickle

from sklearn.cluster import KMeans
from numpy import size

if __name__ == "__main__":

    print("Loading dataset")
    train_features = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_optflow_bow_svm/data/train_keypoints.p", "rb"))
    train=np.vstack(train_features)
    n_features = len(train_features)

    print("Number of feature points to run clustering on: %d" % n_features)

    clusters=500
    print("No of clusterss : ",clusters)
    print("Running KMeans clustering")
    kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10,verbose=1)
    kmeans.fit(train)
    pickle.dump(kmeans, open("data/cb_%dclusters.p" % clusters, "wb"))
