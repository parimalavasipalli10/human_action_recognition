import numpy as np
import os
import pickle

from sklearn.svm import SVC

def make_dataset(dataset):
    X = []
    Y = []

    for video in dataset:
        X.append(video["features"])
        Y.append(video["category"])

    return X, Y

if __name__ == "__main__":

    dataset = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_optflow_bow_svm/data/train_bow_600.p", "rb"))
    X, Y = make_dataset(dataset)

    clf = SVC(C=1, kernel="linear", verbose=True)
    clf.fit(X, Y)
    pickle.dump(clf, open("data/svm_C1_c600.p", "wb"))
