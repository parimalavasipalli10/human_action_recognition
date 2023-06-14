import numpy as np
import os
import pickle
from sklearn.svm import SVC


if __name__ == "__main__":
    C = 1

    print("Loading training set...")
    dataset = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_sift_bow_svm_individual_frame/data1/train1_bow_75.p", "rb"))

    print(len(dataset))
    
    X = []
    Y = []

    for video in dataset:
        for frame in video["features"]:
            X.append(frame)
            Y.append(video["category"])

    print("Length of X : ",len(X))
    print("Length of Y : ",len(Y))
    
    print("Training Linear SVM.....")
    clf = SVC(C=C, kernel="linear", verbose=True)
    print("Fitting..")
    clf.fit(X, Y)

    print("Saving into file....")
    pickle.dump(clf, open("data1/svm1_75.p", "wb"))
