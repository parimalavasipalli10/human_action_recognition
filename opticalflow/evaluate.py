import numpy as np
import os
import pickle

from sklearn.svm import SVC

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running","walking"]

if __name__ == "__main__":

    data = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_optflow_bow_svm/data/test_bow_c600.p", "rb"))

    clf = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_optflow_bow_svm/data/svm_C1_c600.p", "rb"))

    confusion_matrix = np.zeros((6, 6))

    correct = 0
    for video in data:

        predicted = clf.predict([video["features"]])

        if predicted == video["category"]:
            correct += 1

        confusion_matrix[CATEGORIES.index(predicted),
            CATEGORIES.index(video["category"])] += 1

    print("%d/%d Correct" % (correct, len(data)))
    print("Accuracy =", correct / len(data))
    
    print("Confusion matrix")
    print(confusion_matrix)
