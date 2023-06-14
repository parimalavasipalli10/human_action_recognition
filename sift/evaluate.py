import numpy as np
import os
import pickle

from sklearn.svm import SVC

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

if __name__ == "__main__":

    data = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_sift_bow_svm_individual_frame/data1/test1_bow_75.p", "rb"))

    clf = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_sift_bow_svm_individual_frame/data1/svm1_75.p", "rb"))

    confusion_matrix = np.zeros((6, 6))

    correct = 0
    for video in data:
        majority = {}
        for category in CATEGORIES:
            majority[category] = 0
        
        for frame in video["features"]:
            predicted = clf.predict([frame])
            majority[predicted[0]] += 1

        predicted = None
        max_vote = -1
        for category in CATEGORIES:
            if(majority[category] > max_vote):
                max_vote = majority[category]
                predicted = category

        if(predicted == video["category"]):
            correct += 1

        confusion_matrix[CATEGORIES.index(predicted),
            CATEGORIES.index(video["category"])] += 1

    print("%d/%d Correct" % (correct, len(data)))
    print("Accuracy =", correct / len(data))
    
    print("Confusion matrix")
    print(confusion_matrix)
