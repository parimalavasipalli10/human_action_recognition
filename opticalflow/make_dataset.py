import numpy as np
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

TRAIN_PEOPLE_ID = [1,4,11, 12, 13, 14, 15, 16, 17, 18,19,20,21,23,24,25]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

if __name__ == "__main__":

    train = []
    test = []

    keypoints=[]
    train_keypoints = []

    for category in CATEGORIES:
        print("Processing category %s" % category)
        category_features = pickle.load(open("data/optflow_%s.p" % category, "rb"))

        for video in category_features:
            person_id = int(video["filename"].split("_")[0][6:])

            for frame in video["features"]:
                keypoints.append(frame)

            if person_id in TRAIN_PEOPLE_ID:
                train.append(video)

                for frame in video["features"]:
                    train_keypoints.append(frame)
                    
            else:
                test.append(video)

    print("Total number of keypoints : ",len(keypoints))
    print("Number of training keypoints : ",len(train_keypoints))
    print("Saving train/test set to files")
    pickle.dump(train, open("data/train.p", "wb"))
    pickle.dump(test, open("data/test.p", "wb"))

    print("Saving keypoints in training set")
    pickle.dump(train_keypoints, open("data/train_keypoints.p", "wb"))

