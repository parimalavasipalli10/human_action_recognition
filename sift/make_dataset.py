import numpy as np
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

TRAIN_PEOPLE_ID = [1,4,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

if __name__ == "__main__":

    dataset = []

    train_keypoints1 = []
    train_keypoints=[]
    #keypoints_count1 = {}
    keypoints_count={}
    train=[]
    test=[]

    for category in CATEGORIES:
        print("Processing category %s" % category)
        category_features = pickle.load(open("data1/sift_%s.p" % category, "rb"))
        keypoints_count[category] = 0

        for video in category_features:
            dataset.append(video)
            person_id = int(video["filename"].split("_")[0][6:])

            if(person_id in TRAIN_PEOPLE_ID):
                train.append(video)
                for frame in video["features"]:
                    #keypoints_count[category] += frame.shape[0]
                    for i in range(frame.shape[0]):
                        train_keypoints1.append(frame[i])
            else:
                test.append(video)
            for frame in video["features"]:
                keypoints_count[category] += frame.shape[0]
                for i in range(frame.shape[0]):
                    train_keypoints.append(frame[i])

    #print("Saving dataset to files")
    #pickle.dump(dataset, open("data1/dataset.p", "wb"))

    print("Length of training set : ",len(train))
    print("Length of testing set : ",len(test))
    print("Total number of keypoints : ",len(train_keypoints))
    print("Number of training keypoints : ",len(train_keypoints1))
    
    #print("Saving train/test set to files")
    #pickle.dump(train, open("data1/train.p", "wb"))
    #pickle.dump(test, open("data1/test.p", "wb"))

    #print("Saving keypoints in dataset")
    #pickle.dump(train_keypoints, open("data1/train_keypoints.p", "wb"))

    print("Total Number of SIFT keypoints for each category")
    for category in CATEGORIES:
        print("%s: %d" % (category,keypoints_count[category]))

    '''print("Number of SIFT keypoints for training in each category")
    for category in CATEGORIES:
        print("%s: %d" % (category,keypoints_count[category]))'''

    print("Saving keypoints in dataset")
    pickle.dump(train_keypoints1, open("data1/train_keypoints1.p", "wb"))

 
