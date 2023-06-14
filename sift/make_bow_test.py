import numpy as np
import os
import pickle

from scipy.cluster.vq import vq

def make_bow(dataset, clusters, tfidf):
    print("Make bow vector for each frame")
    n_frames = 0
    for video in dataset:
        n_frames += len(video["features"])

    bow = np.zeros((n_frames, clusters.shape[0]), dtype=float)

    frame_index = 0
    for video in dataset:
        for frame in video["features"]:
            visual_word_ids = vq(frame, clusters)[0]
            for word_id in visual_word_ids:
                bow[frame_index, word_id] += 1
            frame_index += 1

    if tfidf:
        print("Applying TF-IDF weighting")
        freq = np.sum((bow > 0) * 1, axis = 0)
        idf = np.log((n_frames + 1) / (freq + 1))
        bow = bow * idf

    frame_index = 0
    for i in range(len(dataset)):
        features = []
        for frame in dataset[i]["features"]:
            features.append(bow[frame_index])
            frame_index += 1

        dataset[i]["features"] = features

        if (i + 1) % 50 == 0:
            print("Processed %d/%d videos" % (i + 1, len(dataset)))

    return dataset

if __name__ == "__main__":

    tfidf=1

    codebook = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_sift_bow_svm_individual_frame/data1/cbt_75clusters.p", "rb"))
    clusters = codebook.cluster_centers_
    print("No of clusters : ",len(clusters))

    dataset = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_sift_bow_svm_individual_frame/data1/test.p", "rb"))

    dataset_bow = make_bow(dataset, clusters, tfidf)

    pickle.dump(dataset_bow, open("data1/test1_bow_75.p", "wb"))

