import numpy as np
import os
import pickle

from scipy.cluster.vq import vq

def make_bow(dataset, clusters, tfidf):
    print("Make bow vector for each frame")

    n_videos = len(dataset)

    bow = np.zeros((n_videos, clusters.shape[0]), dtype=float)

    video_index = 0
    for video in dataset:
        visual_word_ids = vq(video["features"], clusters)[0]
        for word_id in visual_word_ids:
            bow[video_index, word_id] += 1
        video_index += 1

    if tfidf:
        print("Applying TF-IDF weighting")
        freq = np.sum((bow > 0) * 1, axis = 0)
        idf = np.log((n_videos + 1) / (freq + 1))
        bow = bow * idf

    video_index = 0
    for i in range(len(dataset)):

        dataset[i]["features"] = bow[video_index]
        video_index += 1

        if (i + 1) % 50 == 0:
            print("Processed %d/%d videos" % (i + 1, len(dataset)))

    return dataset

if __name__ == "__main__":
    
    codebook = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_optflow_bow_svm/data/cb_600clusters.p", "rb"))
    clusters = codebook.cluster_centers_

    dataset = pickle.load(open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_optflow_bow_svm/data/test.p", "rb"))
    tfidf=1

    dataset_bow = make_bow(dataset, clusters, tfidf)

    pickle.dump(dataset_bow, open("E:/PROJECT@12A/code/KTH-Action-Recognition-master/baseline_optflow_bow_svm/data/test_bow_c600.p", "wb"))

