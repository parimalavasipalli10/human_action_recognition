import cv2
import numpy as np
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    farneback_params = dict(winsize = 20, iterations=1,flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)

    n_processed_files = 0

    for category in CATEGORIES:
        print("Processing category %s" % category)

        folder_path = os.path.join("..", "E:\PROJECT@12A\DataSet\kth_dataset", category)
        filenames = os.listdir(folder_path)
        features = []
        frame_count=0

        for filename in filenames:
            filepath = os.path.join("..", "E:\PROJECT@12A\DataSet\kth_dataset", category, filename)
            vid = cv2.VideoCapture(filepath)
            features_current_file = []

            prev_frame = None

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_count+=1

                if prev_frame is not None:
                    flows = cv2.calcOpticalFlowFarneback(prev_frame, frame,**farneback_params)

                    feature = []
                    for r in range(120):
                        if(r % 10 != 0):
                            continue
                        for c in range(160):
                            if(c % 10 != 0):
                                continue
                            feature.append(flows[r,c,0])
                            feature.append(flows[r,c,1])
                    feature = np.array(feature)
                    features_current_file.append(feature)

                prev_frame = frame

            features.append({
                "filename": filename,
                "category": category,
                "features": features_current_file 
            })

        print("Number of frames : ",frame_count)

        pickle.dump(features, open("data/optflow_%s.p" % category, "wb"))

