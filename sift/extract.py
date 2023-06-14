import cv2
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    sift = cv2.SIFT_create()

    n_processed_files = 0

    for category in CATEGORIES:
        print("Processing category %s" % category)

        folder_path = os.path.join("..", "E:\Quick Access\kth_dataset\kth_dataset", category)
        filenames = os.listdir(folder_path)

        features = []
        frame_count=0

        for filename in filenames:
            filepath = os.path.join("..", "E:\Quick Access\kth_dataset\kth_dataset", category, filename)
            vid = cv2.VideoCapture(filepath)

            features_current_file = []

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_count+=1
                kps,desc = sift.detectAndCompute(frame,None)

                if desc is not None:
                    features_current_file.append(desc)
                
            features.append({
                "filename": filename,
                "category": category,
                "features": features_current_file 
            })

        print("Number of frames : ",frame_count)

        pickle.dump(features, open("data1/sift_%s.p" % category, "wb"))

