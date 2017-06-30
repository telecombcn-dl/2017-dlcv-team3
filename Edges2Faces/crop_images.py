import cv2
import numpy as np
import os

dataset_path = "lfw-deepfunneled"
dataset_path_normal = "lfw-deepfunneled_cropped"

os.chdir(dataset_path)

folders = [name for name in os.listdir(".") if os.path.isdir(name)]

print("Found " + str(len(folders)) + " folders.")

i = 1
for folder in folders:
    # Search every folder of the dataset
    path = os.path.join(dataset_path, folder)

    # Get every image of the dataset process it
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file), 1)
        crop_img = img[50:200, 50:200]

        ima_name = str(i) + ".jpg"
        save_path_normal = os.path.join(dataset_path_normal, ima_name)

        cv2.imwrite(save_path_normal, crop_img)
        i = i + 1
