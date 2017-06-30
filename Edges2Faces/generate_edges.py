import cv2
import numpy as np
import os

dataset_path_edges = "faces_edges"
dataset_path_normal = "faces_resized"


print(os.listdir(dataset_path_normal))

for file in os.listdir(dataset_path_normal):
    img = cv2.imread(os.path.join(dataset_path_normal, file), 1)
    edges = cv2.Canny(img, 80, 150)

    edges = (255 - edges)

    save_path_edges = os.path.join(dataset_path_edges, file)

    cv2.imwrite(save_path_edges, edges)
