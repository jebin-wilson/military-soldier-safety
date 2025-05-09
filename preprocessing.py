import cv2
import os
import numpy as np

def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

def preprocess_images(folder_path, output_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(folder_path, filename))
            resized = resize_image(img)
            cv2.imwrite(os.path.join(output_path, filename), resized)
