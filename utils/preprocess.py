# utils/preprocess.py

import cv2
import numpy as np

def preprocess_eye_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Eye image not found.")
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_brain_image(image_path, target_size=(224, 224), grayscale=False):
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        raise ValueError("Brain image not found.")

    image = cv2.resize(image, target_size)

    if grayscale:
        image = np.expand_dims(image, axis=-1)

    image = image / 255.0
    return np.expand_dims(image, axis=0)