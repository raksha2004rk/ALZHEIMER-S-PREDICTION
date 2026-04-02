import os
import cv2
import numpy as np
from src.exception import CustomException
import sys


class DataPreprocessing:
    def __init__(self):
        self.image_size = (128, 128)

    def preprocess_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.image_size)
            image = image / 255.0
            return image

        except Exception as e:
            raise CustomException(e, sys)