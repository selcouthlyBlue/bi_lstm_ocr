from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

from main.interfaces.image_preprocessor import ImagePreprocessor


class CV2ImagePreprocessor(ImagePreprocessor):
    @staticmethod
    def resize(images, shape):
        resized_images = []
        for image in images:
            resized_images.append(cv2.resize(image, shape).swapaxes(0,1))
        return resized_images

    @staticmethod
    def read(image_paths):
        images = []
        for image_path in image_paths:
            images.append(cv2.imread(image_path, 0).astype(np.float32))
        return images
