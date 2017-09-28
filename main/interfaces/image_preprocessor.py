from abc import ABC, abstractmethod

class ImagePreprocessor(ABC):
    def __init__(self):
        super(ImagePreprocessor, self).__init__()

    @staticmethod
    @abstractmethod
    def resize(images, shape):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def read(image_paths):
        raise NotImplementedError
