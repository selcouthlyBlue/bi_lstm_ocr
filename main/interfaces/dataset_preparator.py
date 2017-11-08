from abc import ABC, abstractmethod

class DatasetPreparator(ABC):
    @staticmethod
    @abstractmethod
    def get_dataset_from(dataset_config):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def split_into_train_validation_and_test_sets(features, validation_size, test_size, labels=None):
        raise NotImplementedError
