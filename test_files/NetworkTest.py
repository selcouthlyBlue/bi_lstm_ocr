import tensorflow as tf

from main.CV2ImagePreprocessor import CV2ImagePreprocessor
from main.IAMDatasetPreparator import IAMDatasetPreparator
from main.TFNetwork import TensorflowNetwork
from test_files.configs.dummy_dataset_config import DatasetConfig
from test_files.configs.dummy_network_config import NetworkConfig
from test_files.configs.dummy_train_config import TrainConfig


class NetworkTest(tf.test.TestCase):
    def setUp(self):
        network_config = NetworkConfig()
        self.network = TensorflowNetwork(network_config)
        dataset_config = DatasetConfig()
        image_paths, labels = IAMDatasetPreparator.get_image_paths_and_labels_from(dataset_config)
        images = CV2ImagePreprocessor.read(image_paths)
        images = CV2ImagePreprocessor.resize(images, (1024, 128))
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = \
            IAMDatasetPreparator.split_into_train_validation_and_test_sets(images, 0.5, 0.5, labels)

    def test_train_network(self):
        train_config = TrainConfig()
        self.network.train(train_config, train_features=self.train_data, validation_features=self.val_data,
                           train_labels=self.train_labels, validation_labels=self.val_labels)


if __name__ == '__main__':
    tf.test.main()
