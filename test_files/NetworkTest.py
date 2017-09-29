import shutil
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
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = \
            IAMDatasetPreparator.split_into_train_validation_and_test_sets(image_paths, 0.5, 0.5, labels)
        self.train_config = TrainConfig()

    def test_train_network(self):
        self.network.train(self.train_config, train_features=self.train_data, validation_features=self.val_data,
                           train_labels=self.train_labels, validation_labels=self.val_labels)

    def tearDown(self):
        shutil.rmtree(self.train_config.checkpoint_dir, ignore_errors=True)
        shutil.rmtree(self.train_config.log_dir, ignore_errors=True)


if __name__ == '__main__':
    tf.test.main()
