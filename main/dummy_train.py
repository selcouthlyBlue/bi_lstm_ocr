from main.IAMDatasetPreparator import IAMDatasetPreparator
from main.TFStackedBidirectionalLstmNetwork import TensorflowNetwork
from test_files.configs.dummy_dataset_config import DatasetConfig
from test_files.configs.dummy_network_config import NetworkConfig
from test_files.configs.dummy_train_config import TrainConfig

network_config = NetworkConfig()
network = TensorflowNetwork(network_config)
dataset_config = DatasetConfig()
image_paths, labels = IAMDatasetPreparator.get_dataset_from(dataset_config)
train_data, train_labels, val_data, val_labels, _, _ = \
            IAMDatasetPreparator.split_into_train_validation_and_test_sets(image_paths, 0.5, 0.5, labels)
train_config = TrainConfig()
network.train(train_config, train_data, val_data, train_labels, val_labels)

