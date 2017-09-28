from abc import ABC, abstractmethod

class Network(ABC):
    def __init__(self, network_config):
        self.network_config = network_config
        super(Network, self).__init__()

    @abstractmethod
    def train(self, train_config, train_features, validation_features, train_labels, validation_labels):
        raise NotImplementedError

    @abstractmethod
    def test(self, test_features, test_labels):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_to_be_predicted):
        raise NotImplementedError
