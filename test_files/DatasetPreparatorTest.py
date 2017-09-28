from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from main.IAMDatasetPreparator import IAMDatasetPreparator
from test_files.configs.dummy_dataset_config import DatasetConfig

class DatasetPreparatorTest(unittest.TestCase):
    def setUp(self):
        super(DatasetPreparatorTest, self).setUp()
        self.dataset_config = DatasetConfig()

    def test_get_image_paths_and_labels_from_dataset_config(self):
        images, labels = IAMDatasetPreparator.get_image_paths_and_labels_from(self.dataset_config)
        assert len(images) == len(labels) == 3

    def test_split_into_train_validation_and_test_sets(self):
        images, labels = IAMDatasetPreparator.get_image_paths_and_labels_from(self.dataset_config)
        train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            IAMDatasetPreparator.split_into_train_validation_and_test_sets(images, 0.5, 0.5, labels)
        assert len(train_data) == len(train_labels) == len(val_data) == len(val_labels) == \
               len(test_data) == len(test_labels) == 1


if __name__ == '__main__':
    unittest.main()
