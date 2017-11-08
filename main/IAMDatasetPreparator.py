from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import train_test_split

from main.interfaces.dataset_preparator import DatasetPreparator


class IAMDatasetPreparator(DatasetPreparator):
    @staticmethod
    def get_dataset_from(iam_dataset_config):
        image_paths = []
        labels = []
        with open(iam_dataset_config.labels_file) as f:
            labeled_data = f.readlines()
        labeled_data = [x.strip() for x in labeled_data]
        for example in labeled_data:
            example_data = example.split()
            if example_data[1] == "ok":
                image_path = iam_dataset_config.data_dir + example_data[0] + ".png"
                image_paths.append(image_path)
                label = example_data[-1]
                labels.append(label)
        return image_paths, labels

    @staticmethod
    def split_into_train_validation_and_test_sets(images, validation_size, test_size, labels=None):
        train_images, test_images, train_labels, test_labels = train_test_split(
            images,
            labels,
            test_size=validation_size,
            random_state=128
        )
        validation_images, test_images, validation_labels, test_labels = train_test_split(
            test_images,
            test_labels,
            test_size=test_size,
            random_state=128
        )
        return train_images, train_labels, validation_images, validation_labels, test_images, test_labels
