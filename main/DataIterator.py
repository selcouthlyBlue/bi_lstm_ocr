import numpy as np

from main.CV2ImagePreprocessor import CV2ImagePreprocessor


class DataIterator:
    def __init__(self, features, labels, batch_size=1):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size

    def get_number_of_examples(self):
        return len(self.labels)

    def the_label(self, indices):
        labels=[]
        for index in indices:
            labels.append(self.labels[index])
        return labels

    def get_next_batch(self, current_batch_number, shuffle_index):
        indices = [shuffle_index[i % self.get_number_of_examples()] for i in range(
            current_batch_number * self.batch_size, (current_batch_number + 1) * self.batch_size
        )]

        batch_inputs = [self.features[i] for i in indices]
        batch_inputs = CV2ImagePreprocessor.read(batch_inputs)
        batch_inputs = CV2ImagePreprocessor.resize(batch_inputs, (1024, 128))
        batch_labels = [self.labels[i] for i in indices]

        batch_seq_len = self._get_input_lens(np.array(batch_inputs))
        batch_labels = self.sparse_tuple_from_label(batch_labels)
        return batch_inputs,batch_seq_len,batch_labels

    def _get_input_lens(self, sequences):
        lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
        return lengths

    def sparse_tuple_from_label(self, sequences, dtype=np.int32):
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    def get_whole_data(self):
        images = CV2ImagePreprocessor.read(self.features)
        images = CV2ImagePreprocessor.resize(images, (1024, 128))
        return images, self._get_input_lens(np.array(self.features)), self.sparse_tuple_from_label(self.labels)
