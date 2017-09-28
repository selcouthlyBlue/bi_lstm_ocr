from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import numpy as np

from main.EncoderDecoder import EncoderDecoder
from main.interfaces.network import Network


def load_charset(charset_file):
    return ''.join([line.rstrip('\n') for line in open(charset_file)])


class TensorflowNetwork(Network):
    def __init__(self, network_config):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, None, network_config.num_features])
            self.labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            logits = self.bidirectional_lstm_layers(network_config.num_hidden_units,
                                                    network_config.num_layers,
                                                    network_config.num_classes)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer(network_config.learning_rate).minimize(self.cost)
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len)
            self.label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))
        super(TensorflowNetwork, self).__init__(network_config)

    def train(self, train_config, train_features, validation_features, train_labels=None, validation_labels=None):
        encoder_decoder = EncoderDecoder()
        encoder_decoder.initialize_encode_and_decode_maps_from(load_charset(train_config.charset_file))
        num_examples = len(train_features)
        num_batches_per_epoch = int(num_examples/train_config.batch_size)
        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()

            for current_epoch in range(train_config.num_epochs):
                train_cost = train_label_error_rate = 0
                start = time.time()

                for batch in range(num_batches_per_epoch):
                    indices = np.asarray([i % num_examples
                                          for i in range(batch * train_config.batch_size,
                                                        (batch + 1) * train_config.batch_size)],
                                                        dtype=np.int64)

                    def get_input_lens(sequences):
                        lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
                        return sequences, lengths

                    batch_train_inputs = [train_features[i] for i in indices]
                    batch_train_inputs, batch_train_seq_len = get_input_lens(np.array(batch_train_inputs))
                    encoded_batch_train_labels = [encoder_decoder.encode(label) for label in train_labels]
                    batch_train_labels = self.__sparse_tuple_from(
                        [encoded_batch_train_labels[i] for i in indices]
                    )

                    print(len(batch_train_seq_len))

                    feed = {
                        self.inputs: batch_train_inputs,
                        self.labels: batch_train_labels,
                        self.seq_len: batch_train_seq_len
                    }

                    batch_cost, _ = sess.run([self.cost, self.optimizer], feed_dict=feed)
                    train_cost += batch_cost * train_config.batch_size
                    train_label_error_rate += sess.run(self.label_error_rate, feed_dict=feed) * train_config.batch_size

                shuffled_indices = np.random.permutation(num_examples)
                train_features = train_features[shuffled_indices]
                train_labels = train_labels[shuffled_indices]

                train_cost /= num_examples
                train_label_error_rate /= num_examples

                log ="Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
                print(log.format(current_epoch+1, train_config.num_epochs, train_cost, train_label_error_rate,
                                 time.time() - start))

    def test(self, test_features, test_labels=None):
        print("I'm testing!")

    def predict(self, data_to_be_predicted):
        print("I'm predicting!")

    def bidirectional_lstm_layers(self, num_hidden, num_layers, num_classes):
        lstm_fw_cells = [tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) for _ in range(num_layers)]
        lstm_bw_cells = [tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) for _ in range(num_layers)]

        try:
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cells, lstm_bw_cells,
                                                                           self.inputs, dtype=tf.float32)
        except Exception:
            outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cells, lstm_bw_cells,
                                                                     self.inputs, dtype=tf.float32)

        batch_size = tf.shape(self.inputs)[0]

        outputs = tf.reshape(outputs, [-1, num_hidden])

        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1, dtype=tf.float32))
        b = tf.Variable(tf.constant(0., shape=[num_classes], dtype=tf.float32))

        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_size, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))
        return logits

    @staticmethod
    def __sparse_tuple_from(labels, dtype=np.int32):
        indices = []
        values = []

        for n, seq in enumerate(labels):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(labels), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape
