from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import numpy as np
import os
import logging

from main.DataIterator import DataIterator
from main.EncoderDecoder import EncoderDecoder
from main.interfaces.network import Network


def load_charset(charset_file):
    return ''.join([line.rstrip('\n') for line in open(charset_file)])


class TensorflowNetwork(Network):
    def __init__(self, network_config):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, None, network_config.num_features], name="input")
            self.labels = tf.sparse_placeholder(tf.int32, name="label")
            self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len_input")

            logits = self._bidirectional_lstm_layers(
                network_config.num_hidden_units,
                network_config.num_layers,
                network_config.num_classes
            )

            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)

            self.optimizer = tf.train.AdamOptimizer(network_config.learning_rate).minimize(self.cost)
            self.decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=self.seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_to_dense(tf.to_int32(self.decoded[0].indices),
                                                           tf.to_int32(self.decoded[0].dense_shape),
                                                           tf.to_int32(self.decoded[0].values),
                                                           name="output")
            self.label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            self.merged_summary = tf.summary.merge_all()
        super(TensorflowNetwork, self).__init__(network_config)

    def train(self, train_config, train_features, validation_features, train_labels=None, validation_labels=None):
        logger = logging.getLogger('Training for ocr using BI_LSTM_CTC')
        logger.setLevel(logging.INFO)

        encoder_decoder = EncoderDecoder()
        encoder_decoder.initialize_encode_and_decode_maps_from(load_charset(train_config.charset_file))

        encoded_train_labels = []
        encoded_val_labels = []
        for train_label in train_labels:
            encoded_train_labels.append(encoder_decoder.encode(train_label))
        for val_label in validation_labels:
            encoded_val_labels.append(encoder_decoder.encode(val_label))
        train_labels = encoded_train_labels
        validation_labels = encoded_val_labels

        print('loading train data, please wait---------------------', end=' ')
        train_feeder = DataIterator(train_features, train_labels, train_config.batch_size)
        print('number of training images: ', train_feeder.get_number_of_examples())

        print('loading validation data, please wait---------------------', end=' ')
        val_feeder = DataIterator(validation_features, validation_labels, train_config.batch_size)
        print('number of validation images: ', val_feeder.get_number_of_examples())

        num_train_samples = train_feeder.get_number_of_examples()
        num_batches_per_epoch = int(num_train_samples/train_config.batch_size)

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)
        with tf.Session(graph=self.graph, config=config) as sess:
            sess.run(tf.global_variables_initializer())

            log_dir = train_config.log_dir + 'train'

            saver = tf.train.Saver(max_to_keep=5)

            train_writer = tf.summary.FileWriter(log_dir, sess.graph)
            if train_config.is_restore:
                ckpt = tf.train.latest_checkpoint(train_config.checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))

            print('=============================begin training=============================')
            val_inputs, val_seq_len, val_labels = val_feeder.get_whole_data()
            val_feed = {
                self.inputs: val_inputs,
                self.labels: val_labels,
                self.seq_len: val_seq_len
            }

            for current_epoch in range(train_config.num_epochs):
                shuffle_index = np.random.permutation(num_train_samples)
                train_cost = 0
                start = time.time()

                for current_batch_number in range(num_batches_per_epoch):
                    train_feed = self._get_batch_feed(current_batch_number, shuffle_index, train_feeder)

                    summary_str, batch_cost, step, _ = sess.run([self.merged_summary, self.cost, self.global_step, self.optimizer], train_feed)
                    train_cost += batch_cost * train_config.batch_size
                    train_writer.add_summary(summary_str, step)

                    if not os.path.isdir(train_config.checkpoint_dir):
                        os.mkdir(train_config.checkpoint_dir)
                    logger.info('save the checkpoint of {}'.format(step))

                    tf.train.write_graph(sess.graph_def, log_dir, 'bi_lstm_ctc_ocr2.pbtxt')
                    saver.save(sess, os.path.join(train_config.checkpoint_dir, 'ocr-model-{}.ckpt'.format(current_epoch)), global_step=step)

                    if step % train_config.validation_steps == 0:
                        dense_decoded, last_batch_err = sess.run([self.dense_decoded, self.label_error_rate], val_feed)
                        avg_train_cost = train_cost/((current_batch_number + 1) * train_config.batch_size)
                        print("Batch {}/{}, Epoch {}/{}, avg_train_cost: {:.3f}, last_batch_err: {:.3f}, time: {:.3f}"
                              .format(current_batch_number, num_batches_per_epoch, current_epoch + 1, train_config.num_epochs, avg_train_cost, last_batch_err, time.time() - start))

    def _get_batch_feed(self, current_batch_number, shuffle_index, train_feeder):
        batch_inputs, batch_seq_len, batch_labels = train_feeder.get_next_batch(current_batch_number,
                                                                                                  shuffle_index)
        feed = {
            self.inputs: batch_inputs,
            self.labels: batch_labels,
            self.seq_len: batch_seq_len
        }
        return feed

    def test(self, test_features, test_labels=None):
        print("I'm testing!")

    def predict(self, data_to_be_predicted):
        print("I'm predicting!")

    def _bidirectional_lstm_layers(self, num_hidden, num_layers, num_classes):
        stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True) for _ in range(num_layers)], state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.seq_len, dtype=tf.float32)

        batch_size = tf.shape(self.inputs)[0]

        outputs = tf.reshape(outputs, [-1, num_hidden])

        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1, dtype=tf.float32))
        b = tf.Variable(tf.constant(0., shape=[num_classes], dtype=tf.float32))

        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_size, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))

        return logits
