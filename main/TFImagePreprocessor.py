from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from main.interfaces.image_preprocessor import ImagePreprocessor


class TFImagePreprocessor(ImagePreprocessor):
    @staticmethod
    def resize(images, shape):
        shape = (len(images), shape[0], shape[1])
        return np.resize(images, shape)

    @staticmethod
    def read(image_paths):
        number_of_images = len(image_paths)
        images = []
        filename_queue = tf.train.string_input_producer(image_paths)
        image_reader = tf.WholeFileReader()
        _, image_files = image_reader.read(filename_queue)
        image_files = tf.image.decode_png(image_files)
        image_files = tf.image.resize_images(image_files, (1024, 128))
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(number_of_images):
                if i % 100 == 0:
                    print("Read images: {}/{}".format(i, number_of_images))
                images.append(image_files.eval())
            coord.request_stop()
            coord.join(threads)
        return images
