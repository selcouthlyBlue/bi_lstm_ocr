import os
import tensorflow as tf

from main.utils import freeze
from main.utils import optimize_graph


class FreezeGraphTest(tf.test.TestCase):
    def setUp(self):
        self.frozen_graph_path="frozen_bi_lstm_ctc_ocr.pb"

    def test_freeze_graph(self):
        self.freeze()
        self.assertTrue(os.path.exists(self.frozen_graph_path))

    def test_optimize_graph(self):
        self.freeze()
        optimize_graph(self.frozen_graph_path, ["input"], ["output"])

    def freeze(self):
        freeze(
            input_graph_path="log/train/bi_lstm_ctc_ocr.pbtxt",
            checkpoint_path="checkpoint/ocr-model-0.ckpt-0",
            output_node_names="output",
            input_saver_def_path="",
            output_frozen_graph_name=self.frozen_graph_path,
            input_binary=False
        )


if __name__ == '__main__':
    tf.test.main()
