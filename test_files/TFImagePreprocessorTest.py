import unittest

from main.TFImagePreprocessor import TFImagePreprocessor


class TFImagePreprocessorTest(unittest.TestCase):
    def setUp(self):
        dummy_data_root_dir = "C:/Users/asus.11/Documents/bi_lstm_ocr/test_files/dummy_data/"
        self.dummy_file_list = [dummy_data_root_dir + "dummy_image1.png", dummy_data_root_dir + "dummy_image2.png",
                                dummy_data_root_dir + "dummy_image3.png"]

    def test_read(self):
        read_images = TFImagePreprocessor.read(self.dummy_file_list)
        assert len(read_images) == 3

    def test_resize(self):
        read_images = TFImagePreprocessor.read(self.dummy_file_list)
        shape_of_each_image = (1024, 128)
        resized_images = TFImagePreprocessor.resize(read_images, shape_of_each_image)
        self.assertEqual(resized_images[0].shape, shape_of_each_image)
        self.assertEqual(len(resized_images), len(read_images))


if __name__ == '__main__':
    unittest.main()
