import unittest

from main.EncoderDecoder import EncoderDecoder


class EncoderDecoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder_decoder = EncoderDecoder()
        self.charset_string = "abc"

    def test_generate_encode_and_decode_maps_from_charset_string(self):
        expected_encode_map = {'a': 0, 'b': 1, 'c':2, '': 3}
        expected_decode_map = {0: 'a', 1: 'b', 2: 'c', 3: ''}
        self.encoder_decoder.initialize_encode_and_decode_maps_from(self.charset_string)
        self.assertEquals(self.encoder_decoder.encode_map, expected_encode_map)
        self.assertEquals(self.encoder_decoder.decode_map, expected_decode_map)

    def test_encode(self):
        self.encoder_decoder.initialize_encode_and_decode_maps_from(self.charset_string)
        string_to_encode = "cba"
        expected_encoded_string = [2,1,0]
        encoded_string = self.encoder_decoder.encode(string_to_encode)
        self.assertEqual(encoded_string, expected_encoded_string)

    def test_decode(self):
        self.encoder_decoder.initialize_encode_and_decode_maps_from(self.charset_string)
        encoded_string_to_decode = [2,1,0]
        expected_decoded_string = "cba"
        decoded_string = self.encoder_decoder.decode(encoded_string_to_decode)
        self.assertEqual(decoded_string, expected_decoded_string)


if __name__ == '__main__':
    unittest.main()
