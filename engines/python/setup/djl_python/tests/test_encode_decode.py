import unittest
import csv
import numpy as np
from io import StringIO
from unittest.mock import Mock, patch
from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.encode_decode import decode, encode, decode_csv, encode_csv


class TestEncodeDecode(unittest.TestCase):

    def setUp(self):
        self.mock_input = Mock(spec=Input)
        self.mock_output = Mock(spec=Output)
        self.mock_output.add_as_string = Mock()
        self.mock_output.add_as_json = Mock()
        self.mock_output.add_as_numpy = Mock()
        self.mock_output.add_as_npz = Mock()
        self.mock_output.add_property = Mock()

    def test_decode_csv_valid_inputs_header(self):
        csv_content = "inputs,other\ntest input 1,value1\ntest input 2,value2"
        self.mock_input.get_as_string.return_value = csv_content

        result = decode_csv(self.mock_input)

        expected = {"inputs": ["test input 1", "test input 2"]}
        self.assertEqual(result, expected)

    def test_decode_csv_valid_question_header(self):
        csv_content = "question,context\nWhat is AI?,Technology context\nWhat is ML?,Machine learning context"
        self.mock_input.get_as_string.return_value = csv_content

        result = decode_csv(self.mock_input)

        expected = {
            "inputs": [{
                "question": "What is AI?",
                "context": "Technology context"
            }, {
                "question": "What is ML?",
                "context": "Machine learning context"
            }]
        }
        self.assertEqual(result, expected)

    def test_decode_csv_invalid_header(self):
        csv_content = "invalid,header\nvalue1,value2"
        self.mock_input.get_as_string.return_value = csv_content

        with self.assertRaises(ValueError) as context:
            decode_csv(self.mock_input)

        self.assertIn("correct CSV with Header columns",
                      str(context.exception))

    def test_encode_csv_list_of_dicts(self):
        content = [{
            "name": "Axl",
            "age": "30"
        }, {
            "name": "Fiona",
            "age": "25"
        }]

        result = encode_csv(content)

        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["name"], "Axl")
        self.assertEqual(rows[1]["name"], "Fiona")

    def test_encode_csv_single_dict(self):
        content = [{"name": "Axl", "age": "30"}]

        result = encode_csv(content)

        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Axl")

    def test_decode_no_content_type(self):
        self.mock_input.get_as_bytes.return_value = None

        result = decode(self.mock_input, None)

        expected = {"inputs": ""}
        self.assertEqual(result, expected)

    def test_decode_no_content_type_with_json(self):
        test_data = {"test": "data"}
        self.mock_input.get_as_bytes.return_value = b'{"test": "data"}'
        self.mock_input.get_as_json.return_value = test_data

        result = decode(self.mock_input, None)

        self.assertEqual(result, test_data)

    def test_decode_application_json(self):
        test_data = {"message": "hello"}
        self.mock_input.get_as_json.return_value = test_data

        result = decode(self.mock_input, "application/json")

        self.assertEqual(result, test_data)

    def test_decode_text_csv(self):
        csv_content = "inputs\ntest input"
        self.mock_input.get_as_string.return_value = csv_content

        with patch('djl_python.encode_decode.decode_csv') as mock_decode_csv:
            mock_decode_csv.return_value = {"inputs": ["test input"]}
            result = decode(self.mock_input, "text/csv")

        mock_decode_csv.assert_called_once_with(self.mock_input)
        self.assertEqual(result, {"inputs": ["test input"]})

    def test_decode_text_plain(self):
        text_content = "Hello world"
        self.mock_input.get_as_string.return_value = text_content

        result = decode(self.mock_input, "text/plain")

        expected = {"inputs": ["Hello world"]}
        self.assertEqual(result, expected)

    def test_decode_image_content_type(self):
        image_data = b"fake_image_data"
        self.mock_input.get_as_image.return_value = image_data

        result = decode(self.mock_input, "image/jpeg")

        expected = {"inputs": image_data}
        self.assertEqual(result, expected)

    def test_decode_audio_content_type(self):
        audio_data = b"fake_audio_data"
        self.mock_input.get_as_bytes.return_value = audio_data

        result = decode(self.mock_input, "audio/wav")

        expected = {"inputs": audio_data}
        self.assertEqual(result, expected)

    def test_decode_tensor_npz(self):
        tensor_data = [np.array([1, 2, 3])]
        self.mock_input.get_as_npz.return_value = tensor_data

        result = decode(self.mock_input, "tensor/npz")

        expected = {"inputs": tensor_data}
        self.assertEqual(result, expected)

    def test_decode_tensor_ndlist(self):
        tensor_data = [np.array([1, 2, 3])]
        self.mock_input.get_as_numpy.return_value = tensor_data

        result = decode(self.mock_input, "tensor/ndlist")

        expected = {"inputs": tensor_data}
        self.assertEqual(result, expected)

    def test_decode_application_x_npy(self):
        tensor_data = [np.array([1, 2, 3])]
        self.mock_input.get_as_numpy.return_value = tensor_data

        result = decode(self.mock_input, "application/x-npy")

        expected = {"inputs": tensor_data}
        self.assertEqual(result, expected)

    def test_decode_form_urlencoded(self):
        form_data = "key1=value1&key2=value2"
        self.mock_input.get_as_string.return_value = form_data

        result = decode(self.mock_input, "application/x-www-form-urlencoded")

        expected = {"inputs": form_data}
        self.assertEqual(result, expected)

    def test_decode_octet_stream(self):
        binary_data = b"binary_data"
        self.mock_input.get_as_bytes.return_value = binary_data

        result = decode(self.mock_input, "application/octet-stream")

        expected = {"inputs": binary_data}
        self.assertEqual(result, expected)

    def test_decode_with_key(self):
        test_data = {"test": "data"}
        self.mock_input.get_as_json.return_value = test_data

        result = decode(self.mock_input, "application/json", key="test_key")

        self.mock_input.get_as_json.assert_called_with(key="test_key")
        self.assertEqual(result, test_data)

    def test_encode_default_json(self):
        prediction = {"result": "success"}

        encode(self.mock_output, prediction, None)

        self.mock_output.add_as_json.assert_called_once_with(prediction,
                                                             key=None)
        self.mock_output.add_property.assert_called_once_with(
            "Content-Type", "application/json")

    def test_encode_application_json(self):
        prediction = {"result": "success"}

        encode(self.mock_output, prediction, "application/json")

        self.mock_output.add_as_json.assert_called_once_with(prediction,
                                                             key=None)
        self.mock_output.add_property.assert_called_once_with(
            "Content-Type", "application/json")

    def test_encode_text_csv(self):
        prediction = [{"name": "Axl", "age": "30"}]

        with patch('djl_python.encode_decode.encode_csv') as mock_encode_csv:
            mock_encode_csv.return_value = "name,age\nAxl,30\n"
            encode(self.mock_output, prediction, "text/csv")

        mock_encode_csv.assert_called_once_with(prediction)
        self.mock_output.add_as_string.assert_called_once_with(
            "name,age\nAxl,30\n", key=None)
        self.mock_output.add_property.assert_called_once_with(
            "Content-Type", "text/csv")

    def test_encode_tensor_npz(self):
        prediction = [np.array([1, 2, 3])]

        encode(self.mock_output, prediction, "tensor/npz")

        self.mock_output.add_as_npz.assert_called_once_with(prediction,
                                                            key=None)
        self.mock_output.add_property.assert_called_once_with(
            "Content-Type", "tensor/npz")

    def test_encode_other_content_type(self):
        prediction = [np.array([1, 2, 3])]

        encode(self.mock_output, prediction, "custom/type")

        self.mock_output.add_as_numpy.assert_called_once_with(prediction,
                                                              key=None)
        self.mock_output.add_property.assert_called_once_with(
            "Content-Type", "tensor/ndlist")

    def test_encode_with_key(self):
        prediction = {"result": "success"}

        encode(self.mock_output,
               prediction,
               "application/json",
               key="test_key")

        self.mock_output.add_as_json.assert_called_once_with(prediction,
                                                             key="test_key")


if __name__ == '__main__':
    unittest.main()
