import unittest
import numpy as np
from djl_python import test_model, Input, Output


class TestInputOutput(unittest.TestCase):

    def test_empty_input(self):
        inputs = Input()
        with self.assertRaises(Exception):
            inputs.get_as_string()

    def test_string_input(self):
        input_text = "Hello World"
        inputs = test_model.create_text_request(input_text)
        result = inputs.get_as_string()
        self.assertEqual(input_text, result)
        with self.assertRaises(KeyError):
            inputs.get_as_string("not-exist-key")

    def test_numpy_input(self):
        nd = [np.ones((1, 3, 2))]
        inputs = test_model.create_numpy_request(nd)
        result = inputs.get_as_numpy()
        self.assertTrue(np.array_equal(result[0], nd[0]))
        inputs = test_model.create_npz_request(nd)
        result = inputs.get_as_npz()
        self.assertTrue(np.array_equal(result[0], nd[0]))

    def test_output(self):
        test_dict = {"Key": "Value"}
        nd = [np.ones((1, 3, 2))]
        outputs = Output().add_as_json(test_dict, "dict").add_as_numpy(
            nd, "ndlist").add_as_npz(nd, "npz")
        result = test_model.extract_output_as_string(outputs, "dict")
        self.assertTrue(result, test_dict)
        result = test_model.extract_output_as_numpy(outputs, "ndlist")
        self.assertTrue(np.array_equal(result[0], nd[0]))
        result = test_model.extract_output_as_npz(outputs, "npz")
        self.assertTrue(np.array_equal(result[0], nd[0]))

    def test_print_message(self):
        nd = [np.ones((1, 3, 2))]
        inputs = test_model.create_numpy_request(nd, "mydata")
        result = inputs.__str__()
        expected = '''properties: {'device_id': '-1', 'content-type': 'tensor/ndlist'}
mydata: [array([[[1., 1.],
        [1., 1.],
        [1., 1.]]])]'''
        self.assertEqual(result, expected)

    def test_finalize(self):

        def finalize_func(a, b, c):
            return a + b + c

        outputs = Output().finalize(finalize_func, 1, 2, 3)
        self.assertEqual(6, outputs.execute_finalize())


if __name__ == '__main__':
    unittest.main()
