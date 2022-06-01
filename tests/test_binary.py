import argparse
import sys
import unittest
import requests
import numpy as np
import io

endpoint = "http://127.0.0.1:8080/predictions/test"


class TestInputOutput(unittest.TestCase):
    shape = []
    outshape = []

    def test_npz_array(self):
        nd = np.random.rand(*self.shape).astype('float32')
        headers = {'content-type': 'tensor/npz'}
        memory_file = io.BytesIO()
        np.savez(memory_file, nd)
        memory_file.seek(0)
        res = requests.post(endpoint, headers=headers, data=memory_file.read(-1))
        result = np.load(io.BytesIO(res.content))
        for item in result.values():
            self.assertEqual(item.shape, tuple(self.outshape))

    def test_ndlist(self):
        from djl_python import np_util
        nd = np.random.rand(*self.shape).astype('float32')
        headers = {'content-type': 'tensor/ndlist'}
        res = requests.post(endpoint, headers=headers, data=np_util.to_nd_list([nd]))
        result = np_util.from_nd_list(res.content)
        for item in result:
            self.assertEqual(item.shape, tuple(self.outshape))


def string2integer(input_str):
    return list(map(int, input_str.split(",")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ML models')
    parser.add_argument("inputshape", type=str, help="the input shape like 1,3,224,224")
    parser.add_argument("outputshape", type=str, help="the output shape like 1,1000")
    args = parser.parse_args()
    TestInputOutput.shape = string2integer(args.inputshape)
    TestInputOutput.outshape = string2integer(args.outputshape)
    unittest.main(verbosity=2, argv=[sys.argv[0]])
