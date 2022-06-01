import requests
import unittest
import numpy as np
import io

endpoint = "http://127.0.0.1:8080/predictions/test"


class TestInputOutput(unittest.TestCase):

    def test_text(self):
        text = "Hello World!"
        headers = {'content-type': 'text/string'}
        res = requests.post(endpoint, headers=headers, data=text)
        self.assertEqual(res.text, text)

    def test_npz_array(self):
        nd = np.random.rand(3, 2)
        headers = {'content-type': 'tensor/npz'}
        memory_file = io.BytesIO()
        np.savez(memory_file, nd, nd)
        memory_file.seek(0)
        res = requests.post(endpoint, headers=headers, data=memory_file.read(-1))
        result = np.load(io.BytesIO(res.content))
        for item in result.values():
            self.assertTrue(np.array_equal(item, nd))

    def test_ndlist(self):
        from djl_python import np_util
        nd = np.random.rand(3, 2)
        headers = {'content-type': 'tensor/ndlist'}
        res = requests.post(endpoint, headers=headers, data=np_util.to_nd_list([nd, nd]))
        result = np_util.from_nd_list(res.content)
        for item in result:
            self.assertTrue(np.array_equal(item, nd))

    def test_json(self):
        data = {"key1": "value1", "key2": [1, 2, 3]}
        headers = {'content-type': 'application/json'}
        res = requests.post(endpoint, headers=headers, json=data)
        result = res.json()
        self.assertEqual(result, data)

    def test_image(self):
        from PIL import Image
        response = requests.get("https://ultralytics.com/images/zidane.jpg")
        headers = {'content-type': 'image/jpg'}
        img = Image.open(io.BytesIO(response.content))
        res = requests.post(endpoint, headers=headers, data=img.tobytes())
        self.assertEqual(res.content, img.tobytes())


if __name__ == '__main__':
    unittest.main()
