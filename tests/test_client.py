#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

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
        res = requests.post(endpoint,
                            headers=headers,
                            data=memory_file.read(-1))
        result = np.load(io.BytesIO(res.content))
        for item in result.values():
            self.assertTrue(np.array_equal(item, nd))

    def test_ndlist(self):
        from djl_python import np_util
        nd = np.random.rand(3, 2)
        headers = {'content-type': 'tensor/ndlist'}
        res = requests.post(endpoint,
                            headers=headers,
                            data=np_util.to_nd_list([nd, nd]))
        result = np_util.from_nd_list(res.content)
        for item in result:
            self.assertTrue(np.array_equal(item, nd))

    def test_json(self):
        data = {"key1": "value1", "key2": [1, 2, 3], "key3": True}
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
    unittest.main(verbosity=2)
