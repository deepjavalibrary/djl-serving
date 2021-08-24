import numpy as np

from util.np_util import djl_to_np_decode


class Preprocessor(object):

    def initialize(self):
        pass

    def preprocess(self, input_data) -> list[np.ndarray]:
        content = input_data.get_content()
        pair_keys = content.get_keys()
        if "data" in pair_keys:
            return content.get_as_numpy("data")
        elif "body" in pair_keys:
            return content.get_as_numpy("body")
        else:
            data = list(content.get_values())[0]
        np_list = djl_to_np_decode(data)
        return np_list
