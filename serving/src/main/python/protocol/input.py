from typing import List

import numpy as np

from util.numpy_djl_util import djl_to_np_decode
from util.pair_list import PairList


class Input(object):
    def __init__(self):
        self.request_id = None
        self.properties = dict()
        self.content = PairList()

    def get_request_id(self) -> str:
        """
        Returns the request id

        :return: request_id
        """
        return self.request_id

    def get_properties(self) -> map:
        """
        Returns the properties

        :return: properties
        """
        return self.properties

    def get_content(self) -> map:
        """
        Returns the content

        :return: content
        """
        return self.content

    def get_properties_value(self, key: str) -> str:
        """
        Returns the value of a property key

        :param key: key of map
        :return: value of the key
        """
        return self.properties[key]

    def set_request_id(self, request_id: str):
        self.request_id = request_id

    def set_properties(self, properties: map):
        self.properties = properties

    def set_content(self, content: PairList):
        self.content = content

    def add_property(self, key: str, val: str):
        self.properties[key] = val

    def get_as_numpy(self, key=None) -> List[np.numpy]:
        # return list of values as numpy list if not provided key
        if key is None:
            values = self.content.get_values()
            result = []
            for value in values:
                result.extend(djl_to_np_decode(value))
            return result
        else:
            value = self.content.get(key=key)
            return djl_to_np_decode(value)
