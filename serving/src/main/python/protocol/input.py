import numpy as np

from util.np_util import djl_to_np_decode
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
        """
        Sets the request id

        :param request_id: request id
        """
        self.request_id = request_id

    def set_properties(self, properties: map):
        """
        Sets the properties
        :param properties: map
        """
        self.properties = properties

    def set_content(self, content: PairList):
        """
        Sets the content

        :param content: bytearray
        """
        self.content = content

    def add_property(self, key: str, val: str):
        """
        Adds a property to properties map
        :param key: key of the property
        :param val: value of the property
        """
        self.properties[key] = val

    def get_as_numpy(self, key=None) -> list[np.ndarray]:
        """
        Returns
            1. value as numpy list if key is provided
            2. list of values as numpy list if key is not provided
        :param key: optional key
        :return: list of numpy array
        """
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
