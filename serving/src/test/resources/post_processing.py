import numpy as np


class PostProcessor(object):
    def initialize(self):
        pass

    def postprocess(self, nd_list: list[np.ndarray]) -> list[np.ndarray]:
        return nd_list
