# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""
import json

import numpy as np


class NpEncoder(json.JSONEncoder):
    """
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741

    fixed
    > Python - TypeError: Object of type 'int64' is not JSON serializable

    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    pass
