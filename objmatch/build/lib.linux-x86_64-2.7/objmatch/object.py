import numpy as np

def check_dim(coord_list):
    assert all(isinstance(n, float) for n in coord_list), 'anchor coordinates must be float numbers'
    assert len(coord_list) % 2 == 0, 'coordinates list length must be even'

def check_length(list1, list2):
    assert (len(list1) == len(list2)), 'two lists must have the same length'

class Anchor(object):
    """
    coord_list = [xmin, ymin, ..., xmax, ymax, ...]
    kwargs: attribute keyword list of the object
    args: attribute value list of the object

    """
    def __init__(self, coord_list, value_list, key_list, *args, **kwargs):
        check_dim(coord_list)
        check_length(value_list, key_list)
        self.dim = len(coord_list)/2
        self.coord_list = coord_list
        self.bndbox = np.asarray(coord_list).reshape(1, len(coord_list))
        self.key_list = key_list
        self.attr_dict = {key: value for (key, value) in zip(key_list, value_list)}
