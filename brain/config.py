from easydict import EasyDict as edict
import os
import numpy as np
from tools.data_preprocess import get_label_classes_from_xls_seg

config = edict()

config.TEST = edict()

config.TEST.CONF_THRESHOLD = np.linspace(0.001, 0.009, num=9).tolist() + np.linspace(0.01, 0.09, num=9).tolist() + np.linspace(0.1, 0.9, num=9).tolist()

config.CLASSES_LABELS_XLS_FILE_NAME = 'brain/classname_labelname_mapping.xls'
config.CLASSES, config.SEG_CLASSES, config.CLASS_DICT, config.CONF_THRESH, config.CLASS_WEIGHTS = get_label_classes_from_xls_seg(config.CLASSES_LABELS_XLS_FILE_NAME)
config.NUM_CLASSES = len(config.CLASSES)

config.FSCORE_BETA = 1.

# binary classification threshold for drawing contour plot for single threshold for comparison
config.THRESH = 0.5