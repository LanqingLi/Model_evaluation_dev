import numpy as np
from easydict import EasyDict as edict
from tools.data_preprocess import get_label_classes_from_xls

config = edict()
config.seqlen = 9
config.OLD_MODEL = True
config.FOR_DICOM = True
config.WINDOW_SHIFT = True
config.NORMALIZATION = True
config.IF_FLOAT = False

config.WINDOW_CENTER = -600
config.WINDOW_WIDTH = 1500

config.INPUT_IMAGE_SIZE = (512,512) # for loading boxes in the pascal_voc.py, should be okay for CT

# custom data loader, enabled in the train_end2end.py,s see anchor_loader.py and data_loader.py
config.DATA_LOADER = edict()
config.DATA_LOADER.ENABLED = False
config.DATA_LOADER.NUM_WORKER = 3
config.DATA_LOADER.WORKER_TYPE = 'thread'

## for dataset load function, support dicom and jpg, see io/image.py and dataset/pasco_voc.py gt_roidb func

# network related params
config.PIXEL_MEANS = np.array([0, 0, 0])
config.IMAGE_STRIDE = 0
config.RPN_FEAT_STRIDE = [4, 8, 16, 32]
config.RCNN_FEAT_STRIDE = [4, 8, 16, 32]
# config.FIXED_PARAMS = []
# config.FIXED_PARAMS_SHARED = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

# for multi_class loader for training or single class loader for training,see pasco_voc.py.
CLASSES_LABELS_XLS_FILE_NAME = 'lung/classname_labelname_mapping.xls'
config.CLASSES, config.NODULE_CLASSES, config.CLASS_DICT, config.CONF_THRESH = get_label_classes_from_xls(CLASSES_LABELS_XLS_FILE_NAME)
config.NUM_CLASSES = len(config.CLASSES)


config.SCALES = [(800, 800)]  # first is scale (the shorter side); second is max size
config.ANCHOR_SCALES = [[32], [16], [8], [4]]

config.ANCHOR_RATIOS = (0.5, 1, 2)
config.NUM_ANCHORS = [len(anchor_scales) * len(config.ANCHOR_RATIOS) for anchor_scales in config.ANCHOR_SCALES]

config.TRAIN = edict()

# # for module setting, related file: module.py
config.TRAIN.MULTI_THREAD_ENABLE = False
config.TRAIN.QUEUE_CAPACITY = 3


# # RPN anchor loader
# # rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 64


# # used for end2end training
# # RPN proposal
config.TRAIN.CXX_PROPOSAL = False   # for cuda proposal, by default: False


config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = False

config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 12000
config.TEST.RPN_POST_NMS_TOP_N = 1000
config.TEST.RPN_MIN_SIZE = [16] * 4

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 1000
config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.01

