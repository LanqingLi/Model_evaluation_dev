# encoding: utf-8
# This file contains functions that might be useful in finding or comparing 3D objects that are made up of 2D boxes.
#
# datatype: box
# [x1, y1, x2, y2, z1, z2,...]
#
# iou and miov are two metrics that compare similarities of two boxes' coordinates
# iou = intersecton area/union area; miov = (intersection area/box1 area + intersection area/box2 area) / 2
# iou is commonly used, but is sometimes not what we want since if two boxes differ in size, iou would always be small,
# no matter how close the boxes are. If we want the metric to consider such situation as a rather good match (we
# usually do want that in medical images), we can use miov, or difference in center as other metrics.
#
# To match a patient's predicted results with ground truth results, first use nodule_list2_patient to add his predicted
# results and ground truth results, then use Patient.match to find match result. If your nodule/plaque information is
# not as specified, you could write your own Patient list constructor.
#
# If you have any question, contact llanqing@infervision.com


import math
import numpy as np
from copy import deepcopy


class AnchorMetric(object):
    '''
    bndbox type: np.array([[xmin, ymin, xmax, ymax, ...]]), shape: (1, 2 * self.dim), dtype = 'float32'
    param dim: anchor的维度
    '''
    def __init__(self, dim=2):
        self.dim = dim

    def check_box(self, bndbox):
        assert (bndbox.shape == (1, 2 * self.dim))
        assert (bndbox.dtype == 'float')
        box = bndbox.copy().reshape(2, self.dim)
        if np.any(box[1] < box[0]):
            raise Exception("Boxes should be represented as np.array([[xmin, ymin, xmax, ymax, ...]]), where xmax >= xmin and ymax >= ymin")

    def reshape(self, bndbox):
        self.check_box(bndbox)
        box = bndbox.copy().reshape(2, self.dim)
        return box

    def overlap1D(self, bndbox1, bndbox2):
        """
        Returns the overlap of 1d segment, returns [0, 0] if not overlapped.
        :params x: 1d np array of 2 elements. [st, ed]
        :params y: 1d np array of 2 elements. [st ,ed]
        """
        dim = self.dim
        self.dim = 1
        self.check_box(bndbox1)
        self.check_box(bndbox2)
        box1 = self.reshape(bndbox1)
        box2 = self.reshape(bndbox2)
        self.dim = dim

        lower_end = max(box1[0][0], box2[0][0])
        higher_end = min(box1[1][0], box2[1][0])
        if lower_end >= higher_end:
            return [0, 0]
        else:
            return np.array([lower_end, higher_end])

    def overlapND(self, bndbox1, bndbox2):
        """
        Returns the overlap of n-d segment, returns [0, 0] in any dimension where x, y do not overlap
        :params x: 2*n np array
        :params y: 2*n np array
        """
        self.check_box(bndbox1)
        self.check_box(bndbox2)
        box1 = self.reshape(bndbox1)
        box2 = self.reshape(bndbox2)
        res = []
        for i in range(box1.shape[1]):
            res.append(self.overlap1D(box1[:, i:i+1].T, box2[:, i:i+1].T))
        return np.vstack(res).T

    def iou(self, bndbox1, bndbox2):
        self.check_box(bndbox1)
        self.check_box(bndbox2)
        area1 = self.cal_box_vol(bndbox1)
        area2 = self.cal_box_vol(bndbox2)
        intersection_area = self.cal_boxes_intersect(bndbox1, bndbox2)
        union_area = area1 + area2 - intersection_area
        iouh = float(intersection_area) / union_area
        return iouh

    # mean intersection over volume of box(miov)
    def miov(self, bndbox1, bndbox2):
        self.check_box(bndbox1)
        self.check_box(bndbox2)
        box1_vol = self.cal_box_vol(bndbox1)
        box2_vol = self.cal_box_vol(bndbox2)
        intersection_vol = self.cal_boxes_intersect(bndbox1, bndbox2)
        return 0.5 * (intersection_vol/box1_vol + intersection_vol/box2_vol)


    def cal_box_vol(self, bndbox):
        """
        计算bndbox的广义体积
        :param bndbox:bbox，list或者np array
        :return:float
        """
        self.check_box(bndbox)
        box = self.reshape(bndbox)
        return float(np.prod(box[1] - box[0]))


    def cal_boxes_intersect(self, bndbox1, bndbox2):
        '''
        #calculate area of intersection between two boxes
        #Not accurate since in function fill_every_layer trapezoidals may appear as boxes, however error is negligible
        :param bndbox1:
        :param bndbox2:
        :return:
        '''
        overlap = self.overlapND(bndbox1, bndbox2)
        return np.prod(overlap[1] - overlap[0])


    def getdistance(self, center1, center2):
        """
        计算两个坐标的欧式距离
        :param center1:
        :param center2:
        :return:float
        """
        assert center1.shape == (1, self.dim)
        assert center2.shape == (1, self.dim)
        return math.sqrt(np.sum((center1 - center2) ** 2))


    def getboxcenter(self, bndbox):
        """
        根据bbox获取中心点
        :param bbox
        :return:float
        """
        self.check_box(bndbox)
        box = self.reshape(bndbox)
        center = (np.sum(box, axis=0)/2.).reshape(1, self.dim)
        return center


    def calc_box_distance(self, bndbox1, bndbox2):
        '''
        distance between box centers
        :param box1:
        :param box2:
        :return:
        '''
        return self.getdistance(self.getboxcenter(bndbox1), self.getboxcenter(bndbox2))

    def center_deviation(self, bndbox1, bndbox2):
        '''
        Center distance divided by the half-width of the longest side.
        :param box1:
        :param box2:
        :return: np.array([x, y, ...]), dtype = 'float'
        '''
        self.check_box(bndbox1)
        self.check_box(bndbox2)
        center1 = self.getboxcenter(bndbox1)
        center2 = self.getboxcenter(bndbox2)
        box1 = self.reshape(bndbox1)
        box2 = self.reshape(bndbox2)
        long_side = np.max(np.stack((box1[1] - box1[0], box2[1] - box2[0])), axis=0) / 2.
        return np.absolute(center1 - center2) / long_side

    def center_deviation_iou(self, bndbox1, bndbox2):
        '''
        Center deviation divided by iou.
        :param bndbox1:
        :param bndbox2:
        :return: np.array([x, y, ...]), dtype = 'float'
        '''
        return self.center_deviation(bndbox1, bndbox2) / self.iou(bndbox1, bndbox2)

    def center_deviation_sqrt(self, bndbox1, bndbox2):
        '''
        Square root of center deviation.
        :param box1:
        :param box2:
        :return: float
        '''
        self.check_box(bndbox1)
        self.check_box(bndbox2)
        center1 = self.getboxcenter(bndbox1)
        center2 = self.getboxcenter(bndbox2)
        box1 = self.reshape(bndbox1)
        box2 = self.reshape(bndbox2)
        long_side = np.max(np.stack((box1[1] - box1[0], box2[1] - box2[0])), axis=0) / 2.
        return math.sqrt(np.sum(((center1 - center2)/long_side) ** 2))

# if __name__ == '__main__':
#     test_cls = AnchorMetric(dim=2)
#     box1 = np.asarray([[1.,2,4,7]])
#     box2 = np.asarray([[2,4.,6,8]])
    # print box1.shape
    # print test_cls.overlapND(box1, box2)
    # print test_cls.iou(box1, box2)
    # print test_cls.miov(box1, box2)
    # print test_cls.cal_box_vol(box1)
    # print test_cls.cal_boxes_intersect(box1, box2)
    # print test_cls.calc_box_distance(box1, box2)
    # print test_cls.center_deviation(box1, box2)
    # print test_cls.center_deviation_iou(box1, box2)
    # print test_cls.center_deviation_sqrt(box1, box2)