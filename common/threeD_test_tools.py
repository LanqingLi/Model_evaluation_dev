# encoding: utf-8
# This file contains functions that might be useful in finding or comparing 3D objects that are made up of 2D boxes.
#
# datatype: box
# [x1, y1, x2, y2, label, layerID, probability(optional)]
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
# If you have any question, contact wjinyi@infervision.com


import math
from copy import deepcopy
from xml_tools import *


def iou(bndbox1, bndbox2):
    area1 = cal_box_square(bndbox1)
    area2 = cal_box_square(bndbox2)
    iw = min(bndbox2[2], bndbox1[2]) - max(bndbox2[0], bndbox1[0])
    ih = min(bndbox2[3], bndbox1[3]) - max(bndbox2[1], bndbox1[1])
    iw = max(0.0, iw)
    ih = max(0.0, ih)
    intersection_area = iw * ih
    union_area = area1 + area2 - intersection_area
    iouh = float(intersection_area) / union_area
    iouh = pow(iouh, 1/max(1.0, float(bndbox2[5]-bndbox1[5])))
    return iouh


# mean intersection over volume of box(miov)
def miov(box1, box2):
    box1_vol = cal_box_square(box1)
    box2_vol = cal_box_square(box2)
    intersection_vol = cal_boxes_intersection(box1, box2)
    return 0.5 * (intersection_vol/box1_vol + intersection_vol/box2_vol)


def cal_box_square(bndbox):
    """
    计算bbox的面积
    :param bndbox:bbox，list或者np array
    :return:float
    """
    return float((bndbox[2] - bndbox[0]) * (bndbox[3] - bndbox[1]))


def cal_boxes_intersection(bndbox1, bndbox2):
    '''
    #calculate area of intersection between two boxes
    #Not accurate since in function fill_every_layer trapezoidals may appear as boxes, however error is negligible
    :param bndbox1:
    :param bndbox2:
    :return:
    '''
    iw = min(bndbox2[2], bndbox1[2]) - max(bndbox2[0], bndbox1[0])
    ih = min(bndbox2[3], bndbox1[3]) - max(bndbox2[1], bndbox1[1])
    iw = max(0.0, iw)
    ih = max(0.0, ih)
    intersection_area = iw * ih
    return intersection_area


def getdistance(center1, center2):
    """
    计算两个坐标的欧式距离
    :param center1:
    :param center2:
    :return:float
    """
    return math.sqrt(math.pow(center1[0] - center2[0], 2) + math.pow(center1[1] - center2[1], 2))


def getboxcenter(bbox):
    """
    根据bbox获取中心点
    :param bbox
    :return:float
    """
    centercoor = [float(bbox[0] + bbox[2]) / 2, float(bbox[1] + bbox[3] / 2)]
    return centercoor


def calc_box_distance(box1, box2):
    '''
    distance between box centers
    :param box1:
    :param box2:
    :return:
    '''
    return getdistance(getboxcenter(box1), getboxcenter(box2))


def center_deviation(box1, box2):
    '''
    Center distance divided by longest side.
    :param box1:
    :param box2:
    :return: float
    '''
    long_side = float(max([box1[2] - box1[0], box1[3] - box1[1], box2[2] - box2[0], box2[3] - box2[1]]))
    return float(getdistance(box1, box2)) / long_side


def nodule_list2_patient(bboxInfo, nodule_list, check, patient_name, patient_list):
    '''
    receive parameters from find_nodules
    :param bboxInfo:
    :param nodule_list:
    :param patient_name:
    :return: an instance of class Patient
    '''
    new_list = []
    for nodule in nodule_list:
        box_list = nodule['noduleList']
        nodule_obj = ThreedObject(patient_name=patient_name)
        for box_index in box_list:
            box = bboxInfo.loc[box_index]
            box_t = [box['xmin'], box['ymin'], box['xmax'], box['ymax'], box['nodule_class'],
                     box['instanceNumber'], box['prob']]
            nodule_obj.add_box(box_t)
        new_list.append(nodule_obj)
    patient = None
    add_to_list = False
    for p in patient_list:
        if patient_name == p.name:
            patient = p
    if patient is None:
        patient = Patient(patient_name)
        add_to_list = True
    patient.add_objectlist(new_list, check)
    if add_to_list:
        patient_list.append(patient)
    return patient_list


class ThreedObject():

    def __init__(self, patient_name='Unknown', LAYER_TOLERANCE=3, MAX_DEVIATION=0.4):
        self.label = {}
        self.boxes = []
        self.patient_name = patient_name
        self.res = None
        self.max_matching = 0
        self.LAYER_TOLERANCE = LAYER_TOLERANCE
        self.MAX_DEVIATION = MAX_DEVIATION

    def btm_box(self):
        if len(self.boxes) > 0:
            return self.boxes[-1]
        else:
            return None

    def noofboxes(self):
        return len(self.boxes)

    def majority_label(self):
        local_max = 0
        for key, value in self.label.iteritems():
            if value > local_max:
                local_max = value
                local_res = [key]
            if value == local_max:
                local_res.append(key)
        return local_res

    def is_close(self, box):
        if self.btm_box() is None:
            return True
        else:
            if box[5] - self.btm_box()[5] not in range(0, self.LAYER_TOLERANCE+1):
                return False
            cd = center_deviation(self.btm_box(), box)
            if cd <= self.MAX_DEVIATION:
                return True
            return False

    def add_box(self, box):
        self.label[box[4]] = self.label.get(box[4], 0) + 1
        self.boxes.append(box)
        sorted(self.boxes, key=lambda x: x[5])

    def volume(self):
        # not actual 3d volume, unit height 1, width and length pixel
        vol = 0.0
        for box, box2 in zip(self.boxes, self.boxes[1:]):
            vol += float(cal_box_square(box) + cal_box_square(box2)) * float(box2[5] - box[5]) / 2
        vol += cal_box_square(self.btm_box())
        return vol

    # fill gaps in plaque
    def fill_every_layer(self):
        fullset = []
        for layer_o, layer2_o in zip(self.boxes, self.boxes[1:]):
            layer = deepcopy(layer_o)
            layer2 = deepcopy(layer2_o)
            fullset.append(layer)
            while layer[5] < layer2[5] - 1:
                for i in range(4):
                    layer[i] += float(layer2[i] - layer[i]) / float(layer2[5] - layer[5])
                layer[5] += 1
        fullset.append(self.btm_box())
        box_new = deepcopy(self.btm_box())
        box_new[5] += 1
        fullset.append(box_new)
        return fullset

    def intersection_vol(self, object):
        '''
        #Calculate the volume of intersection between two objects
        #Note this is not geometrically accurate, this method considers (top_intersection - btm_intersection) * h
        #as the volume of intersection. This may serve as a better metric since position of plaque is moving in
        #different layes. Moreover, it's computationally expensive and mathematically complicated to calculate its
        #volume of intersection
        :param another 3D_object:
        :return:
        '''
        tot_vol = 0.0
        object1 = self.fill_every_layer()
        object2 = object.fill_every_layer()
        if object1[0][5] > object2[0][5]:
            object1, object2 = object2, object1
        i = 0
        while i < len(object1) and object1[i][5] < object2[0][5]:
            i += 1
        j = 0
        while i+1 < len(object1) and j+1 < len(object2):
            tot_vol += float(cal_boxes_intersection(object1[i], object2[j]) +
                        cal_boxes_intersection(object1[i+1], object2[j+1])) / 2
            i+=1
            j+=1
        return tot_vol

    # mean of the centers of the boxes as a two-element list
    def center(self):
        xtot = 0
        ytot = 0
        tn = self.noofboxes()
        for box in self.boxes:
            [xh, yh] = getboxcenter(box)
            xtot += xh
            ytot += yh
        return [float(xtot/tn), float(ytot/tn)]

    def vol_miov(self, object):
        i_vol = self.intersection_vol(object)
        vol1 = self.volume()
        vol2 = object.volume()
        return (i_vol/vol1 + i_vol/vol2) / 2


class Patient:
    '''
        Including a list of one's 3D_objects, his name and optionally number of CT images.
        match_dict match gt_index to pt_index(in cases of true positive)
        function match matches pt object list and gt object list. Results are stored in each object.res.
        Number of tp, fp, tn, fn  are inside Patient.stats
    '''
    def __init__(self, name, pt_list = None, gt_list = None):
        self.name = name
        self.pt_list = pt_list
        self.gt_list = gt_list
        self.stats = {}
        self.match_dict = {}
        self.nooflayers = None

    def record_layers(self, num):
        self.nooflayers = num

    def add_objectlist(self, objectlist, check):
        if check == 'gt':
            self.gt_list = objectlist
        if check == 'pt':
            self.pt_list = objectlist

    def match(self, metric=ThreedObject.vol_miov, metric_threshold=0.3):
        '''
        :param metric: an evaluation function that takes in two 3D objects and measure their similarity
        :param metric_threshold: if similarity > metric_threshold, consider this as successful matching
        :return: a dictionary that states number of tp, fp, ... in this patient.
        '''
        if len(self.stats) > 0:
            return self.stats
        self.stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        if self.pt_list is None:
            raise ValueError('%s has no predicted results.' % self.name)
        if self.gt_list is None:
            raise ValueError('%s has no ground truth results.' % self.name)
        for gt_id, gt_obj in enumerate(self.gt_list):
            matched_pt_id = None
            for pt_id, pt_obj in enumerate(self.pt_list):
                match_score = metric(gt_obj, pt_obj)
                if match_score > gt_obj.max_matching:
                    gt_obj.max_matching = match_score
                    matched_pt_id = pt_id
                if match_score > pt_obj.max_matching:
                    pt_obj.max_matching = match_score
            if gt_obj.max_matching > metric_threshold:
                gt_obj.res = 'tp'
                self.stats['tp'] += 1
                self.match_dict[gt_id] = matched_pt_id
                self.pt_list[matched_pt_id].res = 'tp'
            else:
                gt_obj.res = 'fn'
                self.stats['fn'] += 1
        for pt in self.pt_list:
            if pt.res is None:
                pt.res = 'fp'
                self.stats['fp'] += 1
        return self.stats


