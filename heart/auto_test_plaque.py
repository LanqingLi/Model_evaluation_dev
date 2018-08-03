# coding: utf-8

import os
import pandas as pd
import argparse
import json
import numpy as np
import cv2
import math
import xlrd
import openpyxl
import xml.etree.ElementTree as ET
from copy import deepcopy
from auto_test_config import *


# datatype: box
# [x1, y1, x2, y2, name, layerID, probability(optional)]

# Global variables

def get_label_classes_from_xls(filename):
    '''
    get labels and training classes with mapping relation from xls
    :param filename: xls records
    :return:
    class_list: type: list, a list of classnames
    label_classes: type:list, a list of labels in xml label files
    class_dict: type: dict. mapping label name to class name. e.g: dict[label_name]=class_name
    '''
    # read labelDict dictionary from xlsx file
    classDictSheet = xlrd.open_workbook(filename).sheet_by_index(0)
    label_classes = []
    class_set = set()
    class_dict = dict()  # label to class
    conf_thresh = dict()
    conf_thresh['__background__'] = 1.0
    for i in range(1, classDictSheet.nrows):
        # add class name
        label_name = classDictSheet.row(i)[0].value.strip(' ')
        class_name = classDictSheet.row(i)[1].value.strip(' ')

        label_name = label_name.encode("utf-8")
        class_name = class_name.encode("utf-8")
        label_classes.append(label_name)
        class_set.add(class_name)
        class_dict[label_name] = class_name

        if len(classDictSheet.row(i)) == 3:
            thresh = classDictSheet.row(i)[2].value
            assert isinstance(thresh, float), 'thresh must be float type, check xls'
            conf_thresh[class_name] = thresh

    class_list = []
    class_list.append('__background__')
    for item in class_set:
        class_list.append(item)

    return class_list, label_classes, class_dict, conf_thresh


conclusion = []
conclusion_seg = []
cls_xmlpath = '/home/tx-deepocean/cardiac_plaque_detection/rcnn/classname_labelname_mapping.xls'
class_list, label_classes, class_dict, conf_thresh = get_label_classes_from_xls(cls_xmlpath)
class_list.remove('motion_artifacts')
class_list.remove('abnormal')
class_list.remove('disorder')
class_list.remove('myocardial_bridge')
class_list.remove('stenting')
class_list.remove('__background__')
set_class2 = set([])
set_class = set([])
for item in class_list:
    set_class2.add(item.rsplit('_', 1)[0])
for item in set_class2:
    set_class.add(item + '0')
    set_class.add(item + '1')
# specific_stats records test results for segments and types of plaques. It is a dictionary with segment or type
# name as index and a 2-element array as value.
# Array is [amount of true positive ground truth, amount of total gt plaque, amount of false positive plaque]
specific_stats = {}
set_class2 = sorted(set_class2)
for item in set_class2:
    specific_stats[item] = [0, 0, 0]
specific_stats['ncP'] = [0, 0, 0]
specific_stats['cP'] = [0, 0, 0]
specific_stats['mP'] = [0, 0, 0]


def get_pid_sid(xml_path):
    '''
    #获取 patient_name, slice_id
    :param xml_path:
    :return:
    '''
    xml_name = xml_path.split('/')[-1]
    slice_name = xml_name.split('.')[0]
    [pid, sid] = slice_name.split('_')
    sid = int(sid)
    return pid, sid


def get_pos_type(box_label):
    '''
    #from file name get its blood vessel name and its plaque type
    :param box_label:
    :return:
    '''
    if box_label in class_list:
        [pos, ptype] = box_label.rsplit('_', 1)
        return pos, ptype
    else:
        raise (NameError('Illegal label in function get_pos_type'))


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
    iouh = pow(iouh, 1/float(bndbox2[5]-bndbox1[5]))
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


class Plaque:
    '''
    #position of the plaque : pos (i.e. LM_M)
    #type of the plaque : ptype (a dictionary that maps [nP, ncP, mP] to number of boxes with that label)
    #general type : plaque_type (a string among [nP, ncP, mP])
    #coordinates of boxes: boxes (a list of x1,x2,y1,y2 of each boundbox, in descending order)
    #similarity of two plaques:
    '''

    def __init__(self, pos, ptype, box):
        self.pos = pos
        self.ptype = {'ncP': 0, 'cP': 0, 'mP': 0}
        self.ptype[ptype] += 1
        self.boxes = [box]
        self.res = None
        self.max_matching = 0

    def btm_box(self):
        return self.boxes[-1]

    def noofboxes(self):
        return len(self.boxes)

    def plaque_type(self):
        if self.ptype['mP'] > 0:
            return 'mP'
        if self.ptype['cP'] > 0 and self.ptype['ncP'] > 0:
            return 'mP'
        if self.ptype['cP'] > 0:
            return 'cP'
        return 'ncP'

    def calcium_percentage(self):
        return (self.ptype['cP'] + 0.5 * float(self.ptype['mP'])) / max(float(self.noofboxes()), 1.0) * 100

    def box_is_this_plaque(self, box):
        bpos, btype = get_pos_type(box[4])
        if not (bpos == self.pos):
            return False
        if not (box[5] - LAYER_TOLERANCE <= self.btm_box()[5] < box[5]):
            return False
        dist = float(getdistance(getboxcenter(box), getboxcenter(self.btm_box())) / (box[5] - self.btm_box()[5]))
        if dist > MAX_BOX_DISTANCE:
            return False
        return True

    def add_box(self, box):
        if not (self.box_is_this_plaque(box)):
            return False
        bpos, btype = get_pos_type(box[4])
        self.ptype[btype] += 1
        self.boxes.append(box)

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
        for layer, layer2 in zip(self.boxes, self.boxes[1:]):
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

    # max center distance between boxes in two adjacent layers of this plaque
    def adj_layer_max_dist(self):
        max_dist = 0
        upp_box = 0
        low_box = 0
        for box1, box2 in zip(self.boxes, self.boxes[1:]):
            dist_here = float(getdistance(getboxcenter(box1), getboxcenter(box2)) / (box2[5] - box1[5]))
            if dist_here > max_dist:
                max_dist = dist_here
                upp_box = box1[5]
                low_box = box2[5]
        return max_dist, upp_box, low_box


class Patient:
    '''
        #Including a list of one's plaques, his name and optionally number of CT images.
    '''
    def __init__(self, name, plist):
        self.name = name
        self.plaque_list = plist
        self.nooflayers = None

    def record_layers(self, num):
        self.nooflayers = num


def read_xml(xml_name, restrict_name_list=None, check=None):
    '''
    #读取gt的boxes
    :param xml_name:
    :param restrict_name_list:
    :return:
    '''
    tree = ET.parse(xml_name)
    objs = tree.findall('object')
    box_all = []
    pid, sid = get_pid_sid(xml_name)
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        if name[-1] == 'p':
            str_temp = name[:-1] + 'P'
            name = str_temp
        if restrict_name_list == None:
            restrict_name_list = ['cP', 'mP', 'ncP']
        if check == 'gt':
            if class_dict[name] in restrict_name_list:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                box = [x1, y1, x2, y2, class_dict[name], sid]
                box_all.append(box)
        if check == 'pt':
            if name in restrict_name_list:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                box = [x1, y1, x2, y2, name, sid]
                box_all.append(box)
    return box_all


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


def boxesinto_plaques(patient_path, check=None):
    '''
    #From a patient's directory, read all the boxes and divide them into plaques
    #The classification method is the same for pt and gt
    :param patient_path:
    :param check:
    :return:
    '''
    que_plaque = []
    plaque_list = []
    directory = os.listdir(patient_path)
    directory.sort()
    for path in directory:
        xml_path = os.path.join(patient_path, path)
        boxes_curslide = read_xml(xml_path, class_list, check)
        pid, sid = get_pid_sid(xml_path)
        sid = int(sid)
        # check if all plaques in queue are close enough to this layer
        for plaque in que_plaque:
            if plaque.btm_box()[5] < sid - LAYER_TOLERANCE:
                if plaque not in plaque_list:
                    plaque_list.append(plaque)
            else:
                right_box = None
                box_min = 100000
                for box in boxes_curslide:
                    if plaque.box_is_this_plaque(box):
                        box_dist = calc_box_distance(box, plaque.btm_box())
                        if box_dist < box_min:
                            box_dist2 = 100000
                            for plaque2 in que_plaque:
                                if not(plaque2 == plaque) and plaque2.box_is_this_plaque(box):
                                    box_dist2 = min(box_dist2, calc_box_distance(box, plaque2.btm_box()))
                            if box_dist2 >= box_dist:
                                box_min = box_dist
                                right_box = box
                if right_box is not None:
                    plaque.add_box(right_box)
                    boxes_curslide.remove(right_box)
        for plaque in que_plaque:
            if plaque in plaque_list:
                que_plaque.remove(plaque)
        for box in boxes_curslide:
            bpos, btype = get_pos_type(box[4])
            new_plaque = Plaque(bpos, btype, box)
            que_plaque.append(new_plaque)
    plaque_list.extend(que_plaque)
    return plaque_list


def clean_data(data_path):
    '''
    #去除macbook的隐藏文件
    :param data_path:
    :return:
    '''
    for data_name in os.listdir(data_path):
        if data_name.endswith('DS_Store'):
            remove_path = os.path.join(data_path, data_name)
            print(remove_path)
            os.remove(remove_path)


def result2df(fn_info_list, fp_info_list, tp_info_list, sub_tp_info_list):
    '''
    #将结果保存到dataframe
    '''
    dataframe = pd.DataFrame(columns=['PatientId', 'SliceIndex', 'ptBoxes', 'gtBoxes', 'Result'])
    dataframe = dataframe.append(pd.DataFrame(fp_info_list,
                                              columns=['PatientId', 'SliceIndex', 'ptBoxes', 'gtBoxes', 'Result']))
    dataframe = dataframe.append(pd.DataFrame(fn_info_list,
                                              columns=['PatientId', 'SliceIndex', 'ptBoxes', 'gtBoxes', 'Result']))
    dataframe = dataframe.append(pd.DataFrame(tp_info_list,
                                              columns=['PatientId', 'SliceIndex', 'ptBoxes', 'gtBoxes', 'Result']))
    dataframe = dataframe.append(pd.DataFrame(sub_tp_info_list,
                                              columns=['PatientId', 'SliceIndex', 'ptBoxes', 'gtBoxes', 'Result']))
    dataframe = dataframe.sort_values(by=['PatientId', 'SliceIndex'])
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def print_result(list1, list2, xlsx_name, writeout_path):
    '''
    #save plaque information to dataframes
    :param list1:
    :param list2:
    :return:
    '''
    output = []
    for ix, plaque in enumerate(list1):
        output.append(['Ground Truth', ix, plaque.pos,
                       plaque.plaque_type(), plaque.boxes[0][5], plaque.boxes[-1][5], plaque.noofboxes(), plaque.res])
    for ix, plaque in enumerate(list2):
        output.append(['Predicted', ix, plaque.pos,
                       plaque.plaque_type(), plaque.boxes[0][5], plaque.boxes[-1][5], plaque.noofboxes(), plaque.res])
    dataframe = pd.DataFrame(output, columns=
    ['Check', 'PlaqueId', 'Position', 'Type', 'StartingSlice', 'EndingSlice', 'Noofboxes', 'Result'])
    output_path = os.path.join(writeout_path, xlsx_name)
    writer = pd.ExcelWriter(output_path)
    dataframe.to_excel(writer, 'sheet1')
    writer.save()
    txt_name = xlsx_name.split('.')[-2] + '.txt'
    txt_path = os.path.join(writeout_path, txt_name)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    txt_file = open(txt_path, 'a')
    txt_file.write('Ground Truth\n')
    for plaque in list1:
        txt_file.write(str(plaque.pos) + ' ' + str(plaque.plaque_type()) + ' ' + str(plaque.max_matching) + '\n')
        for box in plaque.boxes:
            txt_file.write(str(getboxcenter(box)) + ' ' + str(box[5]) + '\n')
        txt_file.write('\n')
    txt_file.write('Predicted Results\n')
    for plaque in list2:
        txt_file.write(str(plaque.pos) + ' ' + str(plaque.plaque_type()) + ' ' + str(plaque.max_matching) + '\n')
        for box in plaque.boxes:
            txt_file.write(str(getboxcenter(box)) + ' ' + str(box[5]) + '\n')
        txt_file.write('\n')
    txt_file.close()


def issame_pos(pt, gt):
    if pt.pos == gt.pos:
        return True
    return False


def issame_type(pt, gt):
    if PLAQUE_COMPARISON_MODE == 1:
        if pt.plaque_type() == gt.plaque_type():
            return True
        if pt.plaque_type() != 'ncP' and gt.plaque_type() != 'ncP':
            return True
        return False
    elif PLAQUE_COMPARISON_MODE == 2:
        diff = abs(pt.calcium_percentage() - gt.calcium_percentage())
        if diff < MAX_CAL_PERCENTAGE_DIFF:
            return True


def segment_based_print(pt, gt, writeout_path, patient_name):
    set_pt = set([])
    set_gt = set([])
    for plaque in pt:
        if plaque.plaque_type() == 'ncP':
            mark = 1
        else:
            mark = 0
        set_pt.add(plaque.pos + str(mark))
    #print 'set_pt\n', set_pt
    for plaque in gt:
        if plaque.plaque_type() == 'ncP':
            mark = 1
        else:
            mark = 0
        set_gt.add(plaque.pos + str(mark))
    #print 'set_gt\n', set_gt
    tp = set_gt & set_pt
    fp = set_pt - tp
    fn = set_gt - tp
    tn = set_class - (set_pt | set_gt)
    txt_name = patient_name + '_segment.txt'
    txt_path = os.path.join(writeout_path, txt_name)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    txt_file = open(txt_path, 'a')
    txt_file.write('tp_count = ' + str(len(tp)) + ' ')
    txt_file.write('fp_count = ' + str(len(fp)) + ' ')
    txt_file.write('tn_count = ' + str(len(tn)) + ' ')
    txt_file.write('fn_count = ' + str(len(fn)) + '\n')
    conclusion_seg.append([patient_name, len(tp), len(fp), len(tn), len(fn)])
    sensitivity = float(len(tp) / (len(set_gt) + 0.00000001))
    specificity = float(len(tn) / (len(set_class - set_gt) + 0.00000001))
    fp_rate = float(len(fp) / (len(tp) + 0.00000001))
    txt_file.write('sensitivity = ' + str(sensitivity) + ' ')
    txt_file.write('specificity = ' + str(specificity) + ' ')
    txt_file.write('fp/tp = ' + str(fp_rate) + '\n')
    txt_file.write('Ground Truth\n')
    for segment in set_gt:
        if segment[-1] == '1':
            local_res = segment[:-1] + '_ncP'
        else:
            local_res = segment[:-1] + '_mP'
        txt_file.write(local_res + '\n')
    txt_file.write('\nPredicted Results\n')
    for segment in set_pt:
        if segment[-1] == '1':
            local_res = segment[:-1] + '_ncP'
        else:
            local_res = segment[:-1] + '_mP'
        txt_file.write(local_res + '\n')
    txt_file.close()


def iterate_over_patients(anno_root_path_gt, anno_root_path_pt, writeout_path):
    '''
    #iterate over all patients and compare their plaques in ground truth and predicted results

    :param anno_root_path_gt:
    :param anno_root_path_pt:
    :param cls_xmlpath:
    :return:
    '''
    for patient_name in os.listdir(anno_root_path_gt):
        anno_path_pt = os.path.join(anno_root_path_pt, patient_name)
        anno_path_gt = os.path.join(anno_root_path_gt, patient_name)
        if os.path.exists(anno_path_pt):
            plaques_pt = boxesinto_plaques(anno_path_pt, 'pt')
        else:
            plaques_pt = []
        if os.path.exists(anno_path_gt):
            plagues_gt = boxesinto_plaques(anno_path_gt, 'gt')
        else:
            plagues_gt = []
        for pt in plaques_pt:
            for gt in plagues_gt:
                if pt.res is None and gt.res is None:
                    if issame_pos(pt, gt) and issame_type(pt, gt):
                        #Add another layer for intersection volume calculation for 1-box plaques
                        npt = deepcopy(pt)
                        ngt = deepcopy(gt)
                        npt.boxes.append(pt.boxes[-1])
                        npt.boxes[-1][5] += 1
                        ngt.boxes.append(gt.boxes[-1])
                        ngt.boxes[-1][5] += 1
                        if not(ngt.boxes[-1][5] <= npt.boxes[0][5] or npt.boxes[-1][5] <= ngt.boxes[0][5]):
                            intersect = npt.intersection_vol(ngt)
                            #metric for degree of similarity
                            similarity_percentage = (intersect/npt.volume() + intersect/ngt.volume())/2
                            pt.max_matching = max(pt.max_matching, similarity_percentage)
                            gt.max_matching = max(gt.max_matching, similarity_percentage)
                            if similarity_percentage > VOL_THRESHOLD:
                                pt.res = 'tp'
                                gt.res = 'tp'
        tp_count = fp_count = fn_count = 0
        tp_box_count = fp_box_count = 0
        for pt in plaques_pt:
            if pt.res is None:
                pt.res = 'fp'
                fp_count += 1
                fp_box_count += pt.noofboxes()
                specific_stats[pt.pos][2] += 1
                specific_stats[pt.plaque_type()][2] += 1
            if pt.res == 'tp':
                tp_count += 1
                tp_box_count += pt.noofboxes()
        for gt in plagues_gt:
            specific_stats[gt.pos][0] += 1
            specific_stats[gt.pos][1] += 1
            specific_stats[gt.plaque_type()][0] += 1
            specific_stats[gt.plaque_type()][1] += 1
            if gt.res is None:
                gt.res = 'fn'
                fn_count += 1
                specific_stats[gt.pos][0] -= 1
                specific_stats[gt.plaque_type()][0] -= 1

        segment_based_print(plaques_pt, plagues_gt, writeout_path, patient_name)
        conclusion.append([patient_name, tp_count, fp_count, fn_count, tp_box_count, fp_box_count])
        xlsx_name = patient_name + '.xlsx'
        print_result(plagues_gt, plaques_pt, xlsx_name, writeout_path)


def print_conclusion(output_path, conc):
    dfn = pd.DataFrame(conc, columns=
        ['PatientID', 'tp_count', 'fp_count', 'fn_count', 'tp_box_count', 'fp_box_count'])
    dfn.sort_values(by=['PatientID'])
    df1 = dfn.agg(['sum','mean'])
    df = pd.concat([dfn, df1])
    df['Sensitivity'] = df.apply(lambda row: float(row.tp_count / max(row.tp_count + row.fn_count, 1)), axis=1)
    df['fp/tp'] = df.apply(lambda row: float(row.fp_count / max(row.tp_count, 1)), axis=1)
    df['fp/tp_box'] = df.apply(lambda row:
                               float(row.fp_box_count / max(row.tp_box_count, 1)), axis=1)
    recall_plq = df.at['sum', 'Sensitivity']
    fp_plq = df.at['sum', 'fp/tp']
    fp_box = df.at['sum', 'fp/tp_box']
    print 'Plaque: recall = ', recall_plq, ' fp/tp = ', fp_plq, ' fp/tp(boxes) = ', fp_box
    writer2 = pd.ExcelWriter(output_path)
    df.to_excel(writer2, 'sheet1')
    writer2.save()


def print_conclusion2(output_path, conc):
    dfn = pd.DataFrame(conc, columns=
        ['PatientID', 'tp_count', 'fp_count', 'tn_count', 'fn_count'])
    dfn.sort_values(by=['PatientID'])
    df1 = dfn.agg(['sum','mean'])
    df = pd.concat([dfn, df1])
    df['Sensitivity'] = df.apply(lambda row : float(row.tp_count/ max(row.tp_count + row.fn_count, 1)), axis = 1)
    df['specificity'] = df.apply(lambda  row : float(row.tn_count/ max(row.tn_count + row.fp_count, 1)), axis = 1)
    df['fp/tp'] = df.apply(lambda row: float(row.fp_count / max(row.tp_count, 1)), axis=1)
    recall_plq = df.at['sum', 'Sensitivity']
    fp_plq = df.at['sum', 'fp/tp']
    print 'Segment: recall = ', recall_plq, ' fp/tp = ', fp_plq
    writer2 = pd.ExcelWriter(output_path)
    df.to_excel(writer2, 'sheet1')
    writer2.save()


def print_specific(output_path, stats_dict):
    spec_conc = []
    all_class = set_class2
    all_class.extend(['cP', 'ncP', 'mP'])
    for item in all_class:
        spec_conc.append([item, stats_dict[item][0], stats_dict[item][1], stats_dict[item][2],
                          float(stats_dict[item][0]) / max(stats_dict[item][1], 1),
                          float(stats_dict[item][2]) / max(stats_dict[item][0], 1)])
    df = pd.DataFrame(spec_conc, columns=
                      ['Seg/Type', 'tp_count', 'total_count', 'fp_count', 'recall', 'fp/tp'])
    writer = pd.ExcelWriter(output_path)
    df.to_excel(writer, 'sheet1')
    writer.save()


def parse_args():
    parser = argparse.ArgumentParser(description='Infervision plaque auto test')
    parser.add_argument('--xlsx_save_dir',
                        help='output dir for saving xlsx for each patient',
                        default='/home/tx-deepocean/Desktop/tmp2/compare_result',
                        type=str)
    parser.add_argument('--conclusion_xlsx_path',
                        help='output dir for saving total conclusion xlsx',
                        default='/home/tx-deepocean/Desktop/tmp2/conlusion_result.xlsx',
                        type=str)
    parser.add_argument('--anno_root_path_pt',
                        help='input dir of predict xml path',
                        default='/home/tx-deepocean/Desktop/tmp2/almost_label_768_50_cleanedtwice2_withoutmerge_addprob_plaque_merge',
                        type=str)
    parser.add_argument('--anno_root_path_gt',
                        help='input dir of groundtruth xml path',
                        default='/home/tx-deepocean/Desktop/tmp2/third_anno_new_79',
                        type=str)
    args = parser.parse_args()
    return args


def auto_test(anno_root_path_pt, anno_root_path_gt, writeout_path, output_path):
    clean_data(anno_root_path_gt)
    clean_data(anno_root_path_pt)
    if not os.path.exists(writeout_path):
        os.mkdir(writeout_path)
    iterate_over_patients(anno_root_path_gt, anno_root_path_pt, writeout_path)
    print_conclusion(output_path, conclusion)
    output_path2 = output_path.rsplit('/', 1)[0] + '/conclusion_segement.xlsx'
    print_conclusion2(output_path2, conclusion_seg)
    output_path3 = output_path.rsplit('/', 1)[0] + '/specific_stats.xlsx'
    print_specific(output_path3, specific_stats)


if __name__ == '__main__':
    args = parse_args()
    auto_test(anno_root_path_pt=args.anno_root_path_pt,
              anno_root_path_gt=args.anno_root_path_gt,
              writeout_path=args.xlsx_save_dir,
              output_path=args.conclusion_xlsx_path)