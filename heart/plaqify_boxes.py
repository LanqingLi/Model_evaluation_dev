# coding: utf-8

# This file takes in original unmerged boxes, preferably SSD boxes with low threshold on probability, and merge
# boxes according to plaque structure. This method differs from the original nms as nms only considers information on
# that particular layer, while this method considers information from previous layers. After adding boxes to plaques
# from previous layers, boxes on current layer that have miov greater than MIOV_MAX would be deleted.
#
# After all the plaques are found, the position of each plaque will be standardised to its most appeared position.
# If the reliability of a plaque is too low, it would be deleted. The realibility would be evaluated from prior
# plaque distribution from train data.

import heapq
import xml.etree.cElementTree as ET
import os
from private_config import *
from common.threeD_test_tools import *
import shutil


def get_pos_type(box_label):
    '''
    #from file name get its blood vessel name and its plaque type
    :param box_label:
    :return:
    '''
    [pos, ptype] = box_label.rsplit('_', 1)
    return pos, ptype


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


def read_prob_xml(xml_name, class_dict, restrict_name_list=None, check=None):
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
        if check == 'gt':
            if class_dict[name] in restrict_name_list:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                prob = 1.0
                box = [x1, y1, x2, y2, class_dict[name], sid, prob]
                box_all.append(box)
        if check == 'pt':
            if name in restrict_name_list:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                if bbox.find('prob') is not None:
                    prob = float(bbox.find('prob').text)
                else:
                    prob = 1.0
                box = [x1, y1, x2, y2, name, sid, prob]
                box_all.append(box)
    return box_all


class Plaque(ThreedObject):

    def __init__(self):
        ThreedObject.__init__(self)
        self.ptype = {'cP': 0, 'mP': 0, 'ncP': 0}
        self.pos = None

    def add_box(self, box):
        self.label[box[4]] = self.label.get(box[4], 0) + 1
        self.boxes.append(box)
        sorted(self.boxes, key=lambda x: x[5])
        bpos, btype = get_pos_type(box[4])
        self.ptype[btype] += 1

    def calcium_percentage(self):
        return (self.ptype['cP'] + 0.5 * float(self.ptype['mP'])) / max(float(self.noofboxes()), 1.0) * 100

    def plaque_type(self):
        if self.ptype['mP'] > 0:
            return 'mP'
        if self.ptype['cP'] > 0 and self.ptype['ncP'] > 0:
            return 'mP'
        if self.ptype['cP'] > 0:
            return 'cP'
        return 'ncP'


def isappro_samep(plaque, box):
    p_pos, p_type = get_pos_type(plaque.btm_box()[4])
    b_pos, b_type = get_pos_type(box[4])
    if not p_pos == b_pos:
        return False
    if calc_box_distance(plaque.btm_box(), box) > MAX_BOX_DISTANCE:
        return False
    return  True


# i_w, a_w and t_w are weights of each variable in the calculation of this index
def similarity_index(plaque, box):
    dist = calc_box_distance(box, plaque.btm_box())
    dist_p = max(0, 20-dist)/20
    iouh = iou(plaque.btm_box(), box)
    accuracy = box[6]
    box_pos, box_type = get_pos_type(box[4])
    btm_pos, btm_type = get_pos_type(plaque.btm_box()[4])
    if box_type == 'mP':
        type_ratio = float(2 * min(plaque.ptype['cP'], plaque.ptype['ncP']) + plaque.ptype['mP'] + 1.0) \
                     / (float(plaque.noofboxes()) + 1)
    else:
        type_ratio = (1.0 + float(plaque.ptype[box_type])) / (float(plaque.noofboxes()) + 1)
    pos_bonus = 1
    if box_pos == btm_pos:
        pos_bonus = POS_BONUS
    ansh = pow(iouh, I_W) * pow(accuracy, A_W) * pow(type_ratio, T_W)
    ansh = pow(ansh, 1.0/(I_W + A_W + T_W)) * pos_bonus
    return ansh


def del_sim_boxes(box_list, box):
    new_box_list = []
    for other_box in box_list:
        if not other_box == box and miov(other_box, box) < MIOV_MAX:
            new_box_list.append(other_box)
    return new_box_list


def find_most_likely(box_list):
    pmax = 0
    for box in box_list:
        if box[6] > pmax:
            pmax = box[6]
            resbox = box
    return resbox


def is_reliable_plaque(plaque):
    # calculate mean probability of the top half boxes.
    k = (plaque.noofboxes() + 1) / 2
    k_pq = []
    heapq.heapify(k_pq)
    for box in plaque.boxes:
        heapq.heappush(k_pq, box[6])
        if len(k_pq) > k:
            heapq.heappop(k_pq)
    sum = 0
    for i in k_pq:
        sum += i
    accuracy = sum / k
    accuracy = pow(accuracy, float(max(1, 5 - plaque.noofboxes())))
    if accuracy < MIN_ACCURACY:
        return False
    return True


def plaqify(patient_path, check):
    '''
    from xml directory produce a set of
    :param patient_path:
    :param check:
    :param class_list:
    :param class_dict:
    :return:
    '''
    que_plaque = []
    plaque_list = []
    directory = os.listdir(patient_path)
    directory.sort()
    for path in directory:
        xml_path = os.path.join(patient_path, path)
        boxes_curslide = read_prob_xml(xml_path, class_dict, class_list, check)
        pid, sid = get_pid_sid(xml_path)
        que_plaque_inter = []
        for plaque in que_plaque:
            # check if all plaques in queue are close enough to this layer
            if plaque.btm_box()[5] < sid - LAYER_TOLERANCE:
                if plaque not in plaque_list:
                    plaque_list.append(plaque)
            # else find the box that suits this plaque.
            else:
                que_plaque_inter.append(plaque)
                right_box = None
                sim_max = 0
                for box in boxes_curslide:
                    if isappro_samep(plaque, box):
                        sim = similarity_index(plaque, box)
                        if sim > sim_max:
                            sim2 = 0
                            for plaque2 in que_plaque:
                                if not (plaque2 == plaque) and isappro_samep(plaque2, box):
                                    sim2 = max(similarity_index(plaque2, box), sim2)
                            if sim2 < sim:
                                sim_max = sim
                                right_box = box
                if right_box is not None:
                    plaque.add_box(right_box)
                    boxes_curslide = del_sim_boxes(boxes_curslide, right_box)
        que_plaque = que_plaque_inter
        # Create new plaques from remaining boxes.
        while len(boxes_curslide) > 0:
            box = find_most_likely(boxes_curslide)
            new_plaque = Plaque()
            new_plaque.add_box(box)
            que_plaque.append(new_plaque)
            boxes_curslide = del_sim_boxes(boxes_curslide, box)


    plaque_list.extend(que_plaque)
    final_list = []
    for plaque in plaque_list:
        if is_reliable_plaque(plaque):
            # Make sure every box is in the same segment
            seg_dict = {}
            for box in plaque.boxes:
                box_pos, box_type = get_pos_type(box[4])
                seg_dict[box_pos] = seg_dict.get(box_pos, 0) + box[6]
            max_num = 0
            ans_seg = None
            for key, value in seg_dict.iteritems():
                if value > max_num:
                    max_num = value
                    ans_seg = key
            plaque.pos = ans_seg
            for box in plaque.boxes:
                new_postype = ans_seg + '_' + box[4].split('_')[-1]
                if not box[4] == new_postype:
                    box[4] = new_postype
            final_list.append(plaque)
    return final_list
