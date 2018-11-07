# ------ coding: utf-8 --------
import os
import sys
import xml.etree.cElementTree as ET
from objmatch.object import Anchor
import numpy as np


# 输入一个xml文件(某个slice), 输出[[bbx1],[bbx2],[bbx3]]
# ['nodule', '0-3nodule', '3-6nodule', '6-10nodule',
#                     'calcific nodule', 'pgo', '10-30nodule', 'mass',
#                     '0-5GGN', '5GGN', 'calcific',
#                     'pleural nodule', 'quasi-nodule']
def read_xml(config, xml_path, restrict_name_list=None):
    '''
    读取xml
    :param xml_path: xml文件的路径
    :param restrict_name_list: 结节种类表（list），一般为config.CLASSES
    :return:boxes的list，每个box [xmin,ymin,xmax,ymax,概率（gt为1）,结节种类]
    '''
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    boxes = []
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        if restrict_name_list == None:
            restrict_name_list = ['nodule', '0-3nodule', '3-6nodule', '6-10nodule',
                                  'calcific nodule', 'pgo', '10-30nodule', 'mass',
                                  '0-5GGN', '5GGN', 'calcific',
                                  'pleural nodule', 'quasi-nodule']
        try:
            mapped_name = config.CLASS_DICT[name]
            if mapped_name in restrict_name_list:
                # ['0-5GGN', '5GGN']:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                box = [x1, y1, x2, y2, 1, mapped_name]
                boxes.append(box)
        except:
            pass
    return boxes

def read_xml_anchor(config, xml_path, value_list, key_list, restrict_name_list=None):
    '''
    读取xml
    :param config: 配置参数类，定义在model_eval.lung.config中
    :param xml_path: xml文件的路径
    :param value_list: anchor所包含属性取值的列表
    :param key_list: anchor所包含属性关键词的列表
    :param restrict_name_list: 需要统计的anchor种类表（list），一般为config.CLASSES
    :return anchor的list，每个anchor为一个类，定义在objmatch.object中
    '''
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    anchor_list = []
    assert config.ANCHOR.CLASS_KEY in config.ANCHOR.KEY_LIST, 'keyword arguments must contain the object class key: %s' %config.ANCHOR.CLASS_KEY
    for ix, obj in enumerate(objs):
        values = []
        for key in config.ANCHOR.KEY_LIST:
            # in case the xml doesnt contain certain keyword, pass the assignment of the corresponding value
            try:
                values.append(obj.find(key).text)
            except:
                values.append(None)
        if restrict_name_list == None:
            restrict_name_list = ['nodule', '0-3nodule', '3-6nodule', '6-10nodule',
                                  'calcific nodule', 'pgo', '10-30nodule', 'mass',
                                  '0-5GGN', '5GGN', 'GGN', 'calcific nodule', 'solid nodule',
                                  'pleural nodule', 'quasi-nodule']
        try:
            mapped_name = config.CLASS_DICT[obj.find(config.ANCHOR.CLASS_KEY).text]
            if mapped_name in restrict_name_list:
                bbox = obj.find(config.ANCHOR.BNDBOX_KEY)
                coord_list = [float(bbox.find(key).text) for key in config.ANCHOR.BNDBOX_KEY_LIST]
                # add 'prob' keyword, the corresponding Anchor.attr_dict['prob'] = 1. for ground truth labels
                values = coord_list + values + value_list
                keys = config.ANCHOR.BNDBOX_KEY_LIST + config.ANCHOR.KEY_LIST + key_list
                anchor = Anchor(coord_list, values, keys)
                anchor.attr_dict[config.ANCHOR.CLASS_KEY] = config.CLASS_DICT[anchor.attr_dict[config.ANCHOR.CLASS_KEY]]
                anchor_list.append(anchor)
        except:
            print "config.CLASS_DICT doesn't contain labeled class %s" %(obj.find(config.ANCHOR.CLASS_KEY).text)

    return anchor_list

def read_xml_anchor_multi_classes(config, xml_path, value_list, key_list, restrict_name_list=None):
    '''
    读取xml
    :param config: 配置参数类，定义在model_eval.lung.config中
    :param xml_path: xml文件的路径
    :param value_list: anchor所包含属性取值的列表
    :param key_list: anchor所包含属性关键词的列表
    :param restrict_name_list: 需要统计的anchor种类表（list），一般为config.CLASSES
    :return anchor的list，每个anchor为一个类，定义在objmatch.object中
    '''
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    anchor_list = []
    assert config.ANCHOR.CLASS_KEY in config.ANCHOR.KEY_LIST, 'keyword arguments must contain the object class key: %s' %config.ANCHOR.CLASS_KEY
    for ix, obj in enumerate(objs):
        values = []
        for key in config.ANCHOR.KEY_LIST:
            # in case the xml doesnt contain certain keyword, pass the assignment of the corresponding value
            try:
                values.append(obj.find(key).text)
            except:
                values.append(None)
        if restrict_name_list == None:
            restrict_name_list = ['0-3nodule', '3-6nodule', '6-10nodule', '10-30nodule',
                                  'calcific nodule', 'mass', '0-5pGGN', '0-5mGGN', '5mGGN', '5pGGN',
                                  '0-5GGN', '5GGN', 'GGN', 'pleural nodule', 'suspect GGN', 'suspect nodule',
                                  'pleural calcific nodule']
        if obj.find(config.ANCHOR.CLASS_KEY).text in restrict_name_list:
            bbox = obj.find(config.ANCHOR.BNDBOX_KEY)
            coord_list = [float(bbox.find(key).text) for key in config.ANCHOR.BNDBOX_KEY_LIST]
            # add 'prob' keyword, the corresponding Anchor.attr_dict['prob'] = 1. for ground truth labels
            values = coord_list + values + value_list
            keys = config.ANCHOR.BNDBOX_KEY_LIST + config.ANCHOR.KEY_LIST + key_list
            anchor = Anchor(coord_list, values, keys)
            anchor_list.append(anchor)
    return anchor_list

def read_xml_without_nodule_cls(config, xml_path, restrict_name_list=None):
    '''
    读取xml
    :param xml_path: xml文件的路径
    :param restrict_name_list: 结节种类表（list），一般为config.CLASSES
    :return:boxes的list，每个box [xmin,ymin,xmax,ymax,概率（gt为1）,结节种类]
    '''
    # 如果用不了请联系ylifeng帮你改
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    boxes = []
    for ix, obj in enumerate(objs):
        # no nodule class info in anno, all set to nodule by default
        name = '0-3nodule'
        if restrict_name_list == None:
            restrict_name_list = ['nodule', '0-3nodule', '3-6nodule', '6-10nodule',
                                  'calcific nodule', 'pgo', '10-30nodule', 'mass',
                                  '0-5GGN', '5GGN', 'calcific',
                                  'pleural nodule', 'quasi-nodule']
        try:
            mapped_name = config.CLASS_DICT[name]
            if mapped_name in restrict_name_list:
                # ['0-5GGN', '5GGN']:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                box = [x1, y1, x2, y2, 1., name]
                boxes.append(box)
        except:
            pass
    return boxes

def read_xml_with_nodule_num(config, xml_path, xml_name, restrict_name_list=None):
    '''
    读取xml
    :param xml_path: xml文件的路径
    :param restrict_name_list: 结节种类表（list），一般为config.CLASSES
    :return:boxes的list，每个box [xmin,ymin,xmax,ymax,概率（gt为1）,结节种类,结节编号，层面数]
    '''
    # 如果用不了请联系ylifeng帮你改
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    boxes = []
    slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        nodule_num = int(obj.find('lobe_pos').text)
        if restrict_name_list == None:
            restrict_name_list = ['nodule', '0-3nodule', '3-6nodule', '6-10nodule',
                                  'calcific nodule', 'pgo', '10-30nodule', 'mass',
                                  '0-5GGN', '5GGN', 'calcific',
                                  'pleural nodule', 'quasi-nodule']
        try:
            mapped_name = config.CLASS_DICT[name]
            if mapped_name in restrict_name_list:
                # ['0-5GGN', '5GGN']:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                box = [x1, y1, x2, y2, 1, mapped_name, nodule_num, slice_id]
                boxes.append(box)
        except:
            pass
    return boxes

def read_xml_with_nodule_num_without_nodule_cls(config, xml_path, xml_name, restrict_name_list=None):
    '''
    读取xml
    :param xml_path: xml文件的路径
    :param restrict_name_list: 结节种类表（list），一般为config.CLASSES
    :return:boxes的list，每个box [xmin,ymin,xmax,ymax,概率（gt为1）,结节种类,结节编号，层面数]
    '''
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    boxes = []
    slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
    for ix, obj in enumerate(objs):
        name = '0-3nodule'
        nodule_num = int(obj.find('lobe_pos').text)
        if restrict_name_list == None:
            restrict_name_list = ['nodule', '0-3nodule', '3-6nodule', '6-10nodule',
                                  'calcific nodule', 'pgo', '10-30nodule', 'mass',
                                  '0-5GGN', '5GGN', 'calcific',
                                  'pleural nodule', 'quasi-nodule']
        try:
            mapped_name = config.CLASS_DICT[name]
            if mapped_name in restrict_name_list:
                # ['0-5GGN', '5GGN']:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                box = [x1, y1, x2, y2, 1, mapped_name, nodule_num, slice_id]
                boxes.append(box)
        except:
            pass
    return boxes



def read_xml_for_multi_classes(xml_path, restrict_name_list=None):
    '''
    读取xml
    :param xml_path: xml文件的路径
    :param restrict_name_list: 结节种类表（list），一般为config.CLASSES
    :return:boxes的list，每个box [xmin,ymin,xmax,ymax,概率（gt为1）,结节种类]
    '''
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    boxes = []
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        if restrict_name_list == None:
            restrict_name_list = ['0-3nodule', '3-6nodule', '6-10nodule',
                                  'calcific nodule','10-30nodule', 'mass',
                                  '0-5GGN', '5GGN','pleural nodule']
        try:
            if name in restrict_name_list:
                # ['0-5GGN', '5GGN']:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                box = [x1, y1, x2, y2, 1, name]
                boxes.append(box)
        except:
            pass
    return boxes


def generate_xml(data_dir, filename, name_bboxs_dict):
    root = ET.Element("annotation")
    filename_doc = ET.SubElement(root, "filename")
    filename_doc.text = filename
    for index, row in name_bboxs_dict.iterrows():
        object_doc = ET.SubElement(root, "object")
        ET.SubElement(object_doc, "name").text = row['nodule_class']
        bndbox_doc = ET.SubElement(object_doc, "bndbox")
        ET.SubElement(bndbox_doc, "xmin").text = str(int(round(row['xmin'])))
        ET.SubElement(bndbox_doc, "ymin").text = str(int(round(row['ymin'])))
        ET.SubElement(bndbox_doc, "xmax").text = str(int(round(row['xmax'])))
        ET.SubElement(bndbox_doc, "ymax").text = str(int(round(row['ymax'])))
        ET.SubElement(bndbox_doc, "prob").text = str(row['prob'])
    tree = ET.ElementTree(root)
    tree.write(os.path.join(data_dir, filename))

def xml_to_anchorlist_multi_classes(config, xml_dir):
    '''
    读取文件夹下的xml，返回一个以层面数排列的anchor的列表
    :param config: 配置参数类，定义在model_eval.lung.config中
    :param xml_dir: 包含xml文件的路径
    :return: 一个以层面数排列的列表，每个元素是对应层面的所有anchor
    '''
    return_anchors_list = []
    slice_id_list = []
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)

    for xml_name in xml_names:
        slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
        slice_id_list.append(slice_id)

    slice_id_max = max(slice_id_list)

    for i_row in range(slice_id_max + 1):
        return_anchors_list.append([])

    for xml_name in xml_names:
        slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
        xml_path = os.path.join(xml_dir, xml_name)
        anchors = read_xml_anchor_multi_classes(config, xml_path, config.ANCHOR.ADD_VALUE + [slice_id], config.ANCHOR.ADD_KW + ['sliceId'])
        return_anchors_list[slice_id] = anchors
    return return_anchors_list

def xml_to_anchorlist(config, xml_dir):
    '''
    读取文件夹下的xml，返回一个以层面数排列的anchor的列表
    :param config: 配置参数类，定义在model_eval.lung.config中
    :param xml_dir: 包含xml文件的路径
    :return: 一个以层面数排列的列表，每个元素是对应层面的所有anchor
    '''
    return_boxes_list = []
    slice_id_list = []
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)

    for xml_name in xml_names:
        slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
        slice_id_list.append(slice_id)

    slice_id_max = max(slice_id_list)

    for i_row in range(slice_id_max + 1):
        return_boxes_list.append([])

    for xml_name in xml_names:
        slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
        xml_path = os.path.join(xml_dir, xml_name)
        anchors = read_xml_anchor(config, xml_path, config.ANCHOR.ADD_VALUE + [slice_id], config.ANCHOR.ADD_KW + ['sliceId'])
        return_boxes_list[slice_id] = anchors
    return return_boxes_list

def xml_to_boxeslist_multi_classes(config, xml_dir, box_length):
    '''
    读取文件夹下的xml，返回和get_boxes返回格式一样的boxes
    :param xml_dir:
    :param box_length:
    :return: box list 一个四层的list,第一层id为层面号,第二层id为结节种类，第三层id为bndbox_id,第四层为[xmin,ymin,xmax,ymax,prob]
    例如，box_list[123][1][3] = 表示的是第124层的，大肿块（加入2号类为大肿块）中，的第四个大肿块，的bndbox信息
    '''
    return_boxes_list = []
    num_classes = len(config.NODULE_CLASSES)  # CLASSES有background, 所以要+1
    # 初始化空列表
    for i_row in range(box_length):
        slice_boxes = []
        for i_cls in range(num_classes - 1):
            slice_boxes.append([])
        return_boxes_list.append(slice_boxes)
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)
        for xml_name in xml_names:
            slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
            xml_path = os.path.join(xml_dir, xml_name)
            boxes = read_xml_for_multi_classes(xml_path,config.NODULE_CLASSES)
            for box in boxes:
                name = box[5]
                id_cls = config.NODULE_CLASSES.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:-1])
    return return_boxes_list

def xml_to_list(config, xml_dir, box_length):
    '''
    读取文件夹下的xml，返回和get_boxes返回格式一样的boxes
    :param xml_dir:
    :param box_length:
    :return: box list 一个四层的list,第一层id为层面号,第二层id为结节种类，第三层id为bndbox_id,第四层为[xmin,ymin,xmax,ymax,prob]
    例如，box_list[123][1][3] = 表示的是第124层的，大肿块（加入2号类为大肿块）中，的第四个大肿块，的bndbox信息
    '''
    return_boxes_list = []
    num_classes = len(config.CLASSES)  # CLASSES有background, 所以要-1
    # 初始化空列表
    for i_row in range(box_length):
        slice_boxes = []
        for i_cls in range(num_classes - 1):
            slice_boxes.append([])
        return_boxes_list.append(slice_boxes)
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)
        for xml_name in xml_names:
            slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
            xml_path = os.path.join(xml_dir, xml_name)
            boxes = read_xml(config, xml_path, config.CLASSES)
            for box in boxes:
                name = box[5]
                id_cls = config.CLASSES.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:5] + box[6:])
    return return_boxes_list


def xml_to_boxeslist(config, xml_dir, box_length):
    '''
    读取文件夹下的xml，返回和get_boxes返回格式一样的boxes
    :param xml_dir:
    :param box_length:
    :return: box list 一个四层的list,第一层id为层面号,第二层id为结节种类，第三层id为bndbox_id,第四层为[xmin,ymin,xmax,ymax,prob]
    例如，box_list[123][1][3] = 表示的是第124层的，大肿块（加入2号类为大肿块）中，的第四个大肿块，的bndbox信息
    '''
    return_boxes_list = []
    num_classes = len(config.CLASSES)  # CLASSES有background, 所以要-1
    # 初始化空列表
    for i_row in range(box_length):
        slice_boxes = []
        for i_cls in range(num_classes - 1):
            slice_boxes.append([])
        return_boxes_list.append(slice_boxes)
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)
        for xml_name in xml_names:
            slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
            xml_path = os.path.join(xml_dir, xml_name)
            boxes = read_xml(config, xml_path, config.CLASSES)
            for box in boxes:
                name = box[5]
                id_cls = config.CLASSES.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:5] + box[6:])
    return return_boxes_list

def xml_to_boxeslist_without_nodule_cls(config, xml_dir, box_length):
    '''
    读取文件夹下的xml，返回和get_boxes返回格式一样的boxes
    :param xml_dir:
    :param box_length:
    :return: box list 一个四层的list,第一层id为层面号,第二层id为结节种类，第三层id为bndbox_id,第四层为[xmin,ymin,xmax,ymax,prob]
    例如，box_list[123][1][3] = 表示的是第124层的，大肿块（加入2号类为大肿块）中，的第四个大肿块，的bndbox信息
    '''
    return_boxes_list = []
    num_classes = len(config.CLASSES)  # CLASSES有background, 所以要-1
    # 初始化空列表
    for i_row in range(box_length):
        slice_boxes = []
        for i_cls in range(num_classes - 1):
            slice_boxes.append([])
        return_boxes_list.append(slice_boxes)
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)
        for xml_name in xml_names:
            #print xml_name
            slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
            xml_path = os.path.join(xml_dir, xml_name)
            boxes = read_xml_without_nodule_cls(config, xml_path, config.CLASSES)
            for box in boxes:
                name = box[-1]
                id_cls = config.CLASSES.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:-1])
    return return_boxes_list

def xml_to_boxeslist_with_nodule_num(config, xml_dir, box_length):
    '''
    读取文件夹下的xml，返回和get_boxes返回格式一样的boxes
    :param xml_dir:
    :param box_length:
    :return: box list 一个四层的list,第一层id为层面号,第二层id为结节种类，第三层id为bndbox_id,第四层为[xmin,ymin,xmax,ymax,prob,nodule_num,sliceId]
    例如，box_list[123][1][3] = 表示的是第124层的，大肿块（加入2号类为大肿块）中，的第四个大肿块，的bndbox信息
    '''
    return_boxes_list = []
    box_list_all_slice = []
    num_classes = len(config.CLASSES)  # CLASSES有background, 所以要-1
    # 初始化空列表
    for i_row in range(box_length):
        slice_boxes = []
        for i_cls in range(num_classes - 1):
            slice_boxes.append([])
        return_boxes_list.append(slice_boxes)
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)
        for xml_name in xml_names:
            slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
            xml_path = os.path.join(xml_dir, xml_name)
            # nodule_num = boxes[6], nodule_class = boxes[5]
            boxes = read_xml_with_nodule_num(config, xml_path, xml_name, config.CLASSES)
            for box in boxes:
                name = box[-3]
                id_cls = config.CLASSES.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:5])
                box_list_all_slice.append(box)
    return return_boxes_list, box_list_all_slice

def xml_to_boxeslist_with_nodule_num_without_nodule_cls(config, xml_dir, box_length):
    '''
    读取文件夹下的xml，返回和get_boxes返回格式一样的boxes
    :param xml_dir:
    :param box_length:
    :return: box list 一个四层的list,第一层id为层面号,第二层id为结节种类，第三层id为bndbox_id,第四层为[xmin,ymin,xmax,ymax,prob,nodule_num,sliceId]
    例如，box_list[123][1][3] = 表示的是第124层的，大肿块（加入2号类为大肿块）中，的第四个大肿块，的bndbox信息
    '''
    return_boxes_list = []
    box_list_all_slice = []
    num_classes = len(config.CLASSES)  # CLASSES有background, 所以要-1
    # 初始化空列表
    for i_row in range(box_length):
        slice_boxes = []
        for i_cls in range(num_classes - 1):
            slice_boxes.append([])
        return_boxes_list.append(slice_boxes)
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)
        for xml_name in xml_names:
            slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
            xml_path = os.path.join(xml_dir, xml_name)
            # nodule_num = boxes[6], nodule_class = boxes[5]
            boxes = read_xml_with_nodule_num_without_nodule_cls(config, xml_path, xml_name, config.CLASSES)
            for box in boxes:
                name = box[-3]
                id_cls = config.CLASSES.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:5])
                box_list_all_slice.append(box)
    return return_boxes_list, box_list_all_slice
