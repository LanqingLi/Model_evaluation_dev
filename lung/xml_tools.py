# ------ coding: utf-8 --------
import os
import xml.etree.cElementTree as ET
from lung.config import config


# 输入一个xml文件(某个slice), 输出[[bbx1],[bbx2],[bbx3]]
# ['nodule', '0-3nodule', '3-6nodule', '6-10nodule',
#                     'calcific nodule', 'pgo', '10-30nodule', 'mass',
#                     '0-5GGN', '5GGN', 'calcific',
#                     'pleural nodule', 'quasi-nodule']
def read_xml(xml_path, restrict_name_list=None):
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


def read_xml_for_multi_classes(xml_path, restrict_name_list=None):
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


def generate_xml(filename, name_bboxs_dict):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename")
    for name, bboxList in name_bboxs_dict.items():
        for bbox in bboxList:
            object_doc = ET.SubElement(root, "object")
            ET.SubElement(object_doc, "name").text = name
            bndbox_doc = ET.SubElement(object_doc, "bndbox")
            ET.SubElement(bndbox_doc, "xmin").text = str(int(bbox[0]))
            ET.SubElement(bndbox_doc, "ymin").text = str(int(bbox[1]))
            ET.SubElement(bndbox_doc, "xmax").text = str(int(bbox[2]))
            ET.SubElement(bndbox_doc, "ymax").text = str(int(bbox[3]))
    tree = ET.ElementTree(root)
    tree.write(filename)






def xml_to_boxeslist_multi_classes(xml_dir, box_length):
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
        for i_cls in range(num_classes):
            slice_boxes.append([])
        return_boxes_list.append(slice_boxes)
    if xml_dir != None:
        # 读取xml，找到对应层面，存入对应种类的list
        xml_names = os.listdir(xml_dir)
        for xml_name in xml_names:
            slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
            xml_path = xml_dir + "/" + xml_name
            boxes = read_xml_for_multi_classes(xml_path,config.NODULE_CLASSES)
            for box in boxes:
                name = box[-1]
                id_cls = config.NODULE_CLASSES.index(name)
                return_boxes_list[slice_id][id_cls].append(box[:-1])
    return return_boxes_list

def xml_to_boxeslist(xml_dir, box_length):
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
            xml_path = xml_dir + "/" + xml_name
            boxes = read_xml(xml_path, config.CLASSES)
            for box in boxes:
                name = box[-1]
                id_cls = config.CLASSES.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:-1])
    return return_boxes_list