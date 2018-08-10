# coding: utf-8
#

import xlrd
import xml.etree.ElementTree as ET
import os
import shutil


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


def generate_xml(box, xml_path):
    '''
    write boxes to xmls
    :param all_box: could be a list of boxes length:[4] or length: [5] (append class idx)
    :param path: save xml prefix_dir
    :param basename: save xml filename,save_total_path = path + basename + '.xml'e
    :return: None
    '''
    root = ET.Element("annotation")
    filename_doc = ET.SubElement(root, "filename")
    filename_doc.text = xml_path.split('/')[-1].replace('xml','jpg')
    size_doc = ET.SubElement(root, "size")
    segment_doc = ET.SubElement(root, "segmented")
    segment_doc.text = '0'
    append_object(box, root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


def append_object(bbox, root):
    object_doc = ET.SubElement(root, "object")
    ET.SubElement(object_doc, "name").text = str(bbox[4])
    bndbox_doc = ET.SubElement(object_doc, "bndbox")
    ET.SubElement(bndbox_doc, "xmin").text = str(bbox[0])
    ET.SubElement(bndbox_doc, "ymin").text = str(bbox[1])
    ET.SubElement(bndbox_doc, "xmax").text = str(bbox[2])
    ET.SubElement(bndbox_doc, "ymax").text = str(bbox[3])
    ET.SubElement(bndbox_doc, "prob").text = str(bbox[6])


def append_xml(box, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    append_object(box, root)
    tree.write(xml_path)


def boxesto_xml(box_list, xml_path):
    if os.path.exists(xml_path):
        os.remove(xml_path)
    for box in box_list:
        if os.path.exists(xml_path):
            append_xml(box, xml_path)
        else:
            generate_xml(box, xml_path)


def patientto_xml(box_list, patient_dir, patient_name):
    '''
    given a list of boxes for a patient. Delete previous results if there are such and write box_list to it.
    :param box_list:
    :param patient_dir:
    :return:
    '''
    if os.path.exists(patient_dir):
        shutil.rmtree(patient_dir)
    os.mkdir(patient_dir)
    for box in box_list:
        layerID = box[5]
        xml_name = patient_name + '_' + '{:03}'.format(layerID) + '.xml'
        xml_path = os.path.join(patient_dir, xml_name)
        if os.path.exists(xml_path):
            append_xml(box, xml_path)
        else:
            generate_xml(box, xml_path)
