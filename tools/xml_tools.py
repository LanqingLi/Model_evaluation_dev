import xml.etree.cElementTree as ET
import os

def read_xml_2d(xml_path, cls_dict, if_gt=True):
    '''
    读取xml
    :param xml_path: xml文件的路径
    :return: boxes的list，每个box [xmin,ymin,xmax,ymax,概率（gt为1）,anchor种类]
    '''
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    boxes = []
    for index, obj in enumerate(objs):
        name = obj.find('name').text
        try:
            mapped_name = cls_dict[name]
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            if if_gt:
                box = [x1, y1, x2, y2, 1., mapped_name]
            else:
                prob = float(bbox.find('prob').text)
                box = [x1, y1, x2, y2, prob, mapped_name]
            boxes.append(box)
        except:
            pass
    return boxes


def xml_to_boxeslist_2d(xml_dir, cls_list, cls_dict, box_length, if_gt=True):
    '''
    读取文件夹下的xml，返回和get_boxes返回格式一样的boxes
    :param xml_dir:
    :param cls_list:
    :param cls_dict:
    :param box_length:
    :return: box list 一个四层的list,第一层id为层面号,第二层id为结节种类，第三层id为bndbox_id,第四层为[xmin,ymin,xmax,ymax,prob]
    例如，box_list[123][1][3] = 表示的是第124层的，大肿块（加入2号类为大肿块）中，的第四个大肿块，的bndbox信息
    '''
    return_boxes_list = []
    num_classes = len(cls_list)  # cls_list有background, 所以要-1
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
            boxes = read_xml_2d(xml_path, cls_dict, if_gt=if_gt)
            for box in boxes:
                name = box[-1]
                id_cls = cls_list.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:-1])
    return return_boxes_list