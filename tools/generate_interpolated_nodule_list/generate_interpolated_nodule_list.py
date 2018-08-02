# -- coding: utf-8 --
import numpy as np
import pandas as pd
import os
import xlrd
from easydict import EasyDict as edict
import argparse
import xml.etree.cElementTree as ET
from get_df_nodules import get_nodule_stat, init_df_boxes

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

    class_dict = dict()  # label to class
    conf_thresh = dict()
    cls_weight_dict = dict()
    cls_z_threshold_pred_dict = dict()
    cls_z_threshold_gt_dict = dict()
    conf_thresh['__background__'] = 1.0
    class_list = []
    class_list.append('__background__')
    for i in range(1, classDictSheet.nrows):
        # add class name
        label_name = classDictSheet.row(i)[0].value.strip(' ')
        class_name = classDictSheet.row(i)[1].value.strip(' ')

        label_name = label_name.encode("utf-8")
        class_name = class_name.encode("utf-8")
        label_classes.append(label_name)

        class_list.append(class_name)
        class_dict[label_name] = class_name

        thresh = classDictSheet.row(i)[2].value
        assert isinstance(thresh, float), 'thresh must be float type, check xls'
        conf_thresh[class_name] = thresh

        weight = classDictSheet.row(i)[3].value
        assert isinstance(weight, float), 'weight must be float type, check xls'

        z_threshold_pred = classDictSheet.row(i)[4].value
        assert isinstance(z_threshold_pred, float), 'z_threshold_pred must be float type, check xls'

        z_threshold_gt = classDictSheet.row(i)[5].value
        assert isinstance(z_threshold_gt, float), 'z_threshold_gt must be float type, check xls'
        cls_weight_dict[class_name] = weight
        cls_z_threshold_pred_dict[class_name] = z_threshold_pred
        cls_z_threshold_gt_dict[class_name] = z_threshold_gt
    # remove repeat element in class
    class_list1 = sorted(set(class_list), key=class_list.index)

    return class_list1, label_classes, class_dict, conf_thresh, cls_weight_dict, cls_z_threshold_pred_dict, cls_z_threshold_gt_dict

config = edict()

config.CLASSES_LABELS_XLS_FILE_NAME = 'classname_labelname_mapping.xls'
config.CLASSES, config.NODULE_CLASSES, config.CLASS_DICT, config.CONF_THRESH, config.CLASS_WEIGHTS, config.CLASS_Z_THRESHOLD_PRED,\
    config.CLASS_Z_THRESHOLD_GT= get_label_classes_from_xls(config.CLASSES_LABELS_XLS_FILE_NAME)

#######################
# CONFIG FOR FIND_NODULES
#######################

config.FIND_NODULES = edict()

# 对于同一层面的不同类框，将其视为同一等价类的中心点偏移阈值，对于ground truth设为np.array([0., 0.])
config.FIND_NODULES.SAME_BOX_THRESHOLD_PRED = np.array([0.8, 0.8])
config.FIND_NODULES.SAME_BOX_THRESHOLD_GT = np.array([0., 0.])

# 对于不同层面两个框的匹配，将其视为二分图中一条边的中心点偏移阈值，对于ground truth一般应设置得更小
config.FIND_NODULES.SCORE_THRESHOLD_PRED = 0.8
config.FIND_NODULES.SCORE_THRESHOLD_GT = 1.

class NoduleListGenerator(object):
    def __init__(self, gt_anno_dir, xlsx_save_dir, xlsx_name, csv_save_dir, csv_name):
        self.gt_anno_dir = gt_anno_dir
        self.xlsx_save_dir = xlsx_save_dir
        self.xlsx_name = xlsx_name
        self.csv_save_dir = csv_save_dir
        self.csv_name = csv_name
        self.conf_thresh = 1.
        self.patient_list = []

    def generate_interpolated_nodule_list(self):
        gt_df_boxes_dict = self.load_data_xml()
        print gt_df_boxes_dict
        tot_nodule_count = 0
        gt_df_list = pd.DataFrame(columns=['bbox', 'nodule_class', 'nodule_id', 'pid', 'slice', 'prob'])
        for index, key in enumerate(gt_df_boxes_dict):
            self.patient_list.append(key)
            gt_df_boxes = gt_df_boxes_dict[key]


            print ('processing %s' % key)

            # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
            if not gt_df_boxes_dict[key].empty:
                # print gt_df_boxes
                filtered_gt_boxes = gt_df_boxes[gt_df_boxes["prob"] >= self.conf_thresh]
                filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
            else:
                filtered_gt_boxes = pd.DataFrame(
                    {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                     'nodule_class': [], 'prob': [], 'Mask': []})

            # 将预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
            print "generating ground truth nodules:"
            _, gt_df = get_nodule_stat(dicom_names=None,
                                       hu_img_array=None,
                                       return_boxes=filtered_gt_boxes,
                                       img_spacing=None,
                                       prefix=key,
                                       classes=config.CLASSES,
                                       same_box_threshold=config.FIND_NODULES.SAME_BOX_THRESHOLD_GT,
                                       score_threshold=config.FIND_NODULES.SCORE_THRESHOLD_GT,
                                       z_threshold=config.CLASS_Z_THRESHOLD_GT,
                                       if_dicom=False,
                                       focus_priority_array=None,
                                       skip_init=True)
            #print gt_df

            gt_df = gt_df.reset_index(drop=True)
            gt_df = json_df_2_df(gt_df)

            for index, row in gt_df.iterrows():
                nodule_slices, nodule_boxes = nodule_slice_interpolate_gt(row)
                #print gt_df.iloc[index]
                gt_df.loc[index]['slice'] = nodule_slices
                gt_df.loc[index]['bbox'] = nodule_boxes
                # print gt_df.loc[index]['slice']
                gt_df_list = gt_df_list.append({'bbox': gt_df.loc[index]['bbox'],
                                  'nodule_class': gt_df.loc[index]['nodule_class'],
                                  'nodule_id': gt_df.loc[index]['nodule_id'],
                                  'pid': gt_df.loc[index]['pid'],
                                  'slice': ['{}{}{}{}'.format(gt_df.loc[index]['pid'], '_', slice_num_to_three_digit_str(i), '.xml') for i in gt_df.loc[index]['slice']],
                                  'prob': gt_df.loc[index]['prob']},
                                  ignore_index=True)
                #print gt_df_list

        #print gt_df_list

        if not os.path.exists(self.xlsx_save_dir):
            os.makedirs(self.xlsx_save_dir)
        print ("saving %s" % os.path.join(self.xlsx_save_dir, self.xlsx_name))

        # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(os.path.join(self.xlsx_save_dir, self.xlsx_name)):
            os.remove(os.path.join(self.xlsx_save_dir, self.xlsx_name))

        if not os.path.exists(self.csv_save_dir):
            os.makedirs(self.csv_save_dir)
        print ("saving %s" % os.path.join(self.csv_save_dir, self.csv_name))

        # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(os.path.join(self.csv_save_dir, self.csv_name)):
            os.remove(os.path.join(self.csv_save_dir, self.csv_name))

        writer = pd.ExcelWriter(os.path.join(self.xlsx_save_dir, self.xlsx_name))
        gt_df_list.to_excel(writer, 'nodule_dataframe', index=False)
        writer.save()

        gt_df_list.to_csv(os.path.join(self.csv_save_dir, self.csv_name), index=False)


    def load_data_xml(self):
        ground_truth_boxes_dict = {}
        for PatientID in os.listdir(self.gt_anno_dir):
            ground_truth_path = os.path.join(self.gt_anno_dir, PatientID)
            # try:
            #     # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于1000个层，故我们在这里选择1000
            ground_truth_boxes = xml_to_boxeslist(ground_truth_path, 1000)
            # except:
            #     print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
            #     ground_truth_boxes = [[[[]]]]

            ground_truth_boxes = init_df_boxes(return_boxes=ground_truth_boxes, classes=config.CLASSES)
            ground_truth_boxes = ground_truth_boxes.sort_values(by=['prob'])
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            ground_truth_boxes_dict[PatientID] = ground_truth_boxes

        return ground_truth_boxes_dict

def read_xml(xml_path, xml_name, restrict_name_list=None):
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
        if restrict_name_list == None:
            restrict_name_list = ['nodule', '0-3nodule', '3-6nodule', '6-10nodule',
                                  'calcific nodule', 'pgo', '10-30nodule', 'mass',
                                  '0-5GGN', '5GGN', 'calcific',
                                  'pleural nodule', 'quasi-nodule']
        try:
            mapped_name = config.CLASS_DICT[name]
            #print mapped_name

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
            #print xml_name
            slice_id = int(xml_name.split("_")[-1].split(".")[0]) - 1
            xml_path = xml_dir + "/" + xml_name
            boxes = read_xml(xml_path, xml_name, config.CLASSES)
            for box in boxes:
                name = box[-1]
                id_cls = config.CLASSES.index(name) - 1
                return_boxes_list[slice_id][id_cls].append(box[:-1])
    return return_boxes_list

def nodule_slice_interpolate_gt(ground_truth_df_record):
    """
    :param ground_truth_df_record:一条ground truth记录，代表一个结节，DataFrame
    """

    # print  ground_truth_df_record
    ground_truth_slices = range(ground_truth_df_record['slice'][0],
                                ground_truth_df_record['slice'][-1] + 1)

    ground_truth_bboxs = []

    # 补全缺失层面的bbox
    for ground_truth_slice in ground_truth_slices:
        if not ground_truth_slice in ground_truth_df_record['slice']:
            ground_truth_bboxs.append([None, None, None, None])
        else:
            i = ground_truth_df_record['slice'].index(ground_truth_slice)
            ground_truth_bboxs.append(ground_truth_df_record['bbox'][i][:4])

    # 用插值法补全bbox
    ground_truth_bboxs = np.array(pd.DataFrame(ground_truth_bboxs).interpolate(limit_direction='both'))

    return ground_truth_slices, ground_truth_bboxs

def json_df_2_df(df):
    ret_df = pd.DataFrame({'bbox': [], 'pid': [], 'slice': [], 'nodule_class': [], 'nodule_id': []})
    for index, row in df.iterrows():
        df_add_row = {'bbox': [bbox for bbox in row['Bndbox List']],
                      'pid': row['Pid'],
                      'slice': row['SliceRange'],
                      'nodule_class': row['Type'],
                      'nodule_id': row['Nodule Id'],
                      'prob': row['prob']}
        ret_df = ret_df.append(df_add_row, ignore_index=True)
    return ret_df

def slice_num_to_three_digit_str(slice_num):
    assert isinstance(slice_num, int), 'slice_num must be an integer'
    if slice_num >= 1000:
        print 'we only consider slice num < 1000'
        return NotImplementedError
    elif slice_num <= 0:
        print 'slice num should be a positive integer'
        return ValueError
    elif slice_num >= 100:
        return str(slice_num)
    elif slice_num >= 10:
        return '{}{}'.format('0', slice_num)
    else:
        return '{}{}'.format('00', slice_num)



def parse_args():
    parser = argparse.ArgumentParser(description='Infervision auto test')
    parser.add_argument('--gt_anno_dir',
                        help='ground truth anno storing dir',
                        default='./anno',
                        type=str)
    parser.add_argument('--xlsx_save_dir',\
                        help='xlsx storing dir',
                        default='./excel_result',
                        type=str)
    parser.add_argument('--xlsx_name',
                        help='xlsx name',
                        default='nodule_info.xlsx',
                        type=str)
    parser.add_argument('--csv_save_dir',
                        help='csv storing dir',
                        default='./csv_result',
                        type=str)
    parser.add_argument('--csv_name',
                        help='csv name',
                        default='nodule_info.csv',
                        type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    nodule_list_generator = NoduleListGenerator(gt_anno_dir=args.gt_anno_dir,
                                                xlsx_save_dir=args.xlsx_save_dir,
                                                xlsx_name=args.xlsx_name,
                                                csv_save_dir=args.csv_save_dir,
                                                csv_name=args.csv_name)

    nodule_list_generator.generate_interpolated_nodule_list()
