# coding:utf-8
import xlrd
import pydicom as dicom
import numpy as np
import os


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
    gt_cls_weight_dict = dict()
    cls_z_threshold_pred_dict = dict()
    cls_z_threshold_gt_dict = dict()
    gt_cls_z_threshold_gt_dict = dict()
    conf_thresh['__background__'] = 1.0
    class_list = []
    class_list.append('__background__')
    label_classes.append('__background__')
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
        gt_cls_weight_dict[label_name] = weight
        cls_z_threshold_pred_dict[class_name] = z_threshold_pred
        cls_z_threshold_gt_dict[class_name] = z_threshold_gt
        gt_cls_z_threshold_gt_dict[label_name] = z_threshold_gt
    # remove repeat element in class
    class_list1 = sorted(set(class_list), key=class_list.index)

    return class_list1, label_classes, class_dict, conf_thresh, cls_weight_dict, gt_cls_weight_dict, cls_z_threshold_pred_dict, \
           cls_z_threshold_gt_dict, gt_cls_z_threshold_gt_dict

def get_label_classes_from_xls_seg(filename):
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

    # remove repeat element in class
    class_list1 = sorted(set(class_list), key=class_list.index)

    return class_list1, label_classes, class_dict, conf_thresh, cls_weight_dict


def get_instance_number(dcm_path):
    '''
    用pydicom读取instanceNumber
    :param dcm_path: dcm文件路径
    :return: instanceNumber
    '''
    df = dicom.read_file(dcm_path, stop_before_pixels=True)
    return df.InstanceNumber

def rename_data(src_dir, src_name, tar_name):
    print src_dir
    for id in os.listdir(src_dir):
        print id
        os.rename(os.path.join(src_dir, id), os.path.join(src_dir, id.replace(src_name, tar_name)))

def intensity_window(array, lower_bnd, upper_bnd):
    array[array < lower_bnd] = lower_bnd
    array[array > upper_bnd] = upper_bnd

def window_convert( pix, center, width):
    pix_out = np.zeros(shape=pix.shape, dtype=np.uint8)
    low = center - width / 2 # 0
    hig = center + width / 2 # 60
    w1 = np.where(pix > low) and np.where(pix < hig)
    pix_out[w1] = ((pix[w1] - center + 0.5) / (width - 1) + 0.5) * 255
    pix_out[np.where(pix <= low)] = 0
    pix_out[np.where(pix >= hig)] = 255
    return pix_out

if __name__ == '__main__':
    src_dir = '/mnt/data2/model_evaluation_test/test_data/find_nodules/0801_lulin_review/anno/1000nodules'
    src_name = '52186031'
    tar_name = '1.2.840.113619.2.334.3.2831181473.233.1525908990.436.4'
    print src_dir
    rename_data(src_dir, src_name, tar_name)
