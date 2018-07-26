# coding:utf-8
import xlrd
import pydicom as dicom

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

        if len(classDictSheet.row(i)) == 3:
            thresh = classDictSheet.row(i)[2].value
            assert isinstance(thresh, float), 'thresh must be float type, check xls'
            conf_thresh[class_name] = thresh
    # remove repeat element in class
    class_list1 = sorted(set(class_list), key=class_list.index)

    return class_list1, label_classes, class_dict, conf_thresh

def get_instance_number(dcm_path):
    '''
    用pydicom读取instanceNumber
    :param dcm_path: dcm文件路径
    :return: instanceNumber
    '''
    df = dicom.read_file(dcm_path, stop_before_pixels=True)
    return df.InstanceNumber
