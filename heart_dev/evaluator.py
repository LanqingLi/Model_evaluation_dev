# -- coding: utf-8 --
import json
import pandas as pd
import numpy as np
import os
import argparse
import shutil
from collections import OrderedDict
from common.custom_metric import ClassificationMetric, ClusteringMetric, cls_avg
from tools.data_postprocess import save_xlsx_json
from heart_dev.xml_tools import xml_to_boxeslist, xml_to_boxeslist_with_nodule_num, xml_to_boxeslist_without_nodule_cls, \
    xml_to_boxeslist_with_nodule_num_without_nodule_cls, generate_xml
from heart_dev.config import config
from tools.xml_tools import xml_to_boxeslist_2d
from heart_dev.post_process import df_to_cls_label
from heart_dev.get_object_df import get_object_stat, init_df_boxes
from tools.generate_interpolated_nodule_list.generate_interpolated_nodule_list import slice_num_to_three_digit_str

class HeartPlaqueEvaluatorOnline(object):
    '''
    this class is designed for online-evaluation of our CT heart plaque. It can read anchor boxes from a selection of format (.json/.xml)
    and generate spreadsheets of statistics (tp, fp, etc. see common/custom_metric) for each plaque class under customized
    range of classification (softmax) probability threshold, which can be used for plotting ROC curve and calculating AUC.

    :param data_dir: 存储模型预测出的框的信息的数据路径，我们读入数据的路径
    :param data_type: 存储预测出的框的信息的数据格式，默认为.json，我们读入数据的格式。对于FRCNN,我们将clean_box_new输出的框存成.npy/.json供读取
    :param anno_dir: 存储对应CT ground truth数据标记(annotation)的路径
    :param score_type: 模型得分的函数，对于结节的检出、分类问题，我们默认用'fscore'
    :param result_save_dir: 存放评估结果的路径
    :param cls_name: 包含预测所有类别的列表，默认为config.CLASSES, 包含'__background__'类
    :param cls_dict: 包含'rcnn/classname_labelname_mapping.xls'中label_name到class_name映射的字典，不包含'__background__'类
    :param opt_thresh: 存储最终使得各类别模型预测结果最优的概率阈值及其对应tp,fp,score等信息的字典，index为预测的类别。每个index对应一个类似于
    self.count_df的字典，最终存储在self.xlsx_name的'optimal threshold' sheet中
    :param count_df: 初始化的pandas.DataFrame,用于存储最终输出的evaluation结果
    :param result_save_dir:　存储输出.xlsx结果的路径
    :param xlsx_name: 存储输出.xlsx文件的名字
    :param json_name: 存储输出.json文件的名字，不带后缀
    :param if_nodule_json:　是否根据ground truth annotation生成匹配后结节信息的.json文件
    :param conf_thresh:　自定义的置信度概率阈值采样点，存在列表中，用于求最优阈值及画ROC曲线
    :param nodule_cls_weights:　不同结节种类对于模型综合评分以及ObjMatch.ObjMatch.find_nodules算法中的权重，默认与结节分类信息一起从classname_labelname_mapping.xls中读取,类型为dict
    :param cls_weight: 在求加权平均结果时，每个类别的权重，类型为list
    :param cls_value: 在求加权平均结果时，每个类别的得分，类型为list
    :param thickness_thresh: nodule_thickness_filter根据此阈值对结节的层厚进行筛选
    :param nodule_compare_thresh: 比较两个结节是否算一个的IOU阈值
    '''

    def __init__(self, input_data, input_gt,
                 score_type='fscore', result_save_dir=os.path.join(os.getcwd(), 'HeartPlaqueEvaluation_result'),
                 xlsx_name='HeartPlaqueEvaluation.xlsx', json_name='HeartPlaqueEvaluation.json',
                 conf_thresh=config.TEST.CONF_THRESHOLD, cls_weights=config.CLASS_WEIGHTS,
                 fscore_beta=config.FSCORE_BETA
                 ):
        assert os.path.isdir(data_dir), 'must initialize it with a valid directory of predict data'
        self.data_dir = data_dir
        self.data_type = data_type
        self.gt_dir = gt_dir
        self.cls_name = config.CLASSES
        self.cls_dict = config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.result_df = pd.DataFrame(
            columns=['PatientID', 'class', 'threshold', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision', 'fp/tp', self.score_type])
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.conf_thresh = conf_thresh
        self.cls_weights = cls_weights
        self.fscore_beta = fscore_beta
        self.patient_list = []



class HeartPlaqueEvaluatorOffline(object):
    '''
    this class is designed for offline-evaluation of our CT heart plaque. It can read anchor boxes from a selection of format (.json/.xml)
    and generate spreadsheets of statistics (tp, fp, etc. see common/custom_metric) for each plaque class under customized
    range of classification (softmax) probability threshold, which can be used for plotting ROC curve and calculating AUC.

    :param data_dir: 存储模型预测出的框的信息的数据路径，我们读入数据的路径
    :param data_type: 存储预测出的框的信息的数据格式，默认为.json，我们读入数据的格式。对于FRCNN,我们将clean_box_new输出的框存成.npy/.json供读取
    :param anno_dir: 存储对应CT ground truth数据标记(annotation)的路径
    :param score_type: 模型得分的函数，对于结节的检出、分类问题，我们默认用'fscore'
    :param result_save_dir: 存放评估结果的路径
    :param cls_name: 包含预测所有类别的列表，默认为config.CLASSES, 包含'__background__'类
    :param cls_dict: 包含'rcnn/classname_labelname_mapping.xls'中label_name到class_name映射的字典，不包含'__background__'类
    :param opt_thresh: 存储最终使得各类别模型预测结果最优的概率阈值及其对应tp,fp,score等信息的字典，index为预测的类别。每个index对应一个类似于
    self.count_df的字典，最终存储在self.xlsx_name的'optimal threshold' sheet中
    :param count_df: 初始化的pandas.DataFrame,用于存储最终输出的evaluation结果
    :param result_save_dir:　存储输出.xlsx结果的路径
    :param xlsx_name: 存储输出.xlsx文件的名字
    :param json_name: 存储输出.json文件的名字，不带后缀
    :param if_nodule_json:　是否根据ground truth annotation生成匹配后结节信息的.json文件
    :param conf_thresh:　自定义的置信度概率阈值采样点，存在列表中，用于求最优阈值及画ROC曲线
    :param nodule_cls_weights:　不同结节种类对于模型综合评分以及ObjMatch.ObjMatch.find_nodules算法中的权重，默认与结节分类信息一起从classname_labelname_mapping.xls中读取,类型为dict
    :param cls_weight: 在求加权平均结果时，每个类别的权重，类型为list
    :param cls_value: 在求加权平均结果时，每个类别的得分，类型为list
    :param thickness_thresh: nodule_thickness_filter根据此阈值对结节的层厚进行筛选
    :param nodule_compare_thresh: 比较两个结节是否算一个的IOU阈值
    '''

    def __init__(self, data_dir, data_type, gt_dir,
                 img_save_dir=os.path.join(os.getcwd(), 'HeartPlaqueEvaluation'),
                 score_type='fscore', result_save_dir=os.path.join(os.getcwd(), 'HeartPlaqueEvaluation_result'),
                 xlsx_name='HeartPlaqueEvaluation.xlsx', json_name='HeartPlaqueEvaluation',
                 conf_thresh=config.TEST.CONF_THRESHOLD, cls_weights=config.CLASS_WEIGHTS,
                 fscore_beta=config.FSCORE_BETA
                 ):
        assert os.path.isdir(data_dir), 'must initialize it with a valid directory of predict data'
        self.data_dir = data_dir
        self.data_type = data_type
        self.gt_dir = gt_dir
        self.cls_name = config.CLASSES
        self.cls_dict = config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.result_df = pd.DataFrame(
            columns=['PatientID', 'class', 'threshold', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision', 'fp/tp', self.score_type])
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.conf_thresh = conf_thresh
        self.cls_weights = cls_weights
        self.fscore_beta = fscore_beta
        self.patient_list = []

    def load_data(self):
        predict_df_boxes_dict = {}
        ground_truth_boxes_dict = {}
        for PatientID in os.listdir(self.data_dir):
            if self.data_type == 'xml':
                predict_xml_dir = os.path.join(self.data_dir, PatientID)
                try:
                    predict_boxes = xml_to_boxeslist_2d(predict_xml_dir, self.cls_name, self.cls_dict, 2000, if_gt=False)
                    predict_boxes = init_df_boxes(return_boxes=predict_boxes, classes=self.cls_name)
                except:
                    print ("broken directory structure, maybe no predict xml file found: %s" % predict_xml_dir)
                    raise NameError
            else:
                # to be implemented: HDF5/json format data
                raise NotImplementedError

            predict_boxes = predict_boxes.sort_values(by=['prob'])
            predict_boxes = predict_boxes.reset_index(drop=True)
            predict_df_boxes_dict[PatientID] = predict_boxes

            ground_truth_path = os.path.join(self.anno_dir, PatientID)
            try:
                # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                ground_truth_boxes = xml_to_boxeslist_2d(ground_truth_path, self.cls_name, self.cls_dict, 2000, if_gt=True)
            except:
                print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                raise NameError

            ground_truth_boxes = init_df_boxes(return_boxes=ground_truth_boxes, classes=self.cls_name)
            ground_truth_boxes = ground_truth_boxes.sort_values(by=['prob'])
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            ground_truth_boxes_dict[PatientID] = ground_truth_boxes
            self.patient_list.append(PatientID)
        return predict_df_boxes_dict, ground_truth_boxes_dict


    # 多分类模型评分,每次只选取单类别的检出框，把其余所有类别作为负样本。
    def multi_class_evaluation(self):

        predict_df_boxes_dict, ground_truth_boxes_dict = self.load_data()

        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            self.cls_weight = []
            self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
            for i_cls, cls in enumerate(self.cls_name):
                if cls == "__background__":
                    continue

                cls_predict_df_list = []
                cls_gt_df_list = []
                for index, key in enumerate(predict_df_boxes_dict):
                    predict_df_boxes = predict_df_boxes_dict[key]
                    ground_truth_boxes = ground_truth_boxes_dict[key]

                    print ('plaque class: %s' % cls)
                    print ('processing %s' % key)

                    #筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                    if not predict_df_boxes_dict[key].empty:
                        filtered_predict_boxes = predict_df_boxes[predict_df_boxes["class"] == cls]
                        print filtered_predict_boxes
                        filtered_predict_boxes = filtered_predict_boxes[filtered_predict_boxes["prob"] >= thresh]
                        filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                    else:
                        filtered_predict_boxes = pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'class': [], 'prob': [], 'mask': []})

                    if not ground_truth_boxes_dict[key].empty:
                        filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["class"] == cls]
                        print filtered_gt_boxes
                        filtered_gt_boxes = filtered_gt_boxes[filtered_gt_boxes["prob"] >= thresh]
                        filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                    else:
                        filtered_gt_boxes = pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'class': [], 'prob': [], 'mask': []})

                    # 将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                    HeartPlaqueEvaluatorOnline

                    print "predict_boxes:"
                    print filtered_predict_boxes
                    _, cls_predict_df = get_object_stat(dicom_names=None,
                                                        hu_img_array=None,
                                                        return_boxes=filtered_predict_boxes,
                                                        img_spacing=None,
                                                        prefix=key,
                                                        classes=self.cls_name,
                                                        same_box_threshold=self.same_box_threshold_pred,
                                                        score_threshold=self.score_threshold_pred,
                                                        z_threshold=self.z_threshold_pred,
                                                        nodule_cls_weights=self.nodule_cls_weights,
                                                        if_dicom=False,
                                                        focus_priority_array=None,
                                                        skip_init=True)
                    print "predict_nodules:"
                    print cls_predict_df

                    print "gt_boxes:"
                    print filtered_gt_boxes
                    _, cls_gt_df = get_object_stat(dicom_names=None,
                                                   hu_img_array=None,
                                                   return_boxes=filtered_gt_boxes,
                                                   img_spacing=None,
                                                   prefix=key,
                                                   classes=self.cls_name,
                                                   same_box_threshold=self.same_box_threshold_gt,
                                                   score_threshold=self.score_threshold_gt,
                                                   z_threshold=self.z_threshold_gt,
                                                   nodule_cls_weights=self.nodule_cls_weights,
                                                   if_dicom=False,
                                                   focus_priority_array=None,
                                                   skip_init=True)
                    print "gt_nodules:"
                    print cls_gt_df

                    cls_predict_df = cls_predict_df.reset_index(drop=True)
                    cls_predict_df_list.append(json_df_2_df(cls_predict_df))

                    cls_gt_df = cls_gt_df.reset_index(drop=True)
                    cls_gt_df_list.append(json_df_2_df(cls_gt_df))

                # convert pandas dataframe to list of class labels
                cls_pred_labels, cls_gt_labels = df_to_cls_label(cls_predict_df_list, cls_gt_df_list, self.cls_name,
                                                                 thresh=self.nodule_compare_thresh)

                # initialize ClassificationMetric class and update with ground truth/predict labels
                cls_metric = ClassificationMetric(cls_num=i_cls)

                cls_metric.update(cls_gt_labels, cls_pred_labels)

                if cls_metric.tp == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric.fp / cls_metric.tp

                self.count_df = self.count_df.append({'nodule_class': cls,
                                                      'threshold': thresh,
                                                      'tp_count': cls_metric.tp,
                                                      'fp_count': cls_metric.fp,
                                                      'fn_count': cls_metric.fn,
                                                      'accuracy': cls_metric.get_acc(),
                                                      'recall': cls_metric.get_rec(),
                                                      'precision': cls_metric.get_prec(),
                                                      'fp/tp': fp_tp,
                                                      self.score_type: cls_metric.get_fscore(beta=1.)},
                                                     ignore_index=True)

                # find the optimal threshold
                if cls not in self.opt_thresh:

                    self.opt_thresh[cls] = self.count_df.iloc[-1]

                    self.opt_thresh[cls]["threshold"] = thresh

                else:
                    # we choose the optimal threshold corresponding to the one that gives the highest model score
                    if self.count_df.iloc[-1][self.score_type] > self.opt_thresh[cls][self.score_type]:
                        self.opt_thresh[cls] = self.count_df.iloc[-1]
                        self.opt_thresh[cls]["threshold"] = thresh

                self.cls_value['accuracy'].append(cls_metric.get_acc())
                self.cls_value['recall'].append(cls_metric.get_rec())
                self.cls_value['precision'].append(cls_metric.get_prec())
                self.cls_value[self.score_type].append(cls_metric.get_fscore(beta=1.))

            # 增加多类别加权平均的结果
            self.count_df = self.count_df.append({'nodule_class': 'average',
                                                  'threshold': thresh,
                                                  'tp_count': np.nan,
                                                  'fp_count': np.nan,
                                                  'fn_count': np.nan,
                                                  'accuracy': cls_avg(self.cls_weight, self.cls_value['accuracy']),
                                                  'recall': cls_avg(self.cls_weight, self.cls_value['recall']),
                                                  'precision': cls_avg(self.cls_weight, self.cls_value['precision']),
                                                  'fp/tp': np.nan,
                                                  self.score_type: cls_avg(self.cls_weight,
                                                                           self.cls_value[self.score_type])},
                                                 ignore_index=True)
        self.count_df = self.count_df.sort_values(['nodule_class', 'threshold'])

        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
        for key in self.opt_thresh:
            self.cls_value['accuracy'].append(self.opt_thresh[key]['accuracy'])
            self.cls_value['recall'].append(self.opt_thresh[key]['recall'])
            self.cls_value['precision'].append(self.opt_thresh[key]['precision'])
            self.cls_value[self.score_type].append(self.opt_thresh[key][self.score_type])
            self.cls_weight.append(self.nodule_cls_weights[key])

        opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        opt_thresh = opt_thresh.append({'nodule_class': 'average',
                                        'threshold': np.nan,
                                        'tp_count': np.nan,
                                        'fp_count': np.nan,
                                        'fn_count': np.nan,
                                        'accuracy': cls_avg(self.cls_weight, self.cls_value['accuracy']),
                                        'recall': cls_avg(self.cls_weight, self.cls_value['recall']),
                                        'precision': cls_avg(self.cls_weight, self.cls_value['precision']),
                                        'fp/tp': np.nan,
                                        self.score_type: cls_avg(self.cls_weight,
                                                                 self.cls_value[self.score_type])},
                                       ignore_index=True)

        save_xlsx_json(self.result_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'multi-class_evaluation', 'optimal_threshold')
