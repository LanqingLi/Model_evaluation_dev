# -- coding: utf-8 --
import json
import pandas as pd
import numpy as np
import os
import argparse
import shutil
from collections import OrderedDict
from common.custom_metric import ClassificationMetric, ClusteringMetric, cls_avg

from heart_dev.xml_tools import xml_to_boxeslist, xml_to_boxeslist_with_nodule_num, xml_to_boxeslist_without_nodule_cls, \
    xml_to_boxeslist_with_nodule_num_without_nodule_cls, generate_xml
from heart_dev.config import config
from heart_dev.post_process import df_to_cls_label
from heart_dev.get_df_nodules import get_nodule_stat, init_df_boxes
from tools.generate_interpolated_nodule_list.generate_interpolated_nodule_list import slice_num_to_three_digit_str

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

    def __init__(self, data_dir, data_type, gt_dir, img_dir,
                 img_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_contour'),
                 score_type='fscore', result_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_result'),
                 xlsx_name='BrainSemanticSegEvaluation.xlsx', json_name='BrainSemanticSegEvaluation',
                 conf_thresh=config.TEST.CONF_THRESHOLD, cls_weights=config.CLASS_WEIGHTS,
                 fscore_beta=config.FSCORE_BETA
                 ):
        assert os.path.isdir(data_dir), 'must initialize it with a valid directory of segmentation data'
        self.data_dir = data_dir
        self.data_type = data_type
        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.img_save_dir = img_save_dir
        self.cls_name = config.CLASSES
        self.cls_dict = config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.result_df = pd.DataFrame(
            columns=['PatientID', 'class', 'threshold', 'tp_count', 'tn_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', 'gt_vol', 'pred_vol', 'gt_phys_vol/mm^3', 'pred_phys_vol/mm^3', self.score_type])
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.conf_thresh = conf_thresh
        self.cls_weights = cls_weights
        self.fscore_beta = fscore_beta
        self.patient_list = []

