# -- coding: utf-8 --
import sys
import json
import pandas as pd
import numpy as np
import os
import argparse
import shutil
import logging
from collections import OrderedDict
from model_eval.common.custom_metric import ClassificationMetric, ClusteringMetric, cls_avg

from model_eval.lung.xml_tools import xml_to_boxeslist, xml_to_boxeslist_with_nodule_num, xml_to_boxeslist_without_nodule_cls, \
    xml_to_boxeslist_with_nodule_num_without_nodule_cls, xml_to_boxeslist_multi_classes, generate_xml, xml_to_anchorlist, xml_to_anchorlist_multi_classes
from model_eval.lung.config import LungConfig
from objmatch.post_process import df_to_cls_label,df_to_xlsx_file
from model_eval.lung.get_df_objects import get_object_stat, init_df_objects
from model_eval.lung.get_df_nodules import get_nodule_stat, init_df_boxes
from model_eval.tools.data_postprocess import save_xlsx_json, save_xlsx_json_three_sheets,save_xlsx_sheets

class LungNoduleAnchorEvaluatorOffline(object):
    '''
    this class is designed for evaluation of our CT lung model offline. It can read generalized objects (e.g. anchor boxes)
    from a selection of format (.json/.npy) and generate spreadsheets of statistics (tp, fp, etc. see common/custom_metric)
    for each nodule class under customized range of classification (softmax) probability threshold, which can be used for
    plotting ROC curve and calculating AUC.

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
    :param nodule_cls_weights:　不同结节种类对于模型综合评分以及objmatch.find_nodules算法中的权重，默认与结节分类信息一起从classname_labelname_mapping.xls中读取,类型为dict
    :param cls_weight: 在求加权平均结果时，每个类别的权重，类型为list
    :param cls_value: 在求加权平均结果时，每个类别的得分，类型为list
    :param thickness_thresh: nodule_thickness_filter根据此阈值对结节的层厚进行筛选
    :param nodule_compare_thresh: 比较两个结节是否算一个的IOU阈值
    '''
    def __init__(self, cls_label_xls_path, data_dir, data_type, anno_dir, score_type = 'fscore',  result_save_dir = os.path.join(os.getcwd(), 'LungNoduleEvaluation_result'),
                 xlsx_name = 'LungNoduleEvaluation.xlsx', json_name = 'LungNoduleEvaluation', if_nodule_json = False,
                 conf_thresh = np.linspace(0.1, 0.9, num=9).tolist(), fscore_beta = 1.,
                 same_box_threshold_pred = np.array([1.6, 1.6]), same_box_threshold_gt = np.array([0., 0.]),
                 score_threshold_pred = 0.6, score_threshold_gt = 0.4, if_nodule_threshold = False, thickness_thresh = 0.,
                 key_list = [], class_key = '', matched_key_list = [],
                 cls_focus_priority_array = {"mass": 6,
                                            "calcific nodule": 5,
                                            "solid nodule": 4,
                                            "GGN": 3,
                                            "0-3nodule": 2,
                                            "nodule": 1},
                 gt_cls_focus_priority_array = {"mass": 9,
                                                "10-30nodule": 8,
                                                "6-10nodule": 7,
                                                "calcific nodule": 6,
                                                "pleural nodule": 5,
                                                "3-6nodule": 4,
                                                "5GGN": 3,
                                                "GGN": 3,
                                                "0-5GGN": 2,
                                                "0-3nodule": 1},
                 if_ensemble=False, model_weight_list=[1], model_conf_list=[1], obj_freq_thresh=None, model_list=[]):
        self.config = LungConfig(cls_label_xls_path=cls_label_xls_path)
        assert os.path.isdir(data_dir), 'must initialize it with a valid directory of bbox data'
        self.data_dir = data_dir
        self.data_type = data_type
        self.anno_dir = anno_dir
        # config.CLASSES 包含background class,是结节的粗分类(RCNN分类)
        self.cls_name = self.config.CLASSES
        # config.NODULE_CLASSES 不包含background class,是结节的细分类(ground truth label分类)
        self.gt_cls_name = self.config.NODULE_CLASSES
        self.cls_dict = self.config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.count_df = pd.DataFrame(
                     columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                              'accuracy', 'recall', 'precision',
                              'fp/tp', self.score_type])
        self.gt_cls_count_df = pd.DataFrame(
                     columns=['class', 'threshold', 'tp_count', 'fn_count', 'recall'])
        self.summary_count_df={}
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.if_nodule_json = if_nodule_json
        # customized confidence threshold for plotting ROC curve
        self.conf_thresh = conf_thresh
        self.nodule_cls_weights = self.config.CLASS_WEIGHTS
        self.gt_cls_weights = self.config.GT_CLASSES_WEIGHTS
        self.fscore_beta = fscore_beta
        self.patient_list = []
        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
        # objmatch.find_nodules/find_objects算法的相关超参数，详见config文件
        self.same_box_threshold_pred = same_box_threshold_pred
        self.same_box_threshold_gt = same_box_threshold_gt
        self.score_threshold_pred = score_threshold_pred
        self.score_threshold_gt = score_threshold_gt
        self.z_threshold_pred = self.config.CLASS_Z_THRESHOLD_PRED
        self.z_threshold_gt = self.config.CLASS_Z_THRESHOLD_GT
        self.gt_cls_z_threshold_gt = self.config.GT_CLASS_Z_THRESHOLD_GT
        self.if_nodule_threshold = if_nodule_threshold
        self.model_weight_list = model_weight_list
        self.model_cof_list = model_conf_list
        self.obj_freq_thresh = obj_freq_thresh

        self.thickness_thresh = thickness_thresh
        self.nodule_compare_thresh = self.config.TEST.OBJECT_COMPARE_THRESHOLD

        # keep track of the nodule count in the output of get_df_nodules, including false positives, initialized to be 0
        self.nodule_count = 0.
        self.cls_focus_priority_array = cls_focus_priority_array
        self.gt_cls_focus_priority_array = gt_cls_focus_priority_array
        self.key_list = key_list
        self.class_key = class_key
        self.matched_key_list = matched_key_list

        # whether make prediction based on output of a model ensemble or a single model
        self.if_ensemble = if_ensemble

        if len(model_list) == 0:
            self.model_list = os.listdir(self.data_dir)
        else:
            self.model_list = model_list

    # 多分类模型评分,每次只选取单类别的检出框，把其余所有类别作为负样本。
    def multi_class_evaluation(self):

        self.count_df = pd.DataFrame(
            columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', self.score_type])
        self.opt_thresh = {}

        predict_df_boxes_dict, ground_truth_boxes_dict, _ = self.load_data()

        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            self.cls_weight = []
            self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
            for i_cls, cls in enumerate(self.cls_name):
                if cls == "__background__":
                    continue
                # construct class weight list, for computing class-average result
                self.cls_weight.append(self.nodule_cls_weights[cls])

                cls_predict_df_list = []
                cls_gt_df_list = []
                self.nodule_count = 0.
                for index, key in enumerate(predict_df_boxes_dict):
                    self.patient_list.append(key)
                    print ('nodule class: %s' % cls)
                    print ('processing %s' % key)

                    if self.if_ensemble:
                        predict_df_boxes_list = predict_df_boxes_dict[key]
                        ground_truth_boxes = ground_truth_boxes_dict[key]
                        filtered_predict_boxes_list = []

                        for model_idx, _ in enumerate(predict_df_boxes_list):
                            # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                            if not predict_df_boxes_list[model_idx].empty:
                                filtered_predict_boxes_list.append(predict_df_boxes_list[model_idx][predict_df_boxes_list[model_idx]["class"] == cls])
                                print filtered_predict_boxes_list[model_idx]
                                filtered_predict_boxes_list[model_idx] = filtered_predict_boxes_list[model_idx][filtered_predict_boxes_list[model_idx]["prob"] >= thresh]
                                filtered_predict_boxes_list[model_idx] = filtered_predict_boxes_list[model_idx].reset_index(drop=True)
                            else:
                                filtered_predict_boxes_list.append(pd.DataFrame(
                                    {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                     'class': [], 'prob': [], 'mask': []}))

                        if not ground_truth_boxes_dict[key].empty:
                            filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["class"] == cls]
                            print filtered_gt_boxes
                            filtered_gt_boxes = filtered_gt_boxes[filtered_gt_boxes["prob"] >= thresh]
                            filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                        else:
                            filtered_gt_boxes = filtered_gt_boxes.append(pd.DataFrame(
                                {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                 'class': [], 'prob': [], 'mask': []}))

                        # 将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                        print "predict_boxes:"
                        print filtered_predict_boxes_list
                        _, cls_predict_df = get_object_stat(
                            hu_img_array=None,
                            slice_object_list=filtered_predict_boxes_list,
                            img_spacing=None,
                            prefix=key,
                            classes=self.cls_name,
                            same_box_threshold=self.same_box_threshold_pred,
                            score_threshold=self.score_threshold_pred,
                            z_threshold=self.z_threshold_pred,
                            nodule_cls_weights=self.nodule_cls_weights,
                            if_dicom=False,
                            focus_priority_array=self.cls_focus_priority_array,
                            skip_init=True,
                            key_list=self.key_list,
                            class_key=self.class_key,
                            matched_key_list=self.matched_key_list,
                            if_ensemble=self.if_ensemble,
                            model_weight_list=self.model_weight_list,
                            model_conf_list=self.model_cof_list,
                            obj_freq_thresh=self.obj_freq_thresh)
                        print "predict_nodules:"
                        print cls_predict_df

                        print "gt_boxes:"
                        print filtered_gt_boxes
                        _, cls_gt_df = get_object_stat(
                            hu_img_array=None,
                            slice_object_list=filtered_gt_boxes,
                            img_spacing=None,
                            prefix=key,
                            classes=self.cls_name,
                            same_box_threshold=self.same_box_threshold_gt,
                            score_threshold=self.score_threshold_gt,
                            z_threshold=self.z_threshold_gt,
                            nodule_cls_weights=self.nodule_cls_weights,
                            if_dicom=False,
                            focus_priority_array=self.cls_focus_priority_array,
                            skip_init=True,
                            key_list=self.key_list,
                            class_key=self.class_key,
                            matched_key_list=self.matched_key_list)
                        print "gt_nodules:"
                        print cls_gt_df

                    else:
                        predict_df_boxes = predict_df_boxes_dict[key]
                        ground_truth_boxes = ground_truth_boxes_dict[key]



                        # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
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
                        print "predict_boxes:"
                        print filtered_predict_boxes
                        _, cls_predict_df = get_object_stat(
                            hu_img_array=None,
                            slice_object_list=filtered_predict_boxes,
                            img_spacing=None,
                            prefix=key,
                            classes=self.cls_name,
                            same_box_threshold=self.same_box_threshold_pred,
                            score_threshold=self.score_threshold_pred,
                            z_threshold=self.z_threshold_pred,
                            nodule_cls_weights=self.nodule_cls_weights,
                            if_dicom=False,
                            focus_priority_array=self.cls_focus_priority_array,
                            skip_init=True,
                            key_list=self.key_list,
                            class_key=self.class_key,
                            matched_key_list=self.matched_key_list)
                        print "predict_nodules:"
                        print cls_predict_df

                        print "gt_boxes:"
                        print filtered_gt_boxes
                        _, cls_gt_df = get_object_stat(
                            hu_img_array=None,
                            slice_object_list=filtered_gt_boxes,
                            img_spacing=None,
                            prefix=key,
                            classes=self.cls_name,
                            same_box_threshold=self.same_box_threshold_gt,
                            score_threshold=self.score_threshold_gt,
                            z_threshold=self.z_threshold_gt,
                            nodule_cls_weights=self.nodule_cls_weights,
                            if_dicom=False,
                            focus_priority_array=self.cls_focus_priority_array,
                            skip_init=True,
                            key_list=self.key_list,
                            class_key=self.class_key,
                            matched_key_list=self.matched_key_list)
                        print "gt_nodules:"
                        print cls_gt_df

                    cls_predict_df = cls_predict_df.reset_index(drop=True)
                    cls_predict_df_list.append(json_df_2_df(cls_predict_df))

                    cls_gt_df = cls_gt_df.reset_index(drop=True)
                    cls_gt_df_list.append(json_df_2_df(cls_gt_df))

                    self.nodule_count += len(cls_predict_df.index)

                # convert pandas dataframe to list of class labels
                cls_pred_labels, cls_gt_labels = df_to_cls_label(cls_predict_df_list, cls_gt_df_list, self.cls_name,
                                                                 thresh=self.nodule_compare_thresh)

                # initialize ClassificationMetric class and update with ground truth/predict labels
                cls_metric = ClassificationMetric(cls_num=len(self.cls_name) - 1, if_binary=True,
                                                  pos_cls_fusion=False)

                cls_metric.update(cls_gt_labels, cls_pred_labels, i_cls)

                if cls_metric.tp[i_cls - 1] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric.fp[i_cls - 1] / cls_metric.tp[i_cls - 1]

                self.count_df = self.count_df.append({'class': cls,
                                                      'threshold': thresh,
                                                      'nodule_count': self.nodule_count,
                                                      'tp_count': cls_metric.tp[i_cls - 1],
                                                      'fp_count': cls_metric.fp[i_cls - 1],
                                                      'fn_count': cls_metric.fn[i_cls - 1],
                                                      'accuracy': cls_metric.get_acc(i_cls),
                                                      'recall': cls_metric.get_rec(i_cls),
                                                      'precision': cls_metric.get_prec(i_cls),
                                                      'fp/tp': fp_tp,
                                                      self.score_type: cls_metric.get_fscore(cls_label=i_cls,
                                                                                             beta=self.fscore_beta)},
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

                self.cls_value['accuracy'].append(cls_metric.get_acc(i_cls))
                self.cls_value['recall'].append(cls_metric.get_rec(i_cls))
                self.cls_value['precision'].append(cls_metric.get_prec(i_cls))
                self.cls_value[self.score_type].append(
                    cls_metric.get_fscore(cls_label=i_cls, beta=self.fscore_beta))

            # 增加多类别加权平均的结果
            self.count_df = self.count_df.append({'class': 'average',
                                                  'threshold': thresh,
                                                  'tp_count': np.nan,
                                                  'fp_count': np.nan,
                                                  'fn_count': np.nan,
                                                  'accuracy': cls_avg(self.cls_weight, self.cls_value['accuracy']),
                                                  'recall': cls_avg(self.cls_weight, self.cls_value['recall']),
                                                  'precision': cls_avg(self.cls_weight,
                                                                       self.cls_value['precision']),
                                                  'fp/tp': np.nan,
                                                  self.score_type: cls_avg(self.cls_weight,
                                                                           self.cls_value[self.score_type])},
                                                 ignore_index=True)

        self.count_df = self.count_df.sort_values(['class', 'threshold'])

        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
        for key in self.opt_thresh:
            self.cls_value['accuracy'].append(self.opt_thresh[key]['accuracy'])
            self.cls_value['recall'].append(self.opt_thresh[key]['recall'])
            self.cls_value['precision'].append(self.opt_thresh[key]['precision'])
            self.cls_value[self.score_type].append(self.opt_thresh[key][self.score_type])
            self.cls_weight.append(self.nodule_cls_weights[key])

        opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        opt_thresh = opt_thresh.append({'class': 'average',
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

        save_xlsx_json(self.count_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'multi-class_evaluation', 'optimal_threshold')


    # 多分类模型评分,每次只选取单类别的检出框，把其余所有类别作为负样本。先把框匹配成结节，再用阈值对结节的最高概率进行筛选
    def multi_class_evaluation_nodule_threshold(self):

        predict_df_boxes_dict, ground_truth_boxes_dict, _ = self.load_data()
        self.count_df = pd.DataFrame(
            columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', self.score_type])
        self.opt_thresh = {}

        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            self.cls_weight = []
            self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
            for i_cls, cls in enumerate(self.cls_name):
                if cls == "__background__":
                    continue
                # construct class weight list, for computing class-average result
                self.cls_weight.append(self.nodule_cls_weights[cls])

                cls_predict_df_list = []
                cls_gt_df_list = []
                self.nodule_count = 0.
                for index, key in enumerate(predict_df_boxes_dict):
                    self.patient_list.append(key)


                    print ('nodule class: %s' % cls)
                    print ('processing %s' % key)

                    if self.if_ensemble:
                        predict_df_boxes_list = predict_df_boxes_dict[key]
                        ground_truth_boxes = ground_truth_boxes_dict[key]
                        filtered_predict_boxes_list = []

                        for model_idx, _ in enumerate(predict_df_boxes_list):
                            # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                            if not predict_df_boxes_list[model_idx].empty:
                                filtered_predict_boxes_list.append(predict_df_boxes_list[model_idx][predict_df_boxes_list[model_idx]["class"] == cls])
                                print filtered_predict_boxes_list[model_idx]
                                #filtered_predict_boxes_list[model_idx] = filtered_predict_boxes_list[model_idx][filtered_predict_boxes_list[model_idx]["prob"] >= thresh]
                                filtered_predict_boxes_list[model_idx] = filtered_predict_boxes_list[model_idx].reset_index(drop=True)
                            else:
                                filtered_predict_boxes_list.append(pd.DataFrame(
                                    {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                     'class': [], 'prob': [], 'mask': []}))

                        if not ground_truth_boxes_dict[key].empty:
                            filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["class"] == cls]
                            print filtered_gt_boxes
                            #filtered_gt_boxes = filtered_gt_boxes[filtered_gt_boxes["prob"] >= thresh]
                            filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                        else:
                            filtered_gt_boxes = filtered_gt_boxes.append(pd.DataFrame(
                                {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                 'class': [], 'prob': [], 'mask': []}))

                        # 将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                        print "predict_boxes:"
                        print filtered_predict_boxes_list
                        _, cls_predict_df = get_object_stat(
                            hu_img_array=None,
                            slice_object_list=filtered_predict_boxes_list,
                            img_spacing=None,
                            prefix=key,
                            classes=self.cls_name,
                            same_box_threshold=self.same_box_threshold_pred,
                            score_threshold=self.score_threshold_pred,
                            z_threshold=self.z_threshold_pred,
                            nodule_cls_weights=self.nodule_cls_weights,
                            if_dicom=False,
                            focus_priority_array=self.cls_focus_priority_array,
                            skip_init=True,
                            key_list=self.key_list,
                            class_key=self.class_key,
                            matched_key_list=self.matched_key_list,
                            if_ensemble=self.if_ensemble,
                            model_weight_list=self.model_weight_list,
                            model_conf_list=self.model_cof_list,
                            obj_freq_thresh=self.obj_freq_thresh)
                        print "predict_nodules:"
                        print cls_predict_df

                        print "gt_boxes:"
                        print filtered_gt_boxes
                        _, cls_gt_df = get_object_stat(
                            hu_img_array=None,
                            slice_object_list=filtered_gt_boxes,
                            img_spacing=None,
                            prefix=key,
                            classes=self.cls_name,
                            same_box_threshold=self.same_box_threshold_gt,
                            score_threshold=self.score_threshold_gt,
                            z_threshold=self.z_threshold_gt,
                            nodule_cls_weights=self.nodule_cls_weights,
                            if_dicom=False,
                            focus_priority_array=self.cls_focus_priority_array,
                            skip_init=True,
                            key_list=self.key_list,
                            class_key=self.class_key,
                            matched_key_list=self.matched_key_list)
                        print "gt_nodules:"
                        print cls_gt_df

                    else:
                        predict_df_boxes = predict_df_boxes_dict[key]
                        ground_truth_boxes = ground_truth_boxes_dict[key]



                        # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                        if not predict_df_boxes_dict[key].empty:
                            filtered_predict_boxes = predict_df_boxes[predict_df_boxes["class"] == cls]
                            print filtered_predict_boxes
                            #filtered_predict_boxes = filtered_predict_boxes[filtered_predict_boxes["prob"] >= thresh]
                            filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                        else:
                            filtered_predict_boxes = pd.DataFrame(
                                {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                 'class': [], 'prob': [], 'mask': []})

                        if not ground_truth_boxes_dict[key].empty:
                            filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["class"] == cls]
                            print filtered_gt_boxes
                            #filtered_gt_boxes = filtered_gt_boxes[filtered_gt_boxes["prob"] >= thresh]
                            filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                        else:
                            filtered_gt_boxes = pd.DataFrame(
                                {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                 'class': [], 'prob': [], 'mask': []})

                        # 将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                        print "predict_boxes:"
                        print filtered_predict_boxes
                        _, cls_predict_df = get_object_stat(
                            hu_img_array=None,
                            slice_object_list=filtered_predict_boxes,
                            img_spacing=None,
                            prefix=key,
                            classes=self.cls_name,
                            same_box_threshold=self.same_box_threshold_pred,
                            score_threshold=self.score_threshold_pred,
                            z_threshold=self.z_threshold_pred,
                            nodule_cls_weights=self.nodule_cls_weights,
                            if_dicom=False,
                            focus_priority_array=self.cls_focus_priority_array,
                            skip_init=True,
                            key_list=self.key_list,
                            class_key=self.class_key,
                            matched_key_list=self.matched_key_list)
                        print "predict_nodules:"
                        print cls_predict_df

                        print "gt_boxes:"
                        print filtered_gt_boxes
                        _, cls_gt_df = get_object_stat(
                            hu_img_array=None,
                            slice_object_list=filtered_gt_boxes,
                            img_spacing=None,
                            prefix=key,
                            classes=self.cls_name,
                            same_box_threshold=self.same_box_threshold_gt,
                            score_threshold=self.score_threshold_gt,
                            z_threshold=self.z_threshold_gt,
                            nodule_cls_weights=self.nodule_cls_weights,
                            if_dicom=False,
                            focus_priority_array=self.cls_focus_priority_array,
                            skip_init=True,
                            key_list=self.key_list,
                            class_key=self.class_key,
                            matched_key_list=self.matched_key_list)
                        print "gt_nodules:"
                        print cls_gt_df

                    cls_predict_df = cls_predict_df[cls_predict_df['Prob'] >= thresh]
                    cls_predict_df = cls_predict_df.reset_index(drop=True)
                    cls_predict_df_list.append(json_df_2_df(cls_predict_df))

                    cls_gt_df = cls_gt_df[cls_gt_df['Prob'] >= thresh]
                    cls_gt_df = cls_gt_df.reset_index(drop=True)
                    cls_gt_df_list.append(json_df_2_df(cls_gt_df))

                    self.nodule_count += len(cls_predict_df.index)

                # convert pandas dataframe to list of class labels
                cls_pred_labels, cls_gt_labels = df_to_cls_label(cls_predict_df_list, cls_gt_df_list, self.cls_name,
                                                                 thresh=self.nodule_compare_thresh)

                # initialize ClassificationMetric class and update with ground truth/predict labels
                cls_metric = ClassificationMetric(cls_num=len(self.cls_name)-1, if_binary=True, pos_cls_fusion=False)

                cls_metric.update(cls_gt_labels, cls_pred_labels, i_cls)

                if cls_metric.tp[i_cls-1] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric.fp[i_cls-1] / cls_metric.tp[i_cls-1]

                self.count_df = self.count_df.append({'class': cls,
                                                      'threshold': thresh,
                                                      'nodule_count': self.nodule_count,
                                                      'tp_count': cls_metric.tp[i_cls-1],
                                                      'fp_count': cls_metric.fp[i_cls-1],
                                                      'fn_count': cls_metric.fn[i_cls-1],
                                                      'accuracy': cls_metric.get_acc(i_cls),
                                                      'recall': cls_metric.get_rec(i_cls),
                                                      'precision': cls_metric.get_prec(i_cls),
                                                      'fp/tp': fp_tp,
                                                      self.score_type: cls_metric.get_fscore(cls_label=i_cls, beta=self.fscore_beta)},
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

                self.cls_value['accuracy'].append(cls_metric.get_acc(i_cls))
                self.cls_value['recall'].append(cls_metric.get_rec(i_cls))
                self.cls_value['precision'].append(cls_metric.get_prec(i_cls))
                self.cls_value[self.score_type].append(cls_metric.get_fscore(cls_label=i_cls, beta=self.fscore_beta))

            # 增加多类别加权平均的结果
            self.count_df = self.count_df.append({'class': 'average',
                                                  'threshold': thresh,
                                                  'tp_count': np.nan,
                                                  'fp_count': np.nan,
                                                  'fn_count': np.nan,
                                                  'accuracy': cls_avg(self.cls_weight, self.cls_value['accuracy']),
                                                  'recall': cls_avg(self.cls_weight, self.cls_value['recall']),
                                                  'precision': cls_avg(self.cls_weight,
                                                                       self.cls_value['precision']),
                                                  'fp/tp': np.nan,
                                                  self.score_type: cls_avg(self.cls_weight,
                                                                           self.cls_value[self.score_type])},
                                                  ignore_index=True)
        self.count_df = self.count_df.sort_values(['class', 'threshold'])

        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
        for key in self.opt_thresh:
            self.cls_value['accuracy'].append(self.opt_thresh[key]['accuracy'])
            self.cls_value['recall'].append(self.opt_thresh[key]['recall'])
            self.cls_value['precision'].append(self.opt_thresh[key]['precision'])
            self.cls_value[self.score_type].append(self.opt_thresh[key][self.score_type])
            self.cls_weight.append(self.nodule_cls_weights[key])

        opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        opt_thresh = opt_thresh.append({'class': 'average',
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

        save_xlsx_json(self.count_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'multi-class_evaluation', 'optimal_threshold')

    # 二分类（检出）模型统计,将所有正样本类别统计在一起
    def binary_class_evaluation(self):

        predict_df_boxes_dict, gt_df_boxes_dict, gt_df_boxes_multi_classes_dict = self.load_data()
        self.count_df = pd.DataFrame(
            columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', self.score_type])
        self.gt_cls_count_df = pd.DataFrame(
                     columns=['class', 'threshold', 'tp_count', 'fn_count', 'recall'])
        self.opt_thresh = {}
        self.summary_count_df={}
        self.series_name  = []
        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            predict_df_list = []
            gt_df_multi_list=[]
            self.nodule_count = 0.

            for index, key in enumerate(predict_df_boxes_dict):

                self.patient_list.append(key)

                print ('processing %s' % key)

                if self.if_ensemble:
                    predict_df_boxes_list = predict_df_boxes_dict[key]
                    ground_truth_boxes = gt_df_boxes_dict[key]
                    filtered_predict_boxes_list = []

                    for model_idx, _ in enumerate(predict_df_boxes_list):
                        # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                        if not predict_df_boxes_list[model_idx].empty:
                            filtered_predict_boxes_list.append(
                                predict_df_boxes_list[model_idx])
                            #print filtered_predict_boxes_list[model_idx]
                            filtered_predict_boxes_list[model_idx] = filtered_predict_boxes_list[model_idx][
                                filtered_predict_boxes_list[model_idx]["prob"] >= thresh]
                            filtered_predict_boxes_list[model_idx] = filtered_predict_boxes_list[model_idx].reset_index(
                                drop=True)
                        else:
                            filtered_predict_boxes_list.append(pd.DataFrame(
                                {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                 'class': [], 'prob': [], 'mask': []}))

                    if not gt_df_boxes_dict[key].empty:
                        filtered_gt_boxes = ground_truth_boxes
                        #print filtered_gt_boxes
                        filtered_gt_boxes = filtered_gt_boxes[filtered_gt_boxes["prob"] >= thresh]
                        filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                    else:
                        filtered_gt_boxes = filtered_gt_boxes.append(pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'class': [], 'prob': [], 'mask': []}))

                    # 将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                    print "predict_boxes:"
                    print filtered_predict_boxes_list
                    _, predict_df = get_object_stat(
                        hu_img_array=None,
                        slice_object_list=filtered_predict_boxes_list,
                        img_spacing=None,
                        prefix=key,
                        classes=self.cls_name,
                        same_box_threshold=self.same_box_threshold_pred,
                        score_threshold=self.score_threshold_pred,
                        z_threshold=self.z_threshold_pred,
                        nodule_cls_weights=self.nodule_cls_weights,
                        if_dicom=False,
                        focus_priority_array=self.cls_focus_priority_array,
                        skip_init=True,
                        key_list=self.key_list,
                        class_key=self.class_key,
                        matched_key_list=self.matched_key_list,
                        if_ensemble=self.if_ensemble,
                        model_weight_list=self.model_weight_list,
                        model_conf_list=self.model_cof_list,
                        obj_freq_thresh=self.obj_freq_thresh)
                    print "predict_nodules:"
                    print predict_df

                    self.nodule_count += len(predict_df)
                    predict_df = predict_df.reset_index(drop=True)
                    predict_df_list.append(json_df_2_df(predict_df))

                    # 统计ground truth 结节信息
                    gt_df_boxes_multi_classes = gt_df_boxes_multi_classes_dict[key]

                    if not gt_df_boxes_dict[key].empty:
                        filtered_gt_boxes_multi_classes = gt_df_boxes_multi_classes[
                            gt_df_boxes_multi_classes["prob"] >= thresh]
                        filtered_gt_boxes_multi_classes = filtered_gt_boxes_multi_classes.reset_index(drop=True)
                        print "gt_boxes_multi_classes:"
                        print filtered_gt_boxes_multi_classes
                        _, gt_df_multi_classes = get_object_stat(hu_img_array=None,
                                                                 slice_object_list=filtered_gt_boxes_multi_classes,
                                                                 img_spacing=None,
                                                                 prefix=key,
                                                                 classes=self.gt_cls_name,
                                                                 same_box_threshold=self.same_box_threshold_gt,
                                                                 score_threshold=self.score_threshold_gt,
                                                                 z_threshold=self.gt_cls_z_threshold_gt,
                                                                 nodule_cls_weights=self.gt_cls_weights,
                                                                 if_dicom=False,
                                                                 focus_priority_array=self.gt_cls_focus_priority_array,
                                                                 skip_init=True,
                                                                 key_list=self.key_list,
                                                                 class_key=self.class_key,
                                                                 matched_key_list=self.matched_key_list)
                        print 'gt_nodules_multi_classes:'
                        print gt_df_multi_classes
                        if len(gt_df_multi_classes) != 8:
                            self.series_name.append(key)

                    else:
                        gt_df_multi_classes = pd.DataFrame({'Bndbox List': [], 'Object Id': [], 'Pid': key, 'Type': [],
                                                            'SliceRange': [], 'Prob': [], 'Diameter': [],
                                                            'CT_value': []})

                    gt_df_multi_classes = gt_df_multi_classes.reset_index(drop=True)
                    gt_df_multi_list.append(json_df_2_df(gt_df_multi_classes))

                else:
                    predict_df_boxes = predict_df_boxes_dict[key]

                    # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                    if not predict_df_boxes_dict[key].empty:
                        filtered_predict_boxes = predict_df_boxes.reset_index(drop=True)
                    else:
                        filtered_predict_boxes = pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'class': [], 'prob': [], 'mask': []})

                    # 　将预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                    print "predict_boxes:"
                    print filtered_predict_boxes
                    _, predict_df = get_object_stat(hu_img_array=None,
                                                    slice_object_list=filtered_predict_boxes,
                                                    img_spacing=None,
                                                    prefix=key,
                                                    classes=self.cls_name,
                                                    same_box_threshold=self.same_box_threshold_pred,
                                                    score_threshold=self.score_threshold_pred,
                                                    z_threshold=self.z_threshold_pred,
                                                    nodule_cls_weights=self.nodule_cls_weights,
                                                    if_dicom=False,
                                                    focus_priority_array=self.cls_focus_priority_array,
                                                    skip_init=True,
                                                    key_list=self.key_list,
                                                    class_key=self.class_key,
                                                    matched_key_list=self.matched_key_list)
                    print "predict_nodules:"
                    print predict_df


                    self.nodule_count += len(predict_df)
                    predict_df = predict_df.reset_index(drop=True)
                    predict_df_list.append(json_df_2_df(predict_df))


                    #统计ground truth 结节信息
                    gt_df_boxes_multi_classes = gt_df_boxes_multi_classes_dict[key]

                    if not gt_df_boxes_dict[key].empty:
                        filtered_gt_boxes_multi_classes = gt_df_boxes_multi_classes[
                            gt_df_boxes_multi_classes["prob"] >= thresh]
                        filtered_gt_boxes_multi_classes = filtered_gt_boxes_multi_classes.reset_index(drop=True)
                        print "gt_boxes_multi_classes:"
                        print filtered_gt_boxes_multi_classes
                        _, gt_df_multi_classes = get_object_stat(hu_img_array=None,
                                                                 slice_object_list=filtered_gt_boxes_multi_classes,
                                                                 img_spacing=None,
                                                                 prefix=key,
                                                                 classes=self.gt_cls_name,
                                                                 same_box_threshold=self.same_box_threshold_gt,
                                                                 score_threshold=self.score_threshold_gt,
                                                                 z_threshold=self.gt_cls_z_threshold_gt,
                                                                 nodule_cls_weights=self.gt_cls_weights,
                                                                 if_dicom=False,
                                                                 focus_priority_array=self.gt_cls_focus_priority_array,
                                                                 skip_init=True,
                                                                 key_list=self.key_list,
                                                                 class_key=self.class_key,
                                                                 matched_key_list=self.matched_key_list)
                        print 'gt_nodules_multi_classes:'
                        print gt_df_multi_classes
                        if len(gt_df_multi_classes) != 8:
                            self.series_name.append(key)

                    else:
                        gt_df_multi_classes = pd.DataFrame({'Bndbox List': [], 'Object Id': [], 'Pid': key, 'Type': [],
                                   'SliceRange': [], 'Prob': [], 'Diameter': [], 'CT_value': []})


                    gt_df_multi_classes=gt_df_multi_classes.reset_index(drop=True)
                    gt_df_multi_list.append(json_df_2_df(gt_df_multi_classes))

            summary_count_df = df_to_xlsx_file(predict_df_list,gt_df_multi_list,thresh=self.nodule_compare_thresh)

            summary_count_df = summary_count_df.sort_values(by=['PatientID'])
            summary_count_df = summary_count_df.reset_index(drop=True)

            # 统计TP FP FN  RECALL FP/TP信息
            tp_count = len(summary_count_df[summary_count_df['Result'] == 'TP'])
            fp_count = len(summary_count_df[summary_count_df['Result'] == 'FP'])
            fn_count = len(summary_count_df[summary_count_df['Result'] == 'FN'])

            recall = float(tp_count) / (tp_count + fn_count) if tp_count!=0 else 0
            fp_tp = float(fp_count) / tp_count if tp_count!=0 else np.nan
            precision = float(tp_count)/(tp_count+fp_count) if tp_count!=0 else 0
            fscore = (1+self.fscore_beta**2)*recall*precision/(self.fscore_beta**2*precision+recall) \
                        if (recall != 0 or precision != 0) else 0

            self.count_df = self.count_df.append({'class': 'nodule',
                                                  'threshold': thresh,
                                                  'nodule_count': self.nodule_count,
                                                  'tp_count': tp_count,
                                                  'fp_count': fp_count,
                                                  'fn_count': fn_count,
                                                  'accuracy': np.nan,
                                                  'recall': recall,
                                                  'precision': precision,
                                                  'fp/tp': fp_tp,
                                                  self.score_type: fscore},
                                                 ignore_index=True)

            #统计不同结节的信息
            for gt_cls in self.gt_cls_name:
                if gt_cls=='__background__':
                    continue

                tp_count = len(summary_count_df[(summary_count_df['Result'] == 'TP') & (summary_count_df['ground_truth_class'] == gt_cls)])
                fn_count = len(summary_count_df[(summary_count_df['Result'] == 'FN') & (summary_count_df['ground_truth_class'] == gt_cls)])

                recall = float(tp_count) / (tp_count + fn_count) if tp_count!=0 else 0
                self.gt_cls_count_df = self.gt_cls_count_df.append({'class': gt_cls,
                                                                    'threshold':thresh,
                                                                    'tp_count': tp_count,
                                                                    'fn_count': fn_count,
                                                                    'recall': recall
                                                                    }, ignore_index=True)
            #预处理存储数据
            for index in summary_count_df.index:
                if index == 0:
                    patientID = summary_count_df.loc[index, 'PatientID']
                else:
                    if summary_count_df.loc[index, 'PatientID'] == patientID:
                        summary_count_df.loc[index, 'PatientID'] = np.nan
                    else:
                        patientID = summary_count_df.loc[index, 'PatientID']


            self.summary_count_df[thresh]=summary_count_df

            # find the optimal threshold
            if 'nodule' not in self.opt_thresh:

                self.opt_thresh['nodule'] = self.count_df.iloc[-1]

                self.opt_thresh['nodule']["threshold"] = thresh

            else:
                # we choose the optimal threshold corresponding to the one that gives the highest model score
                if self.count_df.iloc[-1][self.score_type] > self.opt_thresh['nodule'][self.score_type]:
                    self.opt_thresh['nodule'] = self.count_df.iloc[-1]
                    self.opt_thresh['nodule']["threshold"] = thresh


        self.count_df = self.count_df.sort_values('threshold')
        self.gt_cls_count_df = self.gt_cls_count_df.sort_values(['threshold', 'class'])

        save_xlsx_json_three_sheets(self.count_df, self.gt_cls_count_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'binary-class_evaluation', 'gt_cls_evaluation', 'optimal_threshold')

        save_xlsx_sheets(self.summary_count_df,self.result_save_dir,'result.xlsx',self.json_name, columns= ['PatientID', 'PreSlices', 'Prebbox', 'GtSlices',
               'Gtbbox', 'Result', 'predict_class', 'ground_truth_class','Prob', 'Diameter', 'CT_value'])
        print self.series_name

    #先把框匹配成结节，再用阈值对结节的最高概率进行筛选
    def binary_class_evaluation_nodule_threshold(self):

        predict_df_boxes_dict, gt_df_boxes_dict, gt_df_boxes_multi_classes_dict = self.load_data()
        self.count_df = pd.DataFrame(
            columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', self.score_type])
        self.gt_cls_count_df = pd.DataFrame(
            columns=['class', 'threshold', 'tp_count', 'fn_count', 'recall'])
        self.opt_thresh = {}
        self.summary_count_df = {}
        self.nodule_count = [0 for _ in self.conf_thresh]
        self.series_name = []

        predict_df_list = []
        gt_df_multi_list = []

        for index, key in enumerate(predict_df_boxes_dict):

            self.patient_list.append(key)

            print ('processing %s' % key)

            if self.if_ensemble:
                predict_df_boxes_list = predict_df_boxes_dict[key]
                ground_truth_boxes = gt_df_boxes_dict[key]
                filtered_predict_df_boxes_list = []

                for model_idx, _ in enumerate(predict_df_boxes_list):
                    # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                    if not predict_df_boxes_list[model_idx].empty:
                        filtered_predict_df_boxes_list.append(
                            predict_df_boxes_list[model_idx])
                        # print filtered_predict_boxes_list[model_idx]
                        filtered_predict_df_boxes_list[model_idx] = filtered_predict_df_boxes_list[model_idx].reset_index(
                            drop=True)
                    else:
                        filtered_predict_df_boxes_list.append(pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'class': [], 'prob': [], 'mask': []}))

                # if not gt_df_boxes_dict[key].empty:
                #     # print filtered_gt_boxes
                #     filtered_gt_boxes = ground_truth_boxes.reset_index(drop=True)
                # else:
                #     filtered_gt_boxes = pd.DataFrame(
                #         {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                #          'class': [], 'prob': [], 'mask': []})

                # 将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                print "predict_boxes:"
                print predict_df_boxes_list
                _, predict_df = get_object_stat(
                    hu_img_array=None,
                    slice_object_list=filtered_predict_df_boxes_list,
                    img_spacing=None,
                    prefix=key,
                    classes=self.cls_name,
                    same_box_threshold=self.same_box_threshold_pred,
                    score_threshold=self.score_threshold_pred,
                    z_threshold=self.z_threshold_pred,
                    nodule_cls_weights=self.nodule_cls_weights,
                    if_dicom=False,
                    focus_priority_array=self.cls_focus_priority_array,
                    skip_init=True,
                    key_list=self.key_list,
                    class_key=self.class_key,
                    matched_key_list=self.matched_key_list,
                    if_ensemble=self.if_ensemble,
                    model_weight_list=self.model_weight_list,
                    model_conf_list=self.model_cof_list,
                    obj_freq_thresh=self.obj_freq_thresh)
                print "predict_nodules:"
                print predict_df
                for i, thresh in enumerate(self.conf_thresh):
                    self.nodule_count[i] += len(predict_df[predict_df['Prob'] >= thresh])
                predict_df = predict_df.reset_index(drop=True)
                predict_df_list.append(json_df_2_df(predict_df))

                # 统计ground truth 结节信息
                gt_df_boxes_multi_classes = gt_df_boxes_multi_classes_dict[key]

                if not gt_df_boxes_dict[key].empty:
                    gt_df_boxes_multi_classes = gt_df_boxes_multi_classes.reset_index(drop=True)
                    print "gt_df_boxes_multi_classes:"
                    print gt_df_boxes_multi_classes
                    _, gt_df_multi_classes = get_object_stat(hu_img_array=None,
                                                             slice_object_list=gt_df_boxes_multi_classes,
                                                             img_spacing=None,
                                                             prefix=key,
                                                             classes=self.gt_cls_name,
                                                             same_box_threshold=self.same_box_threshold_gt,
                                                             score_threshold=self.score_threshold_gt,
                                                             z_threshold=self.gt_cls_z_threshold_gt,
                                                             nodule_cls_weights=self.gt_cls_weights,
                                                             if_dicom=False,
                                                             focus_priority_array=self.gt_cls_focus_priority_array,
                                                             skip_init=True,
                                                             key_list=self.key_list,
                                                             class_key=self.class_key,
                                                             matched_key_list=self.matched_key_list)
                    print 'gt_nodules_multi_classes:'
                    print gt_df_multi_classes
                    if len(gt_df_multi_classes) != 8:
                        self.series_name.append(key)
            else:
                predict_df_boxes = predict_df_boxes_dict[key]
                # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                if not predict_df_boxes_dict[key].empty:
                    filtered_predict_boxes =predict_df_boxes.reset_index(drop=True)
                else:
                    filtered_predict_boxes = pd.DataFrame(
                        {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                         'class': [], 'prob': [], 'mask': []})

                # 　将预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                print "predict_boxes:"
                print filtered_predict_boxes
                _, predict_df = get_object_stat(hu_img_array=None,
                                                slice_object_list=filtered_predict_boxes,
                                                img_spacing=None,
                                                prefix=key,
                                                classes=self.cls_name,
                                                same_box_threshold=self.same_box_threshold_pred,
                                                score_threshold=self.score_threshold_pred,
                                                z_threshold=self.z_threshold_pred,
                                                nodule_cls_weights=self.nodule_cls_weights,
                                                if_dicom=False,
                                                focus_priority_array=self.cls_focus_priority_array,
                                                skip_init=True,
                                                key_list=self.key_list,
                                                class_key=self.class_key,
                                                matched_key_list=self.matched_key_list)
                print "predict_nodules:"
                print predict_df
                for i, thresh in enumerate(self.conf_thresh):
                    self.nodule_count[i] += len(predict_df[predict_df['Prob'] >= thresh])
                predict_df = predict_df.reset_index(drop=True)
                predict_df_list.append(json_df_2_df(predict_df))

                # 统计ground truth 结节信息
                gt_df_boxes_multi_classes = gt_df_boxes_multi_classes_dict[key]

                if not gt_df_boxes_dict[key].empty:
                    filtered_gt_boxes_multi_classes = gt_df_boxes_multi_classes.reset_index(drop=True)
                    print "gt_boxes_multi_classes:"
                    print filtered_gt_boxes_multi_classes
                    _, gt_df_multi_classes = get_object_stat(
                                                             hu_img_array=None,
                                                             slice_object_list=filtered_gt_boxes_multi_classes,
                                                             img_spacing=None,
                                                             prefix=key,
                                                             classes=self.gt_cls_name,
                                                             same_box_threshold=self.same_box_threshold_gt,
                                                             score_threshold=self.score_threshold_gt,
                                                             z_threshold=self.gt_cls_z_threshold_gt,
                                                             nodule_cls_weights=self.gt_cls_weights,
                                                             if_dicom=False,
                                                             focus_priority_array=self.gt_cls_focus_priority_array,
                                                             skip_init=True,
                                                             key_list=self.key_list,
                                                             class_key=self.class_key,
                                                             matched_key_list=self.matched_key_list
                                                             )
                    print 'gt_df_multi_classes:'
                    print gt_df_multi_classes
                else:
                    gt_df_multi_classes = pd.DataFrame({'Bndbox List': [], 'Object Id': [], 'Pid': key, 'Type': [],
                                                        'SliceRange': [], 'Prob': [], 'Diameter': [], 'CT_value': []})

            gt_df_multi_classes = gt_df_multi_classes.reset_index(drop=True)
            gt_df_multi_list.append(json_df_2_df(gt_df_multi_classes))

        summary_count_dfs = df_to_xlsx_file(predict_df_list, gt_df_multi_list, thresh=self.nodule_compare_thresh)

        print summary_count_dfs

        gt_count = 0
        gt_cls_count = [0 for _ in range(len(self.gt_cls_name))]

        for i, thresh in enumerate(self.conf_thresh):

            if thresh == self.conf_thresh[0]:
                gt_count = len(summary_count_dfs[summary_count_dfs['Result'] == 'TP']) + \
                           len(summary_count_dfs[summary_count_dfs['Result'] == 'FN'])


            summary_count_df=summary_count_dfs[summary_count_dfs['Prob']>=thresh]

            summary_count_df = summary_count_df.sort_values(by=['PatientID'])
            summary_count_df = summary_count_df.reset_index(drop=True)

            # 统计TP FP FN  RECALL FP/TP信息
            tp_count = len(summary_count_df[summary_count_df['Result'] == 'TP'])
            fp_count = len(summary_count_df[summary_count_df['Result'] == 'FP'])
            fn_count = gt_count - tp_count

            recall = float(tp_count) / (tp_count + fn_count) if tp_count != 0 else 0
            fp_tp = float(fp_count) / tp_count if tp_count != 0 else np.nan
            precision = float(tp_count) / (tp_count + fp_count) if tp_count != 0 else 0
            fscore = (1 + self.fscore_beta ** 2) * recall * precision / (self.fscore_beta ** 2 * precision + recall) \
                if (recall != 0 or precision != 0) else 0

            self.count_df = self.count_df.append({'class': 'nodule',
                                                  'threshold': thresh,
                                                  'nodule_count': self.nodule_count[i],
                                                  'tp_count': tp_count,
                                                  'fp_count': fp_count,
                                                  'fn_count': fn_count,
                                                  'accuracy': np.nan,
                                                  'recall': recall,
                                                  'precision': precision,
                                                  'fp/tp': fp_tp,
                                                  self.score_type: fscore},
                                                  ignore_index=True)

            # 统计不同结节的信息
            for i_gt_cls, gt_cls in enumerate(self.gt_cls_name):
                if gt_cls == '__background__':
                    continue

                tp_count = len(summary_count_df[(summary_count_df['Result'] == 'TP') & (
                            summary_count_df['ground_truth_class'] == gt_cls)])
                fn_count = len(summary_count_df[(summary_count_df['Result'] == 'FN') & (
                            summary_count_df['ground_truth_class'] == gt_cls)])

                if thresh == self.conf_thresh[0]:
                    gt_cls_count[i_gt_cls] = tp_count + fn_count

                recall = float(tp_count) / (tp_count + fn_count) if tp_count != 0 else 0
                self.gt_cls_count_df = self.gt_cls_count_df.append({'class': gt_cls,
                                                                    'threshold': thresh,
                                                                    'tp_count': tp_count,
                                                                    'fn_count': gt_cls_count[i_gt_cls] - tp_count,
                                                                    'recall': recall
                                                                    }, ignore_index=True)
            # 预处理存储数据
            for index in summary_count_df.index:
                if index == 0:
                    patientID = summary_count_df.loc[index, 'PatientID']
                else:
                    if summary_count_df.loc[index, 'PatientID'] == patientID:
                        summary_count_df.loc[index, 'PatientID'] = np.nan
                    else:
                        patientID = summary_count_df.loc[index, 'PatientID']

            self.summary_count_df[thresh] = summary_count_df

            # find the optimal threshold
            if 'nodule' not in self.opt_thresh:

                self.opt_thresh['nodule'] = self.count_df.iloc[-1]

                self.opt_thresh['nodule']["threshold"] = thresh

            else:
                # we choose the optimal threshold corresponding to the one that gives the highest model score
                if self.count_df.iloc[-1][self.score_type] > self.opt_thresh['nodule'][self.score_type]:
                    self.opt_thresh['nodule'] = self.count_df.iloc[-1]
                    self.opt_thresh['nodule']["threshold"] = thresh

        self.count_df = self.count_df.sort_values('threshold')
        self.gt_cls_count_df = self.gt_cls_count_df.sort_values(['threshold', 'class'])

        save_xlsx_json_three_sheets(self.count_df, self.gt_cls_count_df, self.opt_thresh, self.result_save_dir,
                                    self.xlsx_name, self.json_name,
                                    'binary-class_evaluation', 'gt_cls_evaluation', 'optimal_threshold')

        save_xlsx_sheets(self.summary_count_df, self.result_save_dir, 'result.xlsx',self.json_name, columns= ['PatientID', 'PreSlices', 'Prebbox', 'GtSlices',
               'Gtbbox', 'Result', 'predict_class', 'ground_truth_class','Prob', 'Diameter', 'CT_value'])

    # 读入预测结果数据

    def load_data(self):
        """
        读入模型输出的.json和ground truth的.xml标记
        :return: 模型预测结果、ground truth标记按病人号排列的pandas.DataFrame
        e.g. | mask | instanceNumber | class | prob | sliceId | xmax | xmin | ymax | ymin |
             |  []  |       106      | solid nodule | 0.9  | 105.0   | 207.0| 182.0| 230.0| 205.0|
        """
        predict_df_anchors_dict = {}
        ground_truth_anchors_dict = {}
        ground_truth_anchors_multi_classes_dict = {}
        #　若为多模型ensemble评估,上述字典的每个病人号关键词对应一个列表,列表中每个元素为一个模型对该病人的预测结果,格式为pandas.Dataframe
        if self.if_ensemble:
            for model_idx, model_id in enumerate(self.model_list):
                # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
                for PatientID in os.listdir(os.path.join(self.data_dir, model_id)):
                    if model_idx == 0:
                        predict_df_anchors_dict[PatientID] = []
                        ground_truth_anchors_dict[PatientID] = []
                        ground_truth_anchors_multi_classes_dict[PatientID] = []
                    if self.data_type == 'json':
                        predict_json_path = os.path.join(self.data_dir, model_id, PatientID, PatientID + '_predict.json')
                        try:
                            predict_df_anchors = pd.read_json(predict_json_path).T
                            predict_df_anchors = predict_df_anchors.rename(index=str, columns={'nodule_class': 'class'})
                        except:
                            print (
                            "broken directory structure, maybe no prediction json file found: %s" % predict_json_path)
                            raise NameError
                        try:
                            check_insnum_sliceid(predict_df_anchors)
                        except:
                            logging.exception('%s has inconsistent instanceNumber and sliceId' % PatientID)
                    elif self.data_type == 'npy':
                        predict_npy_path = os.path.join(self.data_dir, model_id, PatientID, PatientID + '_predict.npy')
                        try:
                            predict_anchors = np.load(predict_npy_path)
                        except:
                            print ("broken directory structure, maybe no prediction npy file found: %s" % predict_npy_path)
                            raise NameError
                        predict_df_anchors = init_df_objects(slice_object_list=predict_anchors, key_list=self.key_list,
                                                             class_key=self.class_key)
                        predict_df_anchors = predict_df_anchors.sort_values(by=['prob'])
                        predict_df_anchors = predict_df_anchors.reset_index(drop=True)
                    else:
                        # 　尚未考虑其他数据存储格式，有需要的话日后添加
                        raise NotImplemented

                    if model_idx == 0:
                        ground_truth_path = os.path.join(self.anno_dir, PatientID)
                        # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                        ground_truth_anchors = xml_to_anchorlist(config=self.config, xml_dir=ground_truth_path)
                        # except:
                        #     print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                        #     ground_truth_anchors = []


                        # 对于ground truth boxes,我们直接读取其xml标签,并保留原始的结节细分类别。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                        ground_truth_anchors_multi_classes = xml_to_anchorlist_multi_classes(config=self.config,
                                                                                             xml_dir=ground_truth_path)
                        # except:
                        #     print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                        #     ground_truth_anchors_multi_classes = []

                        ground_truth_anchors = init_df_objects(slice_object_list=ground_truth_anchors, key_list=self.key_list,
                                                               class_key=self.class_key)
                        ground_truth_anchors = ground_truth_anchors.sort_values(by=['instanceNumber'])
                        ground_truth_anchors = ground_truth_anchors.reset_index(drop=True)

                        ground_truth_anchors_multi_classes = init_df_objects(
                            slice_object_list=ground_truth_anchors_multi_classes, key_list=self.key_list,
                            class_key=self.class_key)

                        ground_truth_anchors_multi_classes = ground_truth_anchors_multi_classes.sort_values(
                            by=['instanceNumber'])
                        ground_truth_anchors_multi_classes = ground_truth_anchors_multi_classes.reset_index(drop=True)

                        ground_truth_anchors_dict[PatientID] = ground_truth_anchors
                        ground_truth_anchors_multi_classes_dict[PatientID] = ground_truth_anchors_multi_classes

                    predict_df_anchors_dict[PatientID].append(predict_df_anchors)

        else:
            # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
            for PatientID in os.listdir(self.data_dir):
                if self.data_type == 'json':
                    predict_json_path = os.path.join(self.data_dir, PatientID, PatientID + '_predict.json')
                    try:
                        predict_df_anchors = pd.read_json(predict_json_path).T
                        predict_df_anchors = predict_df_anchors.rename(index=str, columns={'nodule_class': 'class'})
                    except:
                        print ("broken directory structure, maybe no prediction json file found: %s" % predict_json_path)
                        raise NameError
                    try:
                        check_insnum_sliceid(predict_df_anchors)
                    except:
                        logging.exception('%s has inconsistent instanceNumber and sliceId' %PatientID)
                elif self.data_type == 'npy':
                    predict_npy_path = os.path.join(self.data_dir, PatientID, PatientID + '_predict.npy')
                    try:
                        predict_anchors = np.load(predict_npy_path)
                    except:
                        print ("broken directory structure, maybe no prediction npy file found: %s" % predict_npy_path)
                        raise NameError
                    predict_df_anchors = init_df_objects(slice_object_list=predict_anchors, key_list=self.key_list,
                                                         class_key=self.class_key)
                    predict_df_anchors = predict_df_anchors.sort_values(by=['prob'])
                    predict_df_anchors = predict_df_anchors.reset_index(drop=True)
                else:
                    # 　尚未考虑其他数据存储格式，有需要的话日后添加
                    raise NotImplemented

                ground_truth_path = os.path.join(self.anno_dir, PatientID)
                # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                ground_truth_anchors = xml_to_anchorlist(config=self.config, xml_dir=ground_truth_path)
                # except:
                #     print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                #     ground_truth_anchors = []


                    # 对于ground truth boxes,我们直接读取其xml标签,并保留原始的结节细分类别。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                ground_truth_anchors_multi_classes = xml_to_anchorlist_multi_classes(config=self.config, xml_dir=ground_truth_path)
                # except:
                #     print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                #     ground_truth_anchors_multi_classes = []

                ground_truth_anchors = init_df_objects(slice_object_list=ground_truth_anchors, key_list=self.key_list,
                                                         class_key=self.class_key)
                ground_truth_anchors = ground_truth_anchors.sort_values(by=['instanceNumber'])
                ground_truth_anchors = ground_truth_anchors.reset_index(drop=True)

                ground_truth_anchors_multi_classes = init_df_objects(slice_object_list=ground_truth_anchors_multi_classes, key_list=self.key_list,
                                                                   class_key=self.class_key)
                ground_truth_anchors_multi_classes = ground_truth_anchors_multi_classes.sort_values(by=['instanceNumber'])
                ground_truth_anchors_multi_classes = ground_truth_anchors_multi_classes.reset_index(drop=True)

                predict_df_anchors_dict[PatientID] = predict_df_anchors
                ground_truth_anchors_dict[PatientID] = ground_truth_anchors
                ground_truth_anchors_multi_classes_dict[PatientID] = ground_truth_anchors_multi_classes
        return predict_df_anchors_dict, ground_truth_anchors_dict, ground_truth_anchors_multi_classes_dict

    # 由predict出的框和ground truth anno生成_nodules.json和_gt.json
    def generate_df_nodules_to_json(self):
        """
        读入_predict.json及gt annotation文件，经过get_nodule_stat转换为json文件并存储到指定目录
        """

        predict_df_boxes_dict, ground_truth_boxes_dict, _ = self.load_data()

        # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
        for PatientID in os.listdir(self.data_dir):
            predict_df_boxes = predict_df_boxes_dict[PatientID]
            ground_truth_boxes = ground_truth_boxes_dict[PatientID]

            if predict_df_boxes.empty:
                predict_df_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                   'class': [], 'prob': [], 'mask': []})
            else:
                predict_df_boxes = predict_df_boxes.reset_index(drop=True)

            if ground_truth_boxes.empty:
                ground_truth_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                   'class': [], 'prob': [], 'mask': []})
            else:
                ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            print "prediction:"
            _, predict_df = get_object_stat(dicom_names=None,
                                            hu_img_array=None,
                                            slice_object_list=predict_df_boxes,
                                            img_spacing=None,
                                            prefix=PatientID,
                                            classes=self.cls_name,
                                            same_box_threshold=self.same_box_threshold_pred,
                                            score_threshold=self.score_threshold_pred,
                                            z_threshold=self.z_threshold_pred,
                                            nodule_cls_weights=self.nodule_cls_weights,
                                            if_dicom=False,
                                            focus_priority_array=None,
                                            skip_init=True,
                                            key_list=self.key_list,
                                            class_key=self.class_key,
                                            matched_key_list=self.matched_key_list
                                            )
            print "ground truth"
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)
            _, gt_df = get_object_stat(dicom_names=None,
                                       hu_img_array=None,
                                       slice_object_list=ground_truth_boxes,
                                       img_spacing=None,
                                       prefix=PatientID,
                                       classes=self.cls_name,
                                       same_box_threshold=self.same_box_threshold_gt,
                                       score_threshold=self.score_threshold_gt,
                                       z_threshold=self.z_threshold_gt,
                                       nodule_cls_weights=self.nodule_cls_weights,
                                       if_dicom=False,
                                       focus_priority_array=None,
                                       skip_init=True,
                                       key_list=self.key_list,
                                       class_key=self.class_key,
                                       matched_key_list=self.matched_key_list
                                       )
            str_nodules = predict_df.T.to_json()
            str_gt = gt_df.T.to_json()
            if not os.path.exists(self.result_save_dir):
                os.mkdir(self.result_save_dir)
            json_patient_dir = os.path.join(self.result_save_dir, PatientID)
            print ('processing patient: %s' %PatientID)
            print json_patient_dir
            if not os.path.exists(json_patient_dir):
                os.mkdir(json_patient_dir)
            with open(os.path.join(json_patient_dir, PatientID + '_nodule.json'), "w") as fp:
                js_nodules = json.loads(str_nodules, "utf-8")
                json.dump(js_nodules, fp)
            with open(os.path.join(json_patient_dir, PatientID + '_gt.json'), "w") as fp:
                js_gt = json.loads(str_gt, "utf-8")
                json.dump(js_gt, fp)

    # 筛选一定层厚以上的最终输出的结节（降假阳实验）
    def nodule_thickness_filter(self):
        assert type(self.thickness_thresh) == int, "input thickness_thresh should be an integer, not %s" %self.thickness_thresh
        for PatientID in os.listdir(self.data_dir):
            if self.data_type == 'json':
                predict_json_path = os.path.join(self.result_save_dir, PatientID, PatientID + '_nodule.json')
                try:
                    predict_df_boxes = pd.read_json(predict_json_path).T
                except:
                    raise ("broken directory structure, maybe no prediction json file found: %s" % predict_json_path)
            drop_list = []
            for i, row in predict_df_boxes.iterrows():
                if len(row['SliceRange']) <= self.thickness_thresh:
                    drop_list.append(i)
            predict_df_boxes = predict_df_boxes.drop(drop_list)

            str_nodules = predict_df_boxes.T.to_json()
            if not os.path.exists(self.result_save_dir):
                os.mkdir(self.result_save_dir)
            json_patient_dir = os.path.join(self.result_save_dir, PatientID)
            print ('processing patient: %s' %PatientID)
            print json_patient_dir
            if not os.path.exists(json_patient_dir):
                os.mkdir(json_patient_dir)
            with open(os.path.join(json_patient_dir, PatientID + '_nodule%s.json' %(self.thickness_thresh)), "w") as fp:
                js_nodules = json.loads(str_nodules, "utf-8")
                json.dump(js_nodules, fp)

class LungNoduleEvaluatorOffline(object):
    '''
    this class is designed for evaluation of our CT lung model offline. It can read anchor boxes from a selection of format (.json/.npy)
    and generate spreadsheets of statistics (tp, fp, etc. see common/custom_metric) for each nodule class under customized
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
    :param nodule_cls_weights:　不同结节种类对于模型综合评分以及objmatch.find_nodules算法中的权重，默认与结节分类信息一起从classname_labelname_mapping.xls中读取,类型为dict
    :param cls_weight: 在求加权平均结果时，每个类别的权重，类型为list
    :param cls_value: 在求加权平均结果时，每个类别的得分，类型为list
    :param thickness_thresh: nodule_thickness_filter根据此阈值对结节的层厚进行筛选
    :param nodule_compare_thresh: 比较两个结节是否算一个的IOU阈值
    '''
    def __init__(self, cls_label_xls_path, data_dir, data_type, anno_dir, score_type = 'fscore',  result_save_dir = os.path.join(os.getcwd(), 'LungNoduleEvaluation_result'),
                 xlsx_name = 'LungNoduleEvaluation.xlsx', json_name = 'LungNoduleEvaluation', if_nodule_json = False,
                 conf_thresh = np.linspace(0.1, 0.9, num=9).tolist(), fscore_beta = 1.,
                 same_box_threshold_pred = np.array([1.6, 1.6]), same_box_threshold_gt = np.array([0., 0.]),
                 score_threshold_pred = 0.6, score_threshold_gt = 0.4, if_nodule_threshold = False, thickness_thresh = 0.,
                 cls_focus_priority_array = {"mass": 6,
                                            "calcific nodule": 5,
                                            "solid nodule": 4,
                                            "GGN": 3,
                                            "0-3nodule": 2,
                                            "nodule": 1},
                 gt_cls_focus_priority_array = {"mass": 9,
                                                "10-30nodule": 8,
                                                "6-10nodule": 7,
                                                "calcific nodule": 6,
                                                "pleural nodule": 5,
                                                "3-6nodule": 4,
                                                "5GGN": 3,
                                                "0-5GGN": 2,
                                                "0-3nodule": 1}):
        self.config = LungConfig(cls_label_xls_path=cls_label_xls_path)
        assert os.path.isdir(data_dir), 'must initialize it with a valid directory of bbox data'
        self.data_dir = data_dir
        self.data_type = data_type
        self.anno_dir = anno_dir
        # config.CLASSES 包含background class,是结节的粗分类(RCNN分类)
        self.cls_name = self.config.CLASSES
        # config.NODULE_CLASSES 不包含background class,是结节的细分类(ground truth label分类)
        self.gt_cls_name = self.config.NODULE_CLASSES
        self.cls_dict = self.config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.count_df = pd.DataFrame(
                     columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                              'accuracy', 'recall', 'precision',
                              'fp/tp', self.score_type])
        self.gt_cls_count_df = pd.DataFrame(
                     columns=['class', 'threshold', 'tp_count', 'fn_count', 'recall'])
        self.summary_count_df={}
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.if_nodule_json = if_nodule_json
        # customized confidence threshold for plotting ROC curve
        self.conf_thresh = conf_thresh
        self.nodule_cls_weights = self.config.CLASS_WEIGHTS
        self.gt_cls_weights = self.config.GT_CLASSES_WEIGHTS
        self.fscore_beta = fscore_beta
        self.patient_list = []
        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
        # objmatch.find_nodules/find_objects算法的相关超参数，详见config文件
        self.same_box_threshold_pred = same_box_threshold_pred
        self.same_box_threshold_gt = same_box_threshold_gt
        self.score_threshold_pred = score_threshold_pred
        self.score_threshold_gt = score_threshold_gt
        self.z_threshold_pred = self.config.CLASS_Z_THRESHOLD_PRED
        self.z_threshold_gt = self.config.CLASS_Z_THRESHOLD_GT
        self.gt_cls_z_threshold_gt = self.config.GT_CLASS_Z_THRESHOLD_GT
        self.if_nodule_threshold = if_nodule_threshold

        self.thickness_thresh = thickness_thresh
        self.nodule_compare_thresh = self.config.TEST.OBJECT_COMPARE_THRESHOLD

        # keep track of the nodule count in the output of get_df_nodules, including false positives, initialized to be 0
        self.nodule_count = 0.
        self.cls_focus_priority_array = cls_focus_priority_array
        self.gt_cls_focus_priority_array = gt_cls_focus_priority_array

    # 多分类模型评分,每次只选取单类别的检出框，把其余所有类别作为负样本。
    def multi_class_evaluation(self):

        predict_df_boxes_dict, ground_truth_boxes_dict, _ = self.load_data()
        self.count_df = pd.DataFrame(
            columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', self.score_type])
        self.opt_thresh = {}

        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            self.cls_weight = []
            self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
            for i_cls, cls in enumerate(self.cls_name):
                if cls == "__background__":
                    continue
                # construct class weight list, for computing class-average result
                self.cls_weight.append(self.nodule_cls_weights[cls])

                cls_predict_df_list = []
                cls_gt_df_list = []
                self.nodule_count = 0.
                for index, key in enumerate(predict_df_boxes_dict):
                    self.patient_list.append(key)
                    predict_df_boxes = predict_df_boxes_dict[key]
                    ground_truth_boxes = ground_truth_boxes_dict[key]

                    print ('nodule class: %s' %cls)
                    print ('processing %s' %key)

                    #　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                    if not predict_df_boxes_dict[key].empty:
                        filtered_predict_boxes = predict_df_boxes[predict_df_boxes["class"] == cls]
                        print filtered_predict_boxes
                        filtered_predict_boxes = filtered_predict_boxes[filtered_predict_boxes["prob"] >= thresh]
                        filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                    else:
                        filtered_predict_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                 'class': [], 'prob': [], 'mask': []})

                    if not ground_truth_boxes_dict[key].empty:
                        filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["class"] == cls]
                        print filtered_gt_boxes
                        filtered_gt_boxes = filtered_gt_boxes[filtered_gt_boxes["prob"] >= thresh]
                        filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                    else:
                        filtered_gt_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                 'class': [], 'prob': [], 'mask': []})

                    #　将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                    print "predict_boxes:"
                    print filtered_predict_boxes
                    _, cls_predict_df = get_nodule_stat(dicom_names=None,
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
                                             focus_priority_array=self.cls_focus_priority_array,
                                             skip_init=True)
                    print "predict_nodules:"
                    print cls_predict_df

                    print "gt_boxes:"
                    print filtered_gt_boxes
                    _, cls_gt_df = get_nodule_stat(dicom_names=None,
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
                                                     focus_priority_array=self.cls_focus_priority_array,
                                                     skip_init=True)
                    print "gt_nodules:"
                    print cls_gt_df

                    cls_predict_df = cls_predict_df.reset_index(drop=True)
                    cls_predict_df_list.append(json_df_2_df(cls_predict_df))

                    cls_gt_df = cls_gt_df.reset_index(drop=True)
                    cls_gt_df_list.append(json_df_2_df(cls_gt_df))

                    self.nodule_count += len(cls_predict_df.index)

                # convert pandas dataframe to list of class labels
                cls_pred_labels, cls_gt_labels = df_to_cls_label(cls_predict_df_list, cls_gt_df_list, self.cls_name, thresh=self.nodule_compare_thresh)

                # initialize ClassificationMetric class and update with ground truth/predict labels
                cls_metric = ClassificationMetric(cls_num=len(self.cls_name)-1, if_binary=True, pos_cls_fusion=False)

                cls_metric.update(cls_gt_labels, cls_pred_labels, i_cls)

                if cls_metric.tp[i_cls-1] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric.fp[i_cls-1] / cls_metric.tp[i_cls-1]

                self.count_df = self.count_df.append({'class': cls,
                                                      'threshold': thresh,
                                                      'nodule_count': self.nodule_count,
                                                      'tp_count': cls_metric.tp[i_cls-1],
                                                      'fp_count': cls_metric.fp[i_cls-1],
                                                      'fn_count': cls_metric.fn[i_cls-1],
                                                      'accuracy': cls_metric.get_acc(i_cls),
                                                      'recall': cls_metric.get_rec(i_cls),
                                                      'precision': cls_metric.get_prec(i_cls),
                                                      'fp/tp': fp_tp,
                                                      self.score_type: cls_metric.get_fscore(cls_label=i_cls, beta=self.fscore_beta)},
                                                      ignore_index = True)

                # find the optimal threshold
                if cls not in self.opt_thresh:

                    self.opt_thresh[cls] = self.count_df.iloc[-1]


                    self.opt_thresh[cls]["threshold"] = thresh

                else:
                    # we choose the optimal threshold corresponding to the one that gives the highest model score
                    if self.count_df.iloc[-1][self.score_type] > self.opt_thresh[cls][self.score_type]:

                        self.opt_thresh[cls] = self.count_df.iloc[-1]
                        self.opt_thresh[cls]["threshold"] = thresh

                self.cls_value['accuracy'].append(cls_metric.get_acc(i_cls))
                self.cls_value['recall'].append(cls_metric.get_rec(i_cls))
                self.cls_value['precision'].append(cls_metric.get_prec(i_cls))
                self.cls_value[self.score_type].append(cls_metric.get_fscore(cls_label=i_cls, beta=self.fscore_beta))

            #增加多类别加权平均的结果
            self.count_df = self.count_df.append({'class': 'average',
                                                  'threshold': thresh,
                                                  'tp_count': np.nan,
                                                  'fp_count': np.nan,
                                                  'fn_count': np.nan,
                                                  'accuracy': cls_avg(self.cls_weight, self.cls_value['accuracy']),
                                                  'recall': cls_avg(self.cls_weight, self.cls_value['recall']),
                                                  'precision': cls_avg(self.cls_weight, self.cls_value['precision']),
                                                  'fp/tp': np.nan,
                                                  self.score_type: cls_avg(self.cls_weight, self.cls_value[self.score_type])},
                                                  ignore_index=True)
        self.count_df = self.count_df.sort_values(['class', 'threshold'])

        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
        for key in self.opt_thresh:
            self.cls_value['accuracy'].append(self.opt_thresh[key]['accuracy'])
            self.cls_value['recall'].append(self.opt_thresh[key]['recall'])
            self.cls_value['precision'].append(self.opt_thresh[key]['precision'])
            self.cls_value[self.score_type].append(self.opt_thresh[key][self.score_type])
            self.cls_weight.append(self.nodule_cls_weights[key])

        opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        opt_thresh = opt_thresh.append({'class': 'average',
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

        save_xlsx_json(self.count_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'multi-class_evaluation', 'optimal_threshold')
        # if not os.path.exists(self.result_save_dir):
        #     os.makedirs(self.result_save_dir)
        # print ("saving %s" %os.path.join(self.result_save_dir, self.xlsx_name))
        #
        # # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        # if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name)):
        #     os.remove(os.path.join(self.result_save_dir, self.xlsx_name))
        # writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name))
        # self.count_df.to_excel(writer, 'multi-class_evaluation', index=False)
        #
        # opt_thresh = opt_thresh.reset_index(drop=True)
        # opt_thresh.to_excel(writer, 'optimal_threshold')
        # writer.save()
        #
        # print ("saving %s" %os.path.join(self.result_save_dir, self.json_name))
        # # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        # if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation.json')):
        #     os.remove(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation.json'))
        # if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json')):
        #     os.remove(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json'))
        #
        # json_count_df = self.count_df.T.to_json()
        # with open(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation.json'), "w") as fp:
        #     js_count_df = json.loads(json_count_df, "utf-8")
        #     json.dump(js_count_df, fp)
        #
        # json_opt_thresh = opt_thresh.T.to_json()
        # with open(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json'), "w") as fp:
        #     js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
        #     json.dump(js_opt_thresh, fp)


    # 多分类模型评分,每次只选取单类别的检出框，把其余所有类别作为负样本。先把框匹配成结节，再用阈值对结节的最高概率进行筛选
    def multi_class_evaluation_nodule_threshold(self):

        predict_df_boxes_dict, ground_truth_boxes_dict, _ = self.load_data()
        self.count_df = pd.DataFrame(
            columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', self.score_type])
        self.opt_thresh = {}

        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            self.cls_weight = []
            self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
            for i_cls, cls in enumerate(self.cls_name):
                if cls == "__background__":
                    continue
                # construct class weight list, for computing class-average result
                self.cls_weight.append(self.nodule_cls_weights[cls])

                cls_predict_df_list = []
                cls_gt_df_list = []
                self.nodule_count = 0.
                for index, key in enumerate(predict_df_boxes_dict):
                    self.patient_list.append(key)
                    predict_df_boxes = predict_df_boxes_dict[key]
                    ground_truth_boxes = ground_truth_boxes_dict[key]

                    print ('nodule class: %s' % cls)
                    print ('processing %s' % key)

                    # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                    if not predict_df_boxes_dict[key].empty:
                        filtered_predict_boxes = predict_df_boxes[predict_df_boxes["class"] == cls]
                        print filtered_predict_boxes
                        filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                    else:
                        filtered_predict_boxes = pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'class': [], 'prob': [], 'mask': []})

                    if not ground_truth_boxes_dict[key].empty:
                        filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["class"] == cls]
                        print filtered_gt_boxes
                        filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                    else:
                        filtered_gt_boxes = pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'class': [], 'prob': [], 'mask': []})

                    # 将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                    print "predict_boxes:"
                    print filtered_predict_boxes
                    _, cls_predict_df = get_nodule_stat(dicom_names=None,
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
                                                        focus_priority_array=self.cls_focus_priority_array,
                                                        skip_init=True)
                    print "predict_nodules:"
                    print cls_predict_df

                    print "gt_boxes:"
                    print filtered_gt_boxes

                    _, cls_gt_df = get_nodule_stat(dicom_names=None,
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
                                                   focus_priority_array=self.cls_focus_priority_array,
                                                   skip_init=True)

                    print "gt_nodules:"
                    print cls_gt_df

                    cls_predict_df = cls_predict_df[cls_predict_df['Prob'] >= thresh]
                    cls_predict_df = cls_predict_df.reset_index(drop=True)
                    cls_predict_df_list.append(json_df_2_df(cls_predict_df))

                    cls_gt_df = cls_gt_df[cls_gt_df['Prob'] >= thresh]
                    cls_gt_df = cls_gt_df.reset_index(drop=True)
                    cls_gt_df_list.append(json_df_2_df(cls_gt_df))

                    self.nodule_count += len(cls_predict_df.index)

                # convert pandas dataframe to list of class labels
                cls_pred_labels, cls_gt_labels = df_to_cls_label(cls_predict_df_list, cls_gt_df_list, self.cls_name,
                                                                 thresh=self.nodule_compare_thresh)

                # initialize ClassificationMetric class and update with ground truth/predict labels
                cls_metric = ClassificationMetric(cls_num=len(self.cls_name)-1, if_binary=True, pos_cls_fusion=False)

                cls_metric.update(cls_gt_labels, cls_pred_labels, i_cls)

                if cls_metric.tp[i_cls-1] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric.fp[i_cls-1] / cls_metric.tp[i_cls-1]

                self.count_df = self.count_df.append({'class': cls,
                                                      'threshold': thresh,
                                                      'nodule_count': self.nodule_count,
                                                      'tp_count': cls_metric.tp[i_cls-1],
                                                      'fp_count': cls_metric.fp[i_cls-1],
                                                      'fn_count': cls_metric.fn[i_cls-1],
                                                      'accuracy': cls_metric.get_acc(i_cls),
                                                      'recall': cls_metric.get_rec(i_cls),
                                                      'precision': cls_metric.get_prec(i_cls),
                                                      'fp/tp': fp_tp,
                                                      self.score_type: cls_metric.get_fscore(cls_label=i_cls, beta=self.fscore_beta)},
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

                self.cls_value['accuracy'].append(cls_metric.get_acc(i_cls))
                self.cls_value['recall'].append(cls_metric.get_rec(i_cls))
                self.cls_value['precision'].append(cls_metric.get_prec(i_cls))
                self.cls_value[self.score_type].append(cls_metric.get_fscore(cls_label=i_cls, beta=self.fscore_beta))

            # 增加多类别加权平均的结果
            self.count_df = self.count_df.append({'class': 'average',
                                                  'threshold': thresh,
                                                  'tp_count': np.nan,
                                                  'fp_count': np.nan,
                                                  'fn_count': np.nan,
                                                  'accuracy': cls_avg(self.cls_weight, self.cls_value['accuracy']),
                                                  'recall': cls_avg(self.cls_weight, self.cls_value['recall']),
                                                  'precision': cls_avg(self.cls_weight,
                                                                       self.cls_value['precision']),
                                                  'fp/tp': np.nan,
                                                  self.score_type: cls_avg(self.cls_weight,
                                                                           self.cls_value[self.score_type])},
                                                  ignore_index=True)
        self.count_df = self.count_df.sort_values(['class', 'threshold'])

        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
        for key in self.opt_thresh:
            self.cls_value['accuracy'].append(self.opt_thresh[key]['accuracy'])
            self.cls_value['recall'].append(self.opt_thresh[key]['recall'])
            self.cls_value['precision'].append(self.opt_thresh[key]['precision'])
            self.cls_value[self.score_type].append(self.opt_thresh[key][self.score_type])
            self.cls_weight.append(self.nodule_cls_weights[key])

        opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        opt_thresh = opt_thresh.append({'class': 'average',
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

        save_xlsx_json(self.count_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'multi-class_evaluation', 'optimal_threshold')

        # if not os.path.exists(self.result_save_dir):
        #     os.makedirs(self.result_save_dir)
        # print ("saving %s" % os.path.join(self.result_save_dir, self.xlsx_name))
        #
        # # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        # if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name)):
        #     os.remove(os.path.join(self.result_save_dir, self.xlsx_name))
        # writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name))
        # self.count_df.to_excel(writer, 'multi-class_evaluation', index=False)
        #
        # opt_thresh = opt_thresh.reset_index(drop=True)
        # opt_thresh.to_excel(writer, 'optimal_threshold')
        # writer.save()
        #
        # print ("saving %s" % os.path.join(self.result_save_dir, self.json_name))
        # # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        # if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation_nodule_threshold.json')):
        #     os.remove(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation_nodule_threshold.json'))
        # if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json')):
        #     os.remove(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json'))
        #
        # json_count_df = self.count_df.T.to_json()
        # with open(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation_nodule_threshold.json'), "w") as fp:
        #     js_count_df = json.loads(json_count_df, "utf-8")
        #     json.dump(js_count_df, fp)
        #
        # json_opt_thresh = opt_thresh.T.to_json()
        # with open(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json'), "w") as fp:
        #     js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
        #     json.dump(js_opt_thresh, fp)


    # 二分类（检出）模型统计,将所有正样本类别统计在一起
    def binary_class_evaluation(self):

        predict_df_boxes_dict, gt_df_boxes_dict, gt_df_boxes_multi_classes_dict = self.load_data()
        self.count_df = pd.DataFrame(
            columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', self.score_type])
        self.gt_cls_count_df = pd.DataFrame(
                     columns=['class', 'threshold', 'tp_count', 'fn_count', 'recall'])
        self.opt_thresh = {}
        self.summary_count_df={}
        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            predict_df_list = []
            gt_df_multi_list=[]
            self.nodule_count = 0.

            for index, key in enumerate(predict_df_boxes_dict):

                self.patient_list.append(key)
                predict_df_boxes = predict_df_boxes_dict[key]
                # gt_df_boxes = gt_df_boxes_dict[key]

                print ('processing %s' % key)

                # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                if not predict_df_boxes_dict[key].empty:
                    filtered_predict_boxes = predict_df_boxes[predict_df_boxes["prob"] >= thresh]
                    filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                else:
                    filtered_predict_boxes = pd.DataFrame(
                        {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                         'class': [], 'prob': [], 'mask': []})

                # 　将预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                print "predict_boxes:"
                print filtered_predict_boxes
                _, predict_df = get_nodule_stat(dicom_names=None,
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
                                                    focus_priority_array=self.cls_focus_priority_array,
                                                    skip_init=True)
                print "predict_nodules:"
                print predict_df

                self.nodule_count += len(predict_df)
                predict_df = predict_df.reset_index(drop=True)
                predict_df_list.append(json_df_2_df(predict_df))


                #统计ground truth 结节信息
                gt_df_boxes_multi_classes = gt_df_boxes_multi_classes_dict[key]

                if not gt_df_boxes_dict[key].empty:
                    filtered_gt_boxes_multi_classes = gt_df_boxes_multi_classes[
                        gt_df_boxes_multi_classes["prob"] >= thresh]
                    filtered_gt_boxes_multi_classes = filtered_gt_boxes_multi_classes.reset_index(drop=True)
                    print "gt_boxes_multi_classes:"
                    print filtered_gt_boxes_multi_classes
                    _, gt_df_multi_classes = get_nodule_stat(dicom_names=None,
                                                             hu_img_array=None,
                                                             return_boxes=filtered_gt_boxes_multi_classes,
                                                             img_spacing=None,
                                                             prefix=key,
                                                             classes=self.gt_cls_name,
                                                             same_box_threshold=self.same_box_threshold_gt,
                                                             score_threshold=self.score_threshold_gt,
                                                             z_threshold=self.gt_cls_z_threshold_gt,
                                                             nodule_cls_weights=self.gt_cls_weights,
                                                             if_dicom=False,
                                                             focus_priority_array=self.gt_cls_focus_priority_array,
                                                             skip_init=True)
                else:
                    gt_df_multi_classes = pd.DataFrame({'Bndbox List': [], 'Object Id': [], 'Pid': key, 'Type': [],
                               'SliceRange': [], 'prob': [], 'Diameter': [], 'CT_value': []})


                gt_df_multi_classes=gt_df_multi_classes.reset_index(drop=True)
                gt_df_multi_list.append(json_df_2_df(gt_df_multi_classes))

            summary_count_df=df_to_xlsx_file(predict_df_list,gt_df_multi_list,thresh=self.nodule_compare_thresh)

            summary_count_df = summary_count_df.sort_values(by=['PatientID'])
            summary_count_df = summary_count_df.reset_index(drop=True)

            # 统计TP FP FN  RECALL FP/TP信息
            tp_count = len(summary_count_df[summary_count_df['Result'] == 'TP'])
            fp_count = len(summary_count_df[summary_count_df['Result'] == 'FP'])
            fn_count = len(summary_count_df[summary_count_df['Result'] == 'FN'])

            recall = float(tp_count) / (tp_count + fn_count) if tp_count!=0 else 0
            fp_tp = float(fp_count) / tp_count if tp_count!=0 else np.nan
            precision=float(tp_count)/(tp_count+fp_count) if tp_count!=0 else 0

            self.count_df = self.count_df.append({'class': 'nodule',
                                                  'threshold': thresh,
                                                  'nodule_count': self.nodule_count,
                                                  'tp_count': tp_count,
                                                  'fp_count': fp_count,
                                                  'fn_count': fn_count,
                                                  'accuracy': np.nan,
                                                  'recall': recall,
                                                  'precision': precision,
                                                  'fp/tp': fp_tp,
                                                  self.score_type: (1+self.fscore_beta**2)*recall*precision/(self.fscore_beta**2*precision+recall)},
                                                 ignore_index=True)

            #统计不同结节的信息
            for gt_cls in self.gt_cls_name:
                if gt_cls=='__background__':
                    continue

                tp_count = len(summary_count_df[(summary_count_df['Result'] == 'TP') & (summary_count_df['ground_truth_class'] == gt_cls)])
                fn_count = len(summary_count_df[(summary_count_df['Result'] == 'FN') & (summary_count_df['ground_truth_class'] == gt_cls)])

                recall = float(tp_count) / (tp_count + fn_count) if tp_count!=0 else 0
                self.gt_cls_count_df = self.gt_cls_count_df.append({'class': gt_cls,
                                                                    'threshold':thresh,
                                                                    'tp_count': tp_count,
                                                                    'fn_count': fn_count,
                                                                    'recall': recall
                                                                    }, ignore_index=True)
            #预处理存储数据
            for index in summary_count_df.index:
                if index == 0:
                    patientID = summary_count_df.loc[index, 'PatientID']
                else:
                    if summary_count_df.loc[index, 'PatientID'] == patientID:
                        summary_count_df.loc[index, 'PatientID'] = np.nan
                    else:
                        patientID = summary_count_df.loc[index, 'PatientID']


            self.summary_count_df[thresh]=summary_count_df

            # find the optimal threshold
            if 'nodule' not in self.opt_thresh:

                self.opt_thresh['nodule'] = self.count_df.iloc[-1]

                self.opt_thresh['nodule']["threshold"] = thresh

            else:
                # we choose the optimal threshold corresponding to the one that gives the highest model score
                if self.count_df.iloc[-1][self.score_type] > self.opt_thresh['nodule'][self.score_type]:
                    self.opt_thresh['nodule'] = self.count_df.iloc[-1]
                    self.opt_thresh['nodule']["threshold"] = thresh


        self.count_df = self.count_df.sort_values('threshold')
        self.gt_cls_count_df = self.gt_cls_count_df.sort_values(['threshold', 'class'])

        save_xlsx_json_three_sheets(self.count_df, self.gt_cls_count_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'binary-class_evaluation', 'gt_cls_evaluation', 'optimal_threshold')

        save_xlsx_sheets(self.summary_count_df,self.result_save_dir,'result.xlsx',self.json_name)

        # if not os.path.exists(self.result_save_dir):
        #     os.makedirs(self.result_save_dir)
        # print ("saving %s" % os.path.join(self.result_save_dir, self.xlsx_name))
        #
        # #　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        #
        # if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name)):
        #     os.remove(os.path.join(self.result_save_dir, self.xlsx_name))
        # writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name))
        # self.count_df.to_excel(writer, 'binary-class_evaluation', index=False)
        # opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        # opt_thresh = opt_thresh.reset_index(drop=True)
        # opt_thresh.to_excel(writer, 'optimal_threshold')
        # writer.save()
        #
        # print ("saving %s" % os.path.join(self.result_save_dir, self.json_name))
        # # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        # if os.path.isfile(
        #         os.path.join(self.result_save_dir, self.json_name + '_binary-class_evaluation.json')):
        #     os.remove(
        #         os.path.join(self.result_save_dir, self.json_name + '_binary-class_evaluation.json'))
        # if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json')):
        #     os.remove(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json'))
        #
        # json_count_df = self.count_df.T.to_json()
        # with open(os.path.join(self.result_save_dir, self.json_name + '_binary-class_evaluation.json'),
        #           "w") as fp:
        #     js_count_df = json.loads(json_count_df, "utf-8")
        #     json.dump(js_count_df, fp)
        #
        # json_opt_thresh = opt_thresh.T.to_json()
        # with open(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json'), "w") as fp:
        #     js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
        #     json.dump(js_opt_thresh, fp)

    #先把框匹配成结节，再用阈值对结节的最高概率进行筛选
    def binary_class_evaluation_nodule_threshold(self):

        predict_df_boxes_dict, gt_df_boxes_dict, gt_df_boxes_multi_classes_dict = self.load_data()
        self.count_df = pd.DataFrame(
            columns=['class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', self.score_type])
        self.gt_cls_count_df = pd.DataFrame(
            columns=['class', 'threshold', 'tp_count', 'fn_count', 'recall'])
        self.opt_thresh = {}
        self.summary_count_df = {}

        predict_df_list = []
        gt_df_multi_list = []

        for index, key in enumerate(predict_df_boxes_dict):

            self.patient_list.append(key)
            predict_df_boxes = predict_df_boxes_dict[key]

            print ('processing %s' % key)

            # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
            if not predict_df_boxes_dict[key].empty:
                filtered_predict_boxes =predict_df_boxes.reset_index(drop=True)
            else:
                filtered_predict_boxes = pd.DataFrame(
                    {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                     'class': [], 'prob': [], 'mask': []})

            # 　将预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
            print "predict_boxes:"
            print filtered_predict_boxes
            _, predict_df = get_nodule_stat(dicom_names=None,
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
                                            focus_priority_array=self.cls_focus_priority_array,
                                            skip_init=True)
            print "predict_nodules:"
            print predict_df

            predict_df = predict_df.reset_index(drop=True)
            predict_df_list.append(json_df_2_df(predict_df))

            # 统计ground truth 结节信息
            gt_df_boxes_multi_classes = gt_df_boxes_multi_classes_dict[key]

            if not gt_df_boxes_dict[key].empty:
                filtered_gt_boxes_multi_classes = gt_df_boxes_multi_classes.reset_index(drop=True)
                print "gt_boxes_multi_classes:"
                print filtered_gt_boxes_multi_classes
                _, gt_df_multi_classes = get_nodule_stat(dicom_names=None,
                                                         hu_img_array=None,
                                                         return_boxes=filtered_gt_boxes_multi_classes,
                                                         img_spacing=None,
                                                         prefix=key,
                                                         classes=self.gt_cls_name,
                                                         same_box_threshold=self.same_box_threshold_gt,
                                                         score_threshold=self.score_threshold_gt,
                                                         z_threshold=self.gt_cls_z_threshold_gt,
                                                         nodule_cls_weights=self.gt_cls_weights,
                                                         if_dicom=False,
                                                         focus_priority_array=self.gt_cls_focus_priority_array,
                                                         skip_init=True)
            else:
                gt_df_multi_classes = pd.DataFrame({'Bndbox List': [], 'Object Id': [], 'Pid': key, 'Type': [],
                                                    'SliceRange': [], 'prob': []})

            gt_df_multi_classes = gt_df_multi_classes.reset_index(drop=True)
            gt_df_multi_list.append(json_df_2_df(gt_df_multi_classes))

        summary_count_dfs = df_to_xlsx_file(predict_df_list, gt_df_multi_list, thresh=self.nodule_compare_thresh)

        gt_count = 0
        gt_cls_count = [0 for _ in range(len(self.gt_cls_name))]

        for thresh in self.conf_thresh:
            if thresh == self.conf_thresh[0]:
                gt_count = len(summary_count_dfs[summary_count_dfs['Result'] == 'TP']) + \
                           len(summary_count_dfs[summary_count_dfs['Result'] == 'FN'])
            self.nodule_count=0
            summary_count_df=summary_count_dfs[summary_count_dfs['Prob']>=thresh]

            summary_count_df = summary_count_df.sort_values(by=['PatientID'])
            summary_count_df = summary_count_df.reset_index(drop=True)

            # 统计TP FP FN  RECALL FP/TP信息
            tp_count = len(summary_count_df[summary_count_df['Result'] == 'TP'])
            fp_count = len(summary_count_df[summary_count_df['Result'] == 'FP'])
            fn_count = gt_count - tp_count

            recall = float(tp_count) / (tp_count + fn_count) if tp_count != 0 else 0
            fp_tp = float(fp_count) / tp_count if tp_count != 0 else np.nan
            precision = float(tp_count) / (tp_count + fp_count) if tp_count != 0 else 0
            self.nodule_count=tp_count+fp_count

            self.count_df = self.count_df.append({'class': 'nodule',
                                                  'threshold': thresh,
                                                  'nodule_count': self.nodule_count,
                                                  'tp_count': tp_count,
                                                  'fp_count': fp_count,
                                                  'fn_count': fn_count,
                                                  'accuracy': np.nan,
                                                  'recall': recall,
                                                  'precision': precision,
                                                  'fp/tp': fp_tp,
                                                  self.score_type: (1 + self.fscore_beta ** 2) * recall * precision / (
                                                              self.fscore_beta ** 2 * precision + recall)},
                                                 ignore_index=True)

            # 统计不同结节的信息
            for i_cls, gt_cls in enumerate(self.gt_cls_name):
                if gt_cls == '__background__':
                    continue


                tp_count = len(summary_count_df[(summary_count_df['Result'] == 'TP') & (
                            summary_count_df['ground_truth_class'] == gt_cls)])
                fn_count = len(summary_count_df[(summary_count_df['Result'] == 'FN') & (
                            summary_count_df['ground_truth_class'] == gt_cls)])

                if thresh == self.conf_thresh[0]:
                    gt_cls_count[i_cls] = tp_count+fn_count

                recall = float(tp_count) / (tp_count + fn_count) if tp_count != 0 else 0
                self.gt_cls_count_df = self.gt_cls_count_df.append({'class': gt_cls,
                                                                    'threshold': thresh,
                                                                    'tp_count': tp_count,
                                                                    'fn_count': gt_cls_count[i_cls] - tp_count,
                                                                    'recall': recall
                                                                    }, ignore_index=True)
            # 预处理存储数据
            for index in summary_count_df.index:
                if index == 0:
                    patientID = summary_count_df.loc[index, 'PatientID']
                else:
                    if summary_count_df.loc[index, 'PatientID'] == patientID:
                        summary_count_df.loc[index, 'PatientID'] = np.nan
                    else:
                        patientID = summary_count_df.loc[index, 'PatientID']

            self.summary_count_df[thresh] = summary_count_df

            # find the optimal threshold
            if 'nodule' not in self.opt_thresh:

                self.opt_thresh['nodule'] = self.count_df.iloc[-1]

                self.opt_thresh['nodule']["threshold"] = thresh

            else:
                # we choose the optimal threshold corresponding to the one that gives the highest model score
                if self.count_df.iloc[-1][self.score_type] > self.opt_thresh['nodule'][self.score_type]:
                    self.opt_thresh['nodule'] = self.count_df.iloc[-1]
                    self.opt_thresh['nodule']["threshold"] = thresh

        self.count_df = self.count_df.sort_values('threshold')
        self.gt_cls_count_df = self.gt_cls_count_df.sort_values(['threshold', 'class'])

        save_xlsx_json_three_sheets(self.count_df, self.gt_cls_count_df, self.opt_thresh, self.result_save_dir,
                                    self.xlsx_name, self.json_name,
                                    'binary-class_evaluation', 'gt_cls_evaluation', 'optimal_threshold')

        save_xlsx_sheets(self.summary_count_df, self.result_save_dir, 'result.xlsx',self.json_name)

        # # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        # for thresh in self.conf_thresh:
        #     self.nodule_count = 0
        #     predict_df_list = []
        #     gt_df_list = []
        #     gt_df_multi_classes_list = [[] for _ in range(len(self.gt_cls_name))]
        #     for index, key in enumerate(predict_df_boxes_dict):
        #         self.patient_list.append(key)
        #         predict_df_boxes = predict_df_boxes_dict[key]
        #         gt_df_boxes = gt_df_boxes_dict[key]
        #
        #         print ('processing %s' % key)
        #
        #         # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
        #         if not predict_df_boxes_dict[key].empty:
        #             filtered_predict_boxes = predict_df_boxes.reset_index(drop=True)
        #         else:
        #             filtered_predict_boxes = pd.DataFrame(
        #                 {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
        #                  'class': [], 'prob': [], 'mask': []})
        #
        #         if not gt_df_boxes_dict[key].empty:
        #             filtered_gt_boxes = gt_df_boxes.reset_index(drop=True)
        #         else:
        #             filtered_gt_boxes = pd.DataFrame(
        #                 {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
        #                  'class': [], 'prob': [], 'mask': []})
        #
        #         # 将预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
        #         print "predict_boxes:"
        #         print filtered_predict_boxes
        #         _, predict_df = get_nodule_stat(dicom_names=None,
        #                                         hu_img_array=None,
        #                                         return_boxes=filtered_predict_boxes,
        #                                         img_spacing=None,
        #                                         prefix=key,
        #                                         classes=self.cls_name,
        #                                         same_box_threshold=self.same_box_threshold_pred,
        #                                         score_threshold=self.score_threshold_pred,
        #                                         z_threshold=self.z_threshold_pred,
        #                                         nodule_cls_weights=self.nodule_cls_weights,
        #                                         if_dicom=False,
        #                                         focus_priority_array=None,
        #                                         skip_init=True)
        #         print "predict_nodules:"
        #         print predict_df
        #
        #         print "gt_boxes:"
        #         print filtered_gt_boxes
        #         _, gt_df = get_nodule_stat(dicom_names=None,
        #                                    hu_img_array=None,
        #                                    return_boxes=filtered_gt_boxes,
        #                                    img_spacing=None,
        #                                    prefix=key,
        #                                    classes=self.cls_name,
        #                                    same_box_threshold=self.same_box_threshold_gt,
        #                                    score_threshold=self.score_threshold_gt,
        #                                    z_threshold=self.z_threshold_gt,
        #                                    nodule_cls_weights=self.nodule_cls_weights,
        #                                    if_dicom=False,
        #                                    focus_priority_array=None,
        #                                    skip_init=True)
        #         print "gt_nodules:"
        #         print gt_df
        #
        #         predict_df = predict_df[predict_df['prob'] >= thresh]
        #         self.nodule_count += len(predict_df)
        #         predict_df = predict_df.reset_index(drop=True)
        #         predict_df_list.append(json_df_2_df(predict_df))
        #
        #         gt_df = gt_df[gt_df['prob'] >= thresh]
        #         gt_df = gt_df.reset_index(drop=True)
        #         gt_df_list.append(json_df_2_df(gt_df))
        #
        #         # calculate stat for ground truth labels with original gt classes
        #         for gt_cls_num, gt_cls in enumerate(self.gt_cls_name):
        #             if gt_cls == "__background__":
        #                 continue
        #
        #             gt_df_boxes_multi_classes = gt_df_boxes_multi_classes_dict[key]
        #
        #             if not gt_df_boxes_dict[key].empty:
        #                 filtered_gt_boxes_multi_classes = gt_df_boxes_multi_classes.reset_index(drop=True)
        #             else:
        #                 # if filtered_gt_boxes_multi_classes is empty, append an empty dataframe to prevent
        #                 # invalid type comparison in gt_df_multi_classes[gt_df_multi_classes['class'] == gt_cls]
        #                 gt_df_multi_classes_list[gt_cls_num].append(pd.DataFrame({'bbox': [], 'pid': [], 'slice': [], \
        #                                                                       'class': [], 'nodule_id': []}))
        #                 continue
        #
        #             # 将标记的ground truth框(filtered_gt_boxes_multi_classes)输入get_nodule_stat进行结节匹配
        #             print "gt_boxes_multi_classes:"
        #             print filtered_gt_boxes_multi_classes
        #             _, gt_df_multi_classes = get_nodule_stat(dicom_names=None,
        #                                                      hu_img_array=None,
        #                                                      return_boxes=filtered_gt_boxes_multi_classes,
        #                                                      img_spacing=None,
        #                                                      prefix=key,
        #                                                      classes=self.gt_cls_name,
        #                                                      same_box_threshold=self.same_box_threshold_gt,
        #                                                      score_threshold=self.score_threshold_gt,
        #                                                      z_threshold=self.gt_cls_z_threshold_gt,
        #                                                      nodule_cls_weights=self.gt_cls_weights,
        #                                                      if_dicom=False,
        #                                                      focus_priority_array=self.gt_cls_focus_priority_array,
        #                                                      skip_init=True)
        #
        #             gt_df_multi_classes = json_df_2_df(gt_df_multi_classes)
        #             gt_df_multi_classes = gt_df_multi_classes[gt_df_multi_classes['class'] == gt_cls]
        #             gt_df_multi_classes = gt_df_multi_classes.reset_index(drop=True)
        #             gt_df_multi_classes['class'] = self.cls_dict[gt_cls]
        #             gt_df_multi_classes_list[gt_cls_num].append(gt_df_multi_classes)
        #
        #     # convert pandas dataframe to list of class labels
        #     cls_pred_labels, cls_gt_labels = df_to_cls_label(predict_df_list, gt_df_list, self.cls_name, thresh=self.nodule_compare_thresh)
        #
        #     # initialize ClassificationMetric class and update with ground truth/predict labels
        #     cls_metric = ClassificationMetric(cls_num=1, if_binary=True, pos_cls_fusion=True)
        #
        #     cls_metric.update(cls_gt_labels, cls_pred_labels, cls_label=1)
        #     if cls_metric.tp[0] == 0:
        #         fp_tp = np.nan
        #     else:
        #         fp_tp = cls_metric.fp[0] / cls_metric.tp[0]
        #     self.count_df = self.count_df.append({'class': 'nodule',
        #                                           'threshold': thresh,
        #                                           'nodule_count': self.nodule_count,
        #                                           'tp_count': cls_metric.tp[0],
        #                                           'fp_count': cls_metric.fp[0],
        #                                           'fn_count': cls_metric.fn[0],
        #                                           'accuracy': cls_metric.get_acc(cls_label=1),
        #                                           'recall': cls_metric.get_rec(cls_label=1),
        #                                           'precision': cls_metric.get_prec(cls_label=1),
        #                                           'fp/tp': fp_tp,
        #                                           self.score_type: cls_metric.get_fscore(cls_label=1, beta=self.fscore_beta)},
        #                                          ignore_index=True)
        #
        #     # calculate stat for ground truth labels with original gt classes
        #     for gt_cls_num, gt_cls in enumerate(self.gt_cls_name):
        #         if gt_cls == "__background__":
        #             continue
        #
        #         # convert pandas dataframe to list of gt class labels
        #         cls_pred_labels, cls_gt_multi_classes_labels = df_to_cls_label(predict_df_list,
        #                                                                        gt_df_multi_classes_list[gt_cls_num],
        #                                                                        self.cls_name,
        #                                                                        thresh=self.nodule_compare_thresh)
        #         # initialize ClassificationMetric class and update with ground truth/predict labels
        #         cls_metric = ClassificationMetric(cls_num=1, if_binary=True, pos_cls_fusion=True)
        #
        #         cls_metric.update(cls_gt_multi_classes_labels, cls_pred_labels, cls_label=1)
        #
        #         self.gt_cls_count_df = self.gt_cls_count_df.append({'class': gt_cls,
        #                                                             'threshold': thresh,
        #                                                             'tp_count': cls_metric.tp[0],
        #                                                             'fn_count': cls_metric.fn[0],
        #                                                             'recall': cls_metric.get_rec(cls_label=1)},
        #                                                            ignore_index=True)
        #
        #     # find the optimal threshold
        #     if 'nodule' not in self.opt_thresh:
        #
        #         self.opt_thresh['nodule'] = self.count_df.iloc[-1]
        #
        #         self.opt_thresh['nodule']["threshold"] = thresh
        #
        #     else:
        #         # we choose the optimal threshold corresponding to the one that gives the highest model score
        #         if self.count_df.iloc[-1][self.score_type] > self.opt_thresh['nodule'][self.score_type]:
        #             self.opt_thresh['nodule'] = self.count_df.iloc[-1]
        #             self.opt_thresh['nodule']["threshold"] = thresh
        #
        # self.count_df = self.count_df.sort_values('threshold')
        # self.gt_cls_count_df = self.gt_cls_count_df.sort_values(['threshold', 'class'])
        #
        # save_xlsx_json_three_sheets(self.count_df, self.gt_cls_count_df, self.opt_thresh, self.result_save_dir,
        #                             self.xlsx_name, self.json_name,
        #                             'binary-class_evaluation', 'gt_cls_evaluation', 'optimal_threshold')
        # if not os.path.exists(self.result_save_dir):
        #     os.makedirs(self.result_save_dir)
        # print ("saving %s" % os.path.join(self.result_save_dir, self.xlsx_name))
        #
        # # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        #
        # if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name)):
        #     os.remove(os.path.join(self.result_save_dir, self.xlsx_name))
        # writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name))
        # self.count_df.to_excel(writer, 'binary-class_evaluation', index=False)
        # opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        # opt_thresh = opt_thresh.reset_index(drop=True)
        # opt_thresh.to_excel(writer, 'optimal threshold')
        # writer.save()
        #
        # print ("saving %s" % os.path.join(self.result_save_dir, self.json_name))
        # # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        # if os.path.isfile(
        #         os.path.join(self.result_save_dir,
        #                      self.json_name + '_binary-class_evaluation_nodule_threshold.json')):
        #     os.remove(
        #         os.path.join(self.result_save_dir,
        #                      self.json_name + '_binary-class_evaluation_nodule_threshold.json'))
        # if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json')):
        #     os.remove(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json'))
        #
        # json_count_df = self.count_df.T.to_json()
        # with open(os.path.join(self.result_save_dir,
        #                        self.json_name + '_binary-class_evaluation_nodule_threshold.json'),
        #           "w") as fp:
        #     js_count_df = json.loads(json_count_df, "utf-8")
        #     json.dump(js_count_df, fp)
        #
        # json_opt_thresh = opt_thresh.T.to_json()
        # with open(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json'), "w") as fp:
        #     js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
        #     json.dump(js_opt_thresh, fp)

    # 读入预测结果数据

    def load_data(self):
        """
        读入模型输出的.json和ground truth的.xml标记
        :return: 模型预测结果、ground truth标记按病人号排列的pandas.DataFrame
        e.g. | mask | instanceNumber | class | prob | sliceId | xmax | xmin | ymax | ymin |
             |  []  |       106      | solid nodule | 0.9  | 105.0   | 207.0| 182.0| 230.0| 205.0|
        """
        predict_df_boxes_dict = {}
        ground_truth_boxes_dict = {}
        ground_truth_boxes_multi_classes_dict = {}
        # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
        for PatientID in os.listdir(self.data_dir):
            if self.data_type == 'json':
                predict_json_path = os.path.join(self.data_dir, PatientID, PatientID + '_predict.json')
                try:
                    predict_df_boxes = pd.read_json(predict_json_path).T
                    predict_df_boxes = predict_df_boxes.rename(index=str, columns={'nodule_class': 'class'})
                except:
                    print ("broken directory structure, maybe no prediction json file found: %s" % predict_json_path)
                    raise NameError
                try:
                    check_insnum_sliceid(predict_df_boxes)
                except:
                    logging.exception('%s has inconsistent instanceNumber and sliceId' %PatientID)
            elif self.data_type == 'npy':
                predict_npy_path = os.path.join(self.data_dir, PatientID, PatientID + '_predict.npy')
                try:
                    predict_boxes = np.load(predict_npy_path)
                except:
                    print ("broken directory structure, maybe no prediction npy file found: %s" % predict_npy_path)
                    raise NameError
                predict_df_boxes = init_df_boxes(return_boxes=predict_boxes, classes=self.cls_name)
                predict_df_boxes = predict_df_boxes.sort_values(by=['prob'])
                predict_df_boxes = predict_df_boxes.reset_index(drop=True)
            else:
                # 　尚未考虑其他数据存储格式，有需要的话日后添加
                raise NotImplemented

            ground_truth_path = os.path.join(self.anno_dir, PatientID)
            try:
                # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                ground_truth_boxes = xml_to_boxeslist(config=self.config, xml_dir=ground_truth_path, box_length=2000)
            except:
                print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                ground_truth_boxes = [[[[]]]]

            try:
                # 对于ground truth boxes,我们直接读取其xml标签,并保留原始的结节细分类别。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                ground_truth_boxes_multi_classes = xml_to_boxeslist_multi_classes(config=self.config, xml_dir=ground_truth_path, box_length=2000)
            except:
                print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                ground_truth_boxes_multi_classes = [[[[]]]]

            ground_truth_boxes = init_df_boxes(return_boxes=ground_truth_boxes, classes=self.cls_name)
            ground_truth_boxes = ground_truth_boxes.sort_values(by=['prob'])
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            ground_truth_boxes_multi_classes = init_df_boxes(return_boxes=ground_truth_boxes_multi_classes, classes=self.gt_cls_name)
            ground_truth_boxes_multi_classes = ground_truth_boxes_multi_classes.sort_values(by=['prob'])
            ground_truth_boxes_multi_classes = ground_truth_boxes_multi_classes.reset_index(drop=True)

            predict_df_boxes_dict[PatientID] = predict_df_boxes
            ground_truth_boxes_dict[PatientID] = ground_truth_boxes
            ground_truth_boxes_multi_classes_dict[PatientID] = ground_truth_boxes_multi_classes
        return predict_df_boxes_dict, ground_truth_boxes_dict, ground_truth_boxes_multi_classes_dict

    # 由predict出的框和ground truth anno生成_nodules.json和_gt.json
    def generate_df_nodules_to_json(self):
        """
        读入_predict.json及gt annotation文件，经过get_nodule_stat转换为json文件并存储到指定目录
        """

        predict_df_boxes_dict, ground_truth_boxes_dict, _ = self.load_data()

        # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
        for PatientID in os.listdir(self.data_dir):
            predict_df_boxes = predict_df_boxes_dict[PatientID]
            ground_truth_boxes = ground_truth_boxes_dict[PatientID]

            if predict_df_boxes.empty:
                predict_df_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                   'class': [], 'prob': [], 'mask': []})
            else:
                predict_df_boxes = predict_df_boxes.reset_index(drop=True)

            if ground_truth_boxes.empty:
                ground_truth_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                   'class': [], 'prob': [], 'mask': []})
            else:
                ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            print "prediction:"
            _, predict_df = get_nodule_stat(dicom_names=None,
                                            hu_img_array=None,
                                            return_boxes=predict_df_boxes,
                                            img_spacing=None,
                                            prefix=PatientID,
                                            classes=self.cls_name,
                                            same_box_threshold=self.same_box_threshold_pred,
                                            score_threshold=self.score_threshold_pred,
                                            z_threshold=self.z_threshold_pred,
                                            nodule_cls_weights=self.nodule_cls_weights,
                                            if_dicom=False,
                                            focus_priority_array=None,
                                            skip_init=True)
            print "ground truth"
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)
            _, gt_df = get_nodule_stat(dicom_names=None,
                                       hu_img_array=None,
                                       return_boxes=ground_truth_boxes,
                                       img_spacing=None,
                                       prefix=PatientID,
                                       classes=self.cls_name,
                                       same_box_threshold=self.same_box_threshold_gt,
                                       score_threshold=self.score_threshold_gt,
                                       z_threshold=self.z_threshold_gt,
                                       nodule_cls_weights=self.nodule_cls_weights,
                                       if_dicom=False,
                                       focus_priority_array=None,
                                       skip_init=True)
            str_nodules = predict_df.T.to_json()
            str_gt = gt_df.T.to_json()
            if not os.path.exists(self.result_save_dir):
                os.mkdir(self.result_save_dir)
            json_patient_dir = os.path.join(self.result_save_dir, PatientID)
            print ('processing patient: %s' %PatientID)
            print json_patient_dir
            if not os.path.exists(json_patient_dir):
                os.mkdir(json_patient_dir)
            with open(os.path.join(json_patient_dir, PatientID + '_nodule.json'), "w") as fp:
                js_nodules = json.loads(str_nodules, "utf-8")
                json.dump(js_nodules, fp)
            with open(os.path.join(json_patient_dir, PatientID + '_gt.json'), "w") as fp:
                js_gt = json.loads(str_gt, "utf-8")
                json.dump(js_gt, fp)

    # 筛选一定层厚以上的最终输出的结节（降假阳实验）
    def nodule_thickness_filter(self):
        assert type(self.thickness_thresh) == int, "input thickness_thresh should be an integer, not %s" %self.thickness_thresh
        for PatientID in os.listdir(self.data_dir):
            if self.data_type == 'json':
                predict_json_path = os.path.join(self.result_save_dir, PatientID, PatientID + '_nodule.json')
                try:
                    predict_df_boxes = pd.read_json(predict_json_path).T
                except:
                    raise ("broken directory structure, maybe no prediction json file found: %s" % predict_json_path)
            drop_list = []
            for i, row in predict_df_boxes.iterrows():
                if len(row['SliceRange']) <= self.thickness_thresh:
                    drop_list.append(i)
            predict_df_boxes = predict_df_boxes.drop(drop_list)

            str_nodules = predict_df_boxes.T.to_json()
            if not os.path.exists(self.result_save_dir):
                os.mkdir(self.result_save_dir)
            json_patient_dir = os.path.join(self.result_save_dir, PatientID)
            print ('processing patient: %s' %PatientID)
            print json_patient_dir
            if not os.path.exists(json_patient_dir):
                os.mkdir(json_patient_dir)
            with open(os.path.join(json_patient_dir, PatientID + '_nodule%s.json' %(self.thickness_thresh)), "w") as fp:
                js_nodules = json.loads(str_nodules, "utf-8")
                json.dump(js_nodules, fp)


class FindNodulesEvaluator(object):
    def __init__(self, cls_label_xls_path, gt_anno_dir, conf_thres = 1., result_save_dir = os.path.join(os.getcwd(), 'FindNodulesEvaluation_result'),
                 xlsx_name = 'FindNodulesEvaluation', json_name = 'FindNodulesEvaluation', algorithm = 'find_nodules_new', if_nodule_cls = True,
                 same_box_threshold_gt = np.array([0., 0.]), score_threshold_gt = 0.4):
        self.config = LungConfig(cls_label_xls_path=cls_label_xls_path)
        assert os.path.isdir(gt_anno_dir), 'must initialize it with a valid directory of annotation data'
        self.gt_anno_dir = gt_anno_dir
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.patient_list = []
        self.cls_name = self.config.CLASSES
        self.conf_thresh = conf_thres
        self.algorithm = algorithm
        self.result_df = pd.DataFrame(
            columns=['patientid', 'algorithm', 'gt_nodule_count', 'nodule_count', 'threshold', 'adjusted_rand_index', 'adjusted_mutual_info_score',
                     'normalized_mutual_info_score', 'homogeneity_completeness_v_measure', 'fowlkes_mallows_score', 'silhouette_score'])
        self.nodule_count_df = pd.DataFrame(
            columns=['patientid', 'nodule_count'])
        # whether the ground truth labels include nodule class or not
        self.if_nodule_cls = if_nodule_cls
        self.nodule_cls_weights = self.config.CLASS_WEIGHTS
        self.same_box_threshold_gt = same_box_threshold_gt
        self.score_threshold_gt = score_threshold_gt
        self.z_threshold_gt = self.config.CLASS_Z_THRESHOLD_GT


    def evaluation_with_nodule_num(self):
        # gt_boxes_list: list of patient, each of which contains list of all boxes of a patient
        if self.if_nodule_cls:
            gt_df_boxes_dict, gt_boxes_list = self.load_data_xml_with_nodule_num()
        else:
            gt_df_boxes_dict, gt_boxes_list = self.load_data_xml_with_nodule_num_without_nodule_cls()
        gt_labels = []
        post_find_nodules_labels = []

        for index, key in enumerate(gt_df_boxes_dict):
            self.patient_list.append(key)
            gt_df_boxes = gt_df_boxes_dict[key]
            gt_label = []
            post_find_nodules_label = []

            print ('processing %s' % key)

            # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
            if not gt_df_boxes_dict[key].empty:
                filtered_gt_boxes = gt_df_boxes[gt_df_boxes["prob"] >= self.conf_thresh]
                filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
            else:
                filtered_gt_boxes = pd.DataFrame(
                    {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                     'class': [], 'prob': [], 'mask': []})

            # 将标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
            print "generating ground truth nodules:"
            gt_boxes, gt_df = get_nodule_stat(dicom_names=None,
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
            #之所以可以直接用index调用gt_boxes_list是因为ground_truth_boxes_dict初始化为collections.OrderedDict(),所以不会出现病人错位的问题
            for gt_box_list in gt_boxes_list[index]:
                #对于每个gt_box_list(格式为[x1, y1, x2, y2, 1, mapped_name, nodule_num, slice_id]),找到跟它在同一个层面的所有框
                box_lst = gt_boxes[gt_boxes['sliceId'] == gt_box_list[-1]]
                for i in range(len(box_lst.index)):
                    box1 = [box_lst.iloc[i]['xmin'], box_lst.iloc[i]['ymin'], box_lst.iloc[i]['xmax'], box_lst.iloc[i]['ymax']]
                    box2 = gt_box_list[0:4]
                    if box1 == box2:
                        gt_label.append(gt_box_list[6])
                        post_find_nodules_label.append(box_lst.iloc[i]['object'])
            gt_labels.append(gt_label)
            # # delete nodules that are labeled as -1 (not matched)
            post_find_nodules_labels.append(post_find_nodules_label)
        print 'ground truth labels:'
        print gt_labels
        print 'labels after find_nodules:'
        print post_find_nodules_labels

        clus_metric = ClusteringMetric()
        clus_metric.update(gt_labels, post_find_nodules_labels)
        for index, key in enumerate(gt_df_boxes_dict):
            self.result_df = self.result_df.append({'patientid': key,
                               'algorithm': self.algorithm,
                               'gt_nodule_count': len(set(gt_labels[index])),
                               'nodule_count': len(set(post_find_nodules_labels[index])),
                               'threshold': self.conf_thresh,
                               'adjusted_rand_index': clus_metric.get_adjusted_rand_index()[index],
                               'adjusted_mutual_info_score': clus_metric.get_adjusted_mutual_info_score()[index],
                               'normalized_mutual_info_score': clus_metric.get_normalized_mutual_info_score()[index],
                               'homogeneity_completeness_v_measure': clus_metric.get_homogeneity_completeness_v_measure()[index],
                               'fowlkes_mallows_score': clus_metric.get_fowlkes_mallows_score()[index],
                               'silhouette_score': clus_metric.get_silhouette_score()[index]}, ignore_index=True)

        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        print ("saving %s" % os.path.join(self.result_save_dir, self.xlsx_name + '_eval_with_nodule_num.xlsx'))

        # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name + '_eval_with_nodule_num.xlsx')):
            os.remove(os.path.join(self.result_save_dir, self.xlsx_name + '_eval_with_nodule_num.xlsx'))
        writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name + '_eval_with_nodule_num.xlsx'))
        self.result_df.to_excel(writer, index=False)
        writer.save()

        print ("saving %s" % os.path.join(self.result_save_dir, self.json_name + '_eval_with_nodule_num.json'))
        # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(
                os.path.join(self.result_save_dir,
                             self.json_name + '_eval_with_nodule_num.json')):
            os.remove(
                os.path.join(self.result_save_dir,
                             self.json_name + '_eval_with_nodule_num.json'))

        json_count_df = self.result_df.T.to_json()
        with open(os.path.join(self.result_save_dir,
                               self.json_name + '_eval_with_nodule_num.json'),
                  "w") as fp:
            js_count_df = json.loads(json_count_df, "utf-8")
            json.dump(js_count_df, fp)

    # testing cfda_modified_anno_box_size in comparison with ORIGINDATA, ground truth labels only
    def evaluation_without_nodule_cls(self):
        gt_df_boxes_dict = self.load_data_xml_without_nodule_cls()
        tot_nodule_count = 0
        for index, key in enumerate(gt_df_boxes_dict):
            self.patient_list.append(key)
            gt_df_boxes = gt_df_boxes_dict[key]

            print ('processing %s' % key)
            #print gt_df_boxes_dict
            # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
            if not gt_df_boxes_dict[key].empty:
                # print gt_df_boxes
                filtered_gt_boxes = gt_df_boxes[gt_df_boxes["prob"] >= self.conf_thresh]
                filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
            else:
                filtered_gt_boxes = pd.DataFrame(
                    {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                     'class': [], 'prob': [], 'mask': []})

            # 将标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
            print "generating ground truth nodules:"
            _, gt_df = get_nodule_stat(dicom_names=None,
                                              hu_img_array=None,
                                              return_boxes=filtered_gt_boxes,
                                              img_spacing=None,
                                              prefix=key,
                                              classes=self.cls_name,
                                              same_box_threshold=self.same_box_threshold_gt,
                                              score_threshold=self.score_threshold_gt,
                                              z_threshold=self.z_threshold_gt,
                                              nodule_cls_weights={'0-3nodule': 1.},
                                              if_dicom=False,
                                              focus_priority_array=None,
                                              skip_init=True)
            #print gt_df
            tot_nodule_count += len(gt_df.index)
            print ('nodule_count = %s' %(len(gt_df.index)))
            self.nodule_count_df = self.nodule_count_df.append({'patientid': key,
                                                                'nodule_count': len(gt_df.index)}, ignore_index=True)
        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        print ("saving %s" % os.path.join(self.result_save_dir, self.xlsx_name + 'eval_without_nodule_cls.xlsx'))

        # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name + 'eval_without_nodule_cls.xlsx')):
            os.remove(os.path.join(self.result_save_dir, self.xlsx_name + 'eval_without_nodule_cls.xlsx'))

        writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name + 'eval_without_nodule_cls.xlsx'))
        self.nodule_count_df.to_excel(writer, index=False)
        writer.save()

        print ("saving %s" % os.path.join(self.result_save_dir, self.json_name))
        # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(
                os.path.join(self.result_save_dir,
                             self.json_name + '_eval_without_nodule_cls.json')):
            os.remove(
                os.path.join(self.result_save_dir,
                             self.json_name + '_eval_without_nodule_cls.json'))

        json_count_df = self.result_df.T.to_json()
        with open(os.path.join(self.result_save_dir,
                               self.json_name + '_eval_without_nodule_cls.json'),
                  "w") as fp:
            js_count_df = json.loads(json_count_df, "utf-8")
            json.dump(js_count_df, fp)

    def load_data_xml_with_nodule_num(self):
        ground_truth_boxes_dict = OrderedDict()
        ground_truth_boxes_list = []
        for PatientID in os.listdir(self.gt_anno_dir):
            ground_truth_path = os.path.join(self.gt_anno_dir, PatientID)
            try:
                # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                ground_truth_boxes, ground_truth_boxes_all_slice= xml_to_boxeslist_with_nodule_num(config=self.config, xml_dir=ground_truth_path, box_length=2000)
            except:
                print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                ground_truth_boxes = [[[[]]]]

            ground_truth_boxes_list.append(ground_truth_boxes_all_slice)
            ground_truth_boxes = init_df_boxes(return_boxes=ground_truth_boxes, classes=self.cls_name)
            ground_truth_boxes = ground_truth_boxes.sort_values(by=['prob'])
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            ground_truth_boxes_dict[PatientID] = ground_truth_boxes


        return ground_truth_boxes_dict, ground_truth_boxes_list

    def load_data_xml_without_nodule_cls(self):
        ground_truth_boxes_dict = OrderedDict()
        for PatientID in os.listdir(self.gt_anno_dir):
            ground_truth_path = os.path.join(self.gt_anno_dir, PatientID)
            try:
            # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                ground_truth_boxes = xml_to_boxeslist_without_nodule_cls(config=self.config, xml_dir=ground_truth_path, box_length=2000)
            except:
                print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                ground_truth_boxes = [[[[]]]]

            ground_truth_boxes = init_df_boxes(return_boxes=ground_truth_boxes, classes=self.cls_name)
            ground_truth_boxes = ground_truth_boxes.sort_values(by=['prob'])
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            ground_truth_boxes_dict[PatientID] = ground_truth_boxes

        return ground_truth_boxes_dict

    def load_data_xml_with_nodule_num_without_nodule_cls(self):
        ground_truth_boxes_dict = OrderedDict()
        ground_truth_boxes_list = []
        for PatientID in os.listdir(self.gt_anno_dir):
            ground_truth_path = os.path.join(self.gt_anno_dir, PatientID)
            try:
            # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于2000个层，故我们在这里选择2000
                ground_truth_boxes, ground_truth_boxes_all_slice= xml_to_boxeslist_with_nodule_num_without_nodule_cls(config=self.config, xml_dir=ground_truth_path, box_length=2000)
            except:
                print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                ground_truth_boxes = [[[[]]]]
            ground_truth_boxes_list.append(ground_truth_boxes_all_slice)
            ground_truth_boxes = init_df_boxes(return_boxes=ground_truth_boxes, classes=self.cls_name)
            ground_truth_boxes = ground_truth_boxes.sort_values(by=['prob'])
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            ground_truth_boxes_dict[PatientID] = ground_truth_boxes

        return ground_truth_boxes_dict, ground_truth_boxes_list

def predict_json_to_xml(data_dir, save_dir):
    """

    :return:按病人编号、层面数生成的xml
    e.g.
    ```
    save_dir/
    ├── patient_id_1/
    |   ├─────────── patient_id_1_xxx.xml
    |   ├─────────── patient_id_1_xxx.xml
    ├── patient_id_2/
    ├── patient_id_3/
    ```
    """
    # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for PatientID in os.listdir(data_dir):
        predict_json_path = os.path.join(data_dir, PatientID, PatientID + '_predict.json')
        try:
            predict_df_boxes = pd.read_json(predict_json_path).T
        except:
            print ("broken directory structure, maybe no prediction json file found: %s" % predict_json_path)
            raise NameError

        if predict_df_boxes.empty:
            if not os.path.exists(os.path.join(save_dir, PatientID)):
                os.mkdir(os.path.join(save_dir, PatientID))
            else:
                shutil.rmtree(os.path.join(save_dir, PatientID))
                os.mkdir(os.path.join(save_dir, PatientID))
            continue
        print PatientID
        predict_df_boxes = predict_df_boxes.sort_values(by=['instanceNumber'])
        for instanceNumber in range(1, predict_df_boxes['instanceNumber'].max()+1):
            predict_df = predict_df_boxes[predict_df_boxes['instanceNumber'] == instanceNumber]
            if not predict_df.empty:
                if not os.path.exists(os.path.join(save_dir, PatientID)):
                    os.mkdir(os.path.join(save_dir, PatientID))
                generate_xml(os.path.join(save_dir, PatientID), PatientID + '_' + slice_num_to_three_digit_str(instanceNumber) + '.xml', predict_df)


def json_df_2_df(df):
    ret_df = pd.DataFrame({'bbox': [], 'pid': [], 'slice': [], 'class': [], 'nodule_id': [], 'diameter': [], 'ct_value': []})
    for index, row in df.iterrows():
        try:
            #最新包含直径和ct值的统计结果
            df_add_row = {'bbox': [bbox for bbox in row['Bndbox List']],
                          'pid': row['Pid'],
                          'slice': row['SliceRange'],
                          'class': row['Type'],
                          'nodule_id': row['Object Id'],
                          'prob': row['Prob'],
                          'diameter': row['Diameter'],
                          'ct_value': row['CT_value']}
        except:
            #如果没有直径和ct值信息，则统计其余指标
            df_add_row = {'bbox': [bbox for bbox in row['Bndbox List']],
                          'pid': row['Pid'],
                          'slice': row['SliceRange'],
                          'class': row['Type'],
                          'nodule_id': row['Object Id'],
                          'prob': row['Prob']}
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

def df_2_box_list(df):
    raise NotImplemented


def check_insnum_sliceid(df):
    '''
    Check instance number and slice id consistency
    :param df:
    :return:
    '''
    for _, row in df.iterrows():
        assert row['instanceNumber'] == row['sliceId'] + 1, 'instanceNumber != sliceId + 1'

def parse_args():
    parser = argparse.ArgumentParser(description='Infervision auto test')
    parser.add_argument('--data_dir',
                        help='predict result stored dir, .npy by default',
                        default='./json_for_auto_test',
                        type=str)
    parser.add_argument('--data_type',
                        help='type of data that store prediction result',
                        default='json',
                        type=str
                        )
    parser.add_argument('--result_save_dir',
                        help='dir for saving xlsx',
                        default='./LungNoduleEvaluation_result',
                        type=str)
    parser.add_argument('--image_dir',
                        help='directory of ct we need to predict',
                        default='./dcm',
                        type=str)
    parser.add_argument('--image_save_dir',
                        help='dir for saving FP FN TP pictures',
                        default='./auto_test/NMS',
                        type=str)
    parser.add_argument('--dicom',
                        help='process dicom scan', action='store_true')
    parser.add_argument('--norm',
                        help='if normalize image pixel values to (0, 255)', action='store_true')
    parser.add_argument('--windowshift',
                        help='if apply intensity window to images', action='store_true')
    parser.add_argument('--save_img',
                        help='if store FP FN TP pictures', action='store_true')
    parser.add_argument('--xlsx_name',
                        help='name of generated .xlsx',
                        default='LungNoduleEvaluation.xlsx',
                        type=str)
    parser.add_argument('--json_name',
                        help='name of generated json file, no postfix',
                        default='LungNoduleEvaluation',
                        type=str)
    parser.add_argument('--gt_anno_dir',
                        help='ground truth anno stored dir',
                        default='./anno',
                        type=str)
    parser.add_argument('--multi_class',
                        help='multi-class evaluation', action='store_true')
    parser.add_argument('--nodule_threshold',
                        help='filter nodule instead of boxes with threshold', action='store_true')
    parser.add_argument('--nodule_json',
                        help='whether to generate _nodule.json which contains matched nodules information', action='store_true')
    parser.add_argument('--score_type',
                        help='type of model score',
                        default='F_score',
                        type=str)
    parser.add_argument('--clustering_test',
                        help='evaluate in terms of clustering metric', action='store_true')
    parser.add_argument('--nodule_cls',
                        help='evaluate with specified nodule class', action='store_true')
    parser.add_argument('--thickness_thresh',
                        help='threshold for filtering nodules with thickness greater or equal to certain integer',
                        default= 0,
                        type=int)
    parser.add_argument('--multi_model',
                        help='multi-model evaluation', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    from private_config import config
    import time
    tic=time.time()
    args = parse_args()
    args.nodule_threshold=True
    args.data_dir='/media/tx-deepocean/de3dcdc1-a7ea-4f87-9995-dddf00ac10ff/CT/ssd_liu_auto_test.git/JJYY/ssd-0020-json_for_predict'
    args.gt_anno_dir='/media/tx-deepocean/de3dcdc1-a7ea-4f87-9995-dddf00ac10ff/CT/test_data/20180831jiujiangyiyuan/2018_08_31_JiuJiangXueYuanFuShuYiYuan/anno'
    for _  in range(1):
        model_eval =LungNoduleEvaluatorOffline(cls_label_xls_path=config.CLASSES_LABELS_XLS_FILE_NAME,
                                                          data_dir=args.data_dir,
                                                          data_type=args.data_type,
                                                          anno_dir=args.gt_anno_dir,
                                                          score_type=args.score_type,
                                                          result_save_dir=args.result_save_dir,
                                                          xlsx_name=args.xlsx_name,
                                                          json_name=args.json_name,
                                                          if_nodule_threshold=args.nodule_threshold,
                                                          if_nodule_json=args.nodule_json,
                                                          thickness_thresh=args.thickness_thresh,
                                                          conf_thresh=config.TEST.CONF_THRESHOLD,
                                                          fscore_beta=config.FSCORE_BETA)

        if model_eval.if_nodule_json:
            model_eval.generate_df_nodules_to_json()
            if model_eval.thickness_thresh > 0:
                model_eval.nodule_thickness_filter()
            exit()

        if args.multi_class:
            if not model_eval.if_nodule_threshold:
                model_eval.multi_class_evaluation()
            else:
                model_eval.multi_class_evaluation_nodule_threshold()
            print model_eval.opt_thresh


        else:
            if not model_eval.if_nodule_threshold:
                model_eval.binary_class_evaluation()
            else:
                model_eval.binary_class_evaluation_nodule_threshold()

    toc=time.time()

    print('time cost %f'%(toc-tic))



