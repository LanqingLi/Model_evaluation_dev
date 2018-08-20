# -- coding: utf-8 --
import json
import pandas as pd
import numpy as np
import os
import argparse
import shutil
from collections import OrderedDict
from model_eval.common.custom_metric import ClassificationMetric, ClusteringMetric, cls_avg

from xml_tools import xml_to_boxeslist, xml_to_boxeslist_with_nodule_num, xml_to_boxeslist_without_nodule_cls, \
    xml_to_boxeslist_with_nodule_num_without_nodule_cls, generate_xml
from config import config
from post_process import df_to_cls_label
from get_df_nodules import get_nodule_stat, init_df_boxes

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
    def __init__(self, data_dir, data_type, anno_dir, score_type = 'fscore',  result_save_dir = os.path.join(os.getcwd(), 'LungNoduleEvaluation_result'),
                 xlsx_name = 'LungNoduleEvaluation.xlsx', json_name = 'LungNoduleEvaluation', if_nodule_json = False,
                 conf_thresh = config.TEST.CONF_THRESHOLD, nodule_cls_weights = config.CLASS_WEIGHTS, fscore_beta = config.FSCORE_BETA,
                 same_box_threshold_pred = config.FIND_NODULES.SAME_BOX_THRESHOLD_PRED, same_box_threshold_gt = config.FIND_NODULES.SAME_BOX_THRESHOLD_GT,
                 score_threshold_pred = config.FIND_NODULES.SCORE_THRESHOLD_PRED, score_threshold_gt = config.FIND_NODULES.SCORE_THRESHOLD_GT,
                 z_threshold_pred = config.CLASS_Z_THRESHOLD_PRED, z_threshold_gt = config.CLASS_Z_THRESHOLD_GT,
                 nodule_compare_thresh = config.TEST.IOU_THRESHOLD, thickness_thresh = config.THICKNESS_THRESHOLD, if_nodule_threshold = False):
        assert os.path.isdir(data_dir), 'must initialize it with a valid directory of bbox data'
        self.data_dir = data_dir
        self.data_type = data_type
        self.anno_dir = anno_dir
        # config.CLASSES 包含background class,是结节的粗分类(RCNN分类)
        self.cls_name = config.CLASSES
        self.cls_dict = config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.count_df = pd.DataFrame(
                     columns=['nodule_class', 'threshold', 'nodule_count', 'tp_count', 'fp_count', 'fn_count',
                              'accuracy', 'recall', 'precision',
                              'fp/tp', self.score_type])

        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.if_nodule_json = if_nodule_json
        # customized confidence threshold for plotting ROC curve
        self.conf_thresh = conf_thresh
        self.nodule_cls_weights = nodule_cls_weights
        self.fscore_beta = fscore_beta
        self.patient_list = []
        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}
        # objmatch.find_nodules算法的相关超参数，详见config文件
        self.same_box_threshold_pred = same_box_threshold_pred
        self.same_box_threshold_gt = same_box_threshold_gt
        self.score_threshold_pred = score_threshold_pred
        self.score_threshold_gt = score_threshold_gt
        self.z_threshold_pred = z_threshold_pred
        self.z_threshold_gt = z_threshold_gt
        self.if_nodule_threshold = if_nodule_threshold

        self.thickness_thresh = thickness_thresh
        self.nodule_compare_thresh = nodule_compare_thresh

        # keep track of the nodule count in the output of get_df_nodules, including false positives, initialized to be 0
        self.nodule_count = 0.

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
                        filtered_predict_boxes = predict_df_boxes[predict_df_boxes["nodule_class"] == cls]
                        print filtered_predict_boxes
                        filtered_predict_boxes = filtered_predict_boxes[filtered_predict_boxes["prob"] >= thresh]
                        filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                    else:
                        filtered_predict_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                 'nodule_class': [], 'prob': [], 'Mask': []})

                    if not ground_truth_boxes_dict[key].empty:
                        filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["nodule_class"] == cls]
                        print filtered_gt_boxes
                        filtered_gt_boxes = filtered_gt_boxes[filtered_gt_boxes["prob"] >= thresh]
                        filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                    else:
                        filtered_gt_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                 'nodule_class': [], 'prob': [], 'Mask': []})

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
                                             focus_priority_array=None,
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
                                                     focus_priority_array=None,
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
                cls_metric = ClassificationMetric(cls_num=i_cls)

                cls_metric.update(cls_gt_labels, cls_pred_labels)

                if cls_metric.tp == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric.fp / cls_metric.tp

                self.count_df = self.count_df.append({'nodule_class': cls,
                                                      'threshold': thresh,
                                                      'nodule_count': self.nodule_count,
                                                      'tp_count': cls_metric.tp,
                                                      'fp_count': cls_metric.fp,
                                                      'fn_count': cls_metric.fn,
                                                      'accuracy': cls_metric.get_acc(),
                                                      'recall': cls_metric.get_rec(),
                                                      'precision': cls_metric.get_prec(),
                                                      'fp/tp': fp_tp,
                                                      self.score_type: cls_metric.get_fscore(beta=self.fscore_beta)},
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

                self.cls_value['accuracy'].append(cls_metric.get_acc())
                self.cls_value['recall'].append(cls_metric.get_rec())
                self.cls_value['precision'].append(cls_metric.get_prec())
                self.cls_value[self.score_type].append(cls_metric.get_fscore(beta=self.fscore_beta))

            #增加多类别加权平均的结果
            self.count_df = self.count_df.append({'nodule_class': 'average',
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

        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        print ("saving %s" %os.path.join(self.result_save_dir, self.xlsx_name))

        # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name)):
            os.remove(os.path.join(self.result_save_dir, self.xlsx_name))
        writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name))
        self.count_df.to_excel(writer, 'multi-class_evaluation', index=False)

        opt_thresh = opt_thresh.reset_index(drop=True)
        opt_thresh.to_excel(writer, 'optimal_threshold')
        writer.save()

        print ("saving %s" %os.path.join(self.result_save_dir, self.json_name))
        # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation.json')):
            os.remove(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation.json'))
        if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json')):
            os.remove(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json'))

        json_count_df = self.count_df.T.to_json()
        with open(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation.json'), "w") as fp:
            js_count_df = json.loads(json_count_df, "utf-8")
            json.dump(js_count_df, fp)

        json_opt_thresh = opt_thresh.T.to_json()
        with open(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json'), "w") as fp:
            js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
            json.dump(js_opt_thresh, fp)


    # 多分类模型评分,每次只选取单类别的检出框，把其余所有类别作为负样本。先把框匹配成结节，再用阈值对结节的最高概率进行筛选
    def multi_class_evaluation_nodule_threshold(self):

        predict_df_boxes_dict, ground_truth_boxes_dict = self.load_data()

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
                        filtered_predict_boxes = predict_df_boxes[predict_df_boxes["nodule_class"] == cls]
                        print filtered_predict_boxes
                        filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                    else:
                        filtered_predict_boxes = pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'nodule_class': [], 'prob': [], 'Mask': []})

                    if not ground_truth_boxes_dict[key].empty:
                        filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["nodule_class"] == cls]
                        print filtered_gt_boxes
                        filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                    else:
                        filtered_gt_boxes = pd.DataFrame(
                            {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'nodule_class': [], 'prob': [], 'Mask': []})

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
                                                        focus_priority_array=None,
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
                                                   focus_priority_array=None,
                                                   skip_init=True)

                    print "gt_nodules:"
                    print cls_gt_df

                    cls_predict_df = cls_predict_df[cls_predict_df['prob'] >= thresh]
                    cls_predict_df = cls_predict_df.reset_index(drop=True)
                    cls_predict_df_list.append(json_df_2_df(cls_predict_df))

                    cls_gt_df = cls_gt_df[cls_gt_df['prob'] >= thresh]
                    cls_gt_df = cls_gt_df.reset_index(drop=True)
                    cls_gt_df_list.append(json_df_2_df(cls_gt_df))

                    self.nodule_count += len(cls_predict_df.index)

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
                                                      'nodule_count': self.nodule_count,
                                                      'tp_count': cls_metric.tp,
                                                      'fp_count': cls_metric.fp,
                                                      'fn_count': cls_metric.fn,
                                                      'accuracy': cls_metric.get_acc(),
                                                      'recall': cls_metric.get_rec(),
                                                      'precision': cls_metric.get_prec(),
                                                      'fp/tp': fp_tp,
                                                      self.score_type: cls_metric.get_fscore(beta=self.fscore_beta)},
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
                self.cls_value[self.score_type].append(cls_metric.get_fscore(beta=self.fscore_beta))

            # 增加多类别加权平均的结果
            self.count_df = self.count_df.append({'nodule_class': 'average',
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

        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        print ("saving %s" % os.path.join(self.result_save_dir, self.xlsx_name))

        # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name)):
            os.remove(os.path.join(self.result_save_dir, self.xlsx_name))
        writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name))
        self.count_df.to_excel(writer, 'multi-class_evaluation', index=False)

        opt_thresh = opt_thresh.reset_index(drop=True)
        opt_thresh.to_excel(writer, 'optimal_threshold')
        writer.save()

        print ("saving %s" % os.path.join(self.result_save_dir, self.json_name))
        # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation_nodule_threshold.json')):
            os.remove(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation_nodule_threshold.json'))
        if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json')):
            os.remove(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json'))

        json_count_df = self.count_df.T.to_json()
        with open(os.path.join(self.result_save_dir, self.json_name + '_multi-class_evaluation_nodule_threshold.json'), "w") as fp:
            js_count_df = json.loads(json_count_df, "utf-8")
            json.dump(js_count_df, fp)

        json_opt_thresh = opt_thresh.T.to_json()
        with open(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json'), "w") as fp:
            js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
            json.dump(js_opt_thresh, fp)


    # 二分类（检出）模型统计,将所有正样本类别统计在一起
    def binary_class_evaluation(self):

        predict_df_boxes_dict, gt_df_boxes_dict = self.load_data()

        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            predict_df_list = []
            gt_df_list = []
            self.nodule_count = 0.
            for index, key in enumerate(predict_df_boxes_dict):
                self.patient_list.append(key)
                predict_df_boxes = predict_df_boxes_dict[key]
                gt_df_boxes = gt_df_boxes_dict[key]

                print ('processing %s' % key)

                # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                if not predict_df_boxes_dict[key].empty:
                    filtered_predict_boxes = predict_df_boxes[predict_df_boxes["prob"] >= thresh]
                    filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                else:
                    filtered_predict_boxes = pd.DataFrame(
                        {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                         'nodule_class': [], 'prob': [], 'Mask': []})

                if not gt_df_boxes_dict[key].empty:
                    filtered_gt_boxes = gt_df_boxes[gt_df_boxes["prob"] >= thresh]
                    filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                else:
                    filtered_gt_boxes = pd.DataFrame(
                        {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                         'nodule_class': [], 'prob': [], 'Mask': []})

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
                                                    focus_priority_array=None,
                                                    skip_init=True)
                print "predict_nodules:"
                print predict_df

                print "gt_boxes:"
                print filtered_gt_boxes
                _, gt_df = get_nodule_stat(dicom_names=None,
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
                print gt_df
                self.nodule_count += len(predict_df)
                predict_df = predict_df.reset_index(drop=True)
                predict_df_list.append(json_df_2_df(predict_df))

                gt_df = gt_df.reset_index(drop=True)
                gt_df_list.append(json_df_2_df(gt_df))

            # convert pandas dataframe to list of class labels
            cls_pred_labels, cls_gt_labels = df_to_cls_label(predict_df_list, gt_df_list, self.cls_name)

            # initialize ClassificationMetric class and update with ground truth/predict labels
            cls_metric = ClassificationMetric(cls_num=1, if_binary=True)


            cls_metric.update(cls_gt_labels, cls_pred_labels)
            if cls_metric.tp == 0:
                fp_tp = np.nan
            else:
                fp_tp = cls_metric.fp / cls_metric.tp
            self.count_df = self.count_df.append({'nodule_class': 'nodule',
                                                  'threshold': thresh,
                                                  'nodule_count': self.nodule_count,
                                                  'tp_count': cls_metric.tp,
                                                  'fp_count': cls_metric.fp,
                                                  'fn_count': cls_metric.fn,
                                                  'accuracy': cls_metric.get_acc(),
                                                  'recall': cls_metric.get_rec(),
                                                  'precision': cls_metric.get_prec(),
                                                  'fp/tp': fp_tp,
                                                  self.score_type: cls_metric.get_fscore(beta=self.fscore_beta)},
                                                 ignore_index=True)

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
        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        print ("saving %s" % os.path.join(self.result_save_dir, self.xlsx_name))

        #　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件

        if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name)):
            os.remove(os.path.join(self.result_save_dir, self.xlsx_name))
        writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name))
        self.count_df.to_excel(writer, 'binary-class_evaluation', index=False)
        opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        opt_thresh = opt_thresh.reset_index(drop=True)
        opt_thresh.to_excel(writer, 'optimal_threshold')
        writer.save()

        print ("saving %s" % os.path.join(self.result_save_dir, self.json_name))
        # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(
                os.path.join(self.result_save_dir, self.json_name + '_binary-class_evaluation.json')):
            os.remove(
                os.path.join(self.result_save_dir, self.json_name + '_binary-class_evaluation.json'))
        if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json')):
            os.remove(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json'))

        json_count_df = self.count_df.T.to_json()
        with open(os.path.join(self.result_save_dir, self.json_name + '_binary-class_evaluation.json'),
                  "w") as fp:
            js_count_df = json.loads(json_count_df, "utf-8")
            json.dump(js_count_df, fp)

        json_opt_thresh = opt_thresh.T.to_json()
        with open(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold.json'), "w") as fp:
            js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
            json.dump(js_opt_thresh, fp)

    #先把框匹配成结节，再用阈值对结节的最高概率进行筛选
    def binary_class_evaluation_nodule_threshold(self):

        predict_df_boxes_dict, gt_df_boxes_dict = self.load_data()

        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            self.nodule_count = 0
            predict_df_list = []
            gt_df_list = []
            for index, key in enumerate(predict_df_boxes_dict):
                self.patient_list.append(key)
                predict_df_boxes = predict_df_boxes_dict[key]
                gt_df_boxes = gt_df_boxes_dict[key]

                print ('processing %s' % key)

                # 　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                if not predict_df_boxes_dict[key].empty:
                    filtered_predict_boxes = predict_df_boxes.reset_index(drop=True)
                else:
                    filtered_predict_boxes = pd.DataFrame(
                        {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                         'nodule_class': [], 'prob': [], 'Mask': []})

                if not gt_df_boxes_dict[key].empty:
                    filtered_gt_boxes = gt_df_boxes.reset_index(drop=True)
                else:
                    filtered_gt_boxes = pd.DataFrame(
                        {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                         'nodule_class': [], 'prob': [], 'Mask': []})

                # 将预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
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
                                                focus_priority_array=None,
                                                skip_init=True)
                print "predict_nodules:"
                print predict_df

                print "gt_boxes:"
                print filtered_gt_boxes
                _, gt_df = get_nodule_stat(dicom_names=None,
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
                print gt_df
                self.nodule_count += len(predict_df)
                predict_df = predict_df[predict_df['prob'] >= thresh]
                predict_df = predict_df.reset_index(drop=True)
                predict_df_list.append(json_df_2_df(predict_df))

                gt_df = gt_df[gt_df['prob'] >= thresh]
                gt_df = gt_df.reset_index(drop=True)
                gt_df_list.append(json_df_2_df(gt_df))

            # convert pandas dataframe to list of class labels
            cls_pred_labels, cls_gt_labels = df_to_cls_label(predict_df_list, gt_df_list, self.cls_name)

            # initialize ClassificationMetric class and update with ground truth/predict labels
            cls_metric = ClassificationMetric(cls_num=1, if_binary=True)

            cls_metric.update(cls_gt_labels, cls_pred_labels)
            if cls_metric.tp == 0:
                fp_tp = np.nan
            else:
                fp_tp = cls_metric.fp / cls_metric.tp
            self.count_df = self.count_df.append({'nodule_class': 'nodule',
                                                  'threshold': thresh,
                                                  'nodule_count': self.nodule_count,
                                                  'tp_count': cls_metric.tp,
                                                  'fp_count': cls_metric.fp,
                                                  'fn_count': cls_metric.fn,
                                                  'accuracy': cls_metric.get_acc(),
                                                  'recall': cls_metric.get_rec(),
                                                  'precision': cls_metric.get_prec(),
                                                  'fp/tp': fp_tp,
                                                  self.score_type: cls_metric.get_fscore(beta=self.fscore_beta)},
                                                 ignore_index=True)

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
        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        print ("saving %s" % os.path.join(self.result_save_dir, self.xlsx_name))

        # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件

        if os.path.isfile(os.path.join(self.result_save_dir, self.xlsx_name)):
            os.remove(os.path.join(self.result_save_dir, self.xlsx_name))
        writer = pd.ExcelWriter(os.path.join(self.result_save_dir, self.xlsx_name))
        self.count_df.to_excel(writer, 'binary-class_evaluation', index=False)
        opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        opt_thresh = opt_thresh.reset_index(drop=True)
        opt_thresh.to_excel(writer, 'optimal threshold')
        writer.save()

        print ("saving %s" % os.path.join(self.result_save_dir, self.json_name))
        # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
        if os.path.isfile(
                os.path.join(self.result_save_dir,
                             self.json_name + '_binary-class_evaluation_nodule_threshold.json')):
            os.remove(
                os.path.join(self.result_save_dir,
                             self.json_name + '_binary-class_evaluation_nodule_threshold.json'))
        if os.path.isfile(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json')):
            os.remove(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json'))

        json_count_df = self.count_df.T.to_json()
        with open(os.path.join(self.result_save_dir,
                               self.json_name + '_binary-class_evaluation_nodule_threshold.json'),
                  "w") as fp:
            js_count_df = json.loads(json_count_df, "utf-8")
            json.dump(js_count_df, fp)

        json_opt_thresh = opt_thresh.T.to_json()
        with open(os.path.join(self.result_save_dir, self.json_name + '_optimal_threshold_nodule_threshold.json'), "w") as fp:
            js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
            json.dump(js_opt_thresh, fp)

    # 读入预测结果数据

    def load_data(self):
        """
        读入模型输出的.json和ground truth的.xml标记
        :return: 模型预测结果、ground truth标记按病人号排列的pandas.DataFrame
        e.g. | Mask | instanceNumber | nodule_class | prob | sliceId | xmax | xmin | ymax | ymin |
             |  []  |       106      | solid nodule | 0.9  | 105.0   | 207.0| 182.0| 230.0| 205.0|
        """
        predict_df_boxes_dict = {}
        ground_truth_boxes_dict = {}
        # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
        for PatientID in os.listdir(self.data_dir):
            if self.data_type == 'json':
                predict_json_path = os.path.join(self.data_dir, PatientID, PatientID + '_predict.json')
                try:
                    predict_df_boxes = pd.read_json(predict_json_path).T
                except:
                    print ("broken directory structure, maybe no prediction json file found: %s" % predict_json_path)
                    raise NameError
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
                # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于1000个层，故我们在这里选择1000
                ground_truth_boxes = xml_to_boxeslist(ground_truth_path, 1000)
            except:
                print ("broken directory structure, maybe no ground truth xml file found: %s" % ground_truth_path)
                ground_truth_boxes = [[[[]]]]

            ground_truth_boxes = init_df_boxes(return_boxes=ground_truth_boxes, classes=self.cls_name)
            ground_truth_boxes = ground_truth_boxes.sort_values(by=['prob'])
            ground_truth_boxes = ground_truth_boxes.reset_index(drop=True)

            predict_df_boxes_dict[PatientID] = predict_df_boxes
            ground_truth_boxes_dict[PatientID] = ground_truth_boxes
        return predict_df_boxes_dict, ground_truth_boxes_dict

    # 由predict出的框和ground truth anno生成_nodules.json和_gt.json
    def generate_df_nodules_to_json(self):
        """
        读入_predict.json及gt annotation文件，经过get_nodule_stat转换为json文件并存储到指定目录
        """

        predict_df_boxes_dict, ground_truth_boxes_dict = self.load_data()

        # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
        for PatientID in os.listdir(self.data_dir):
            predict_df_boxes = predict_df_boxes_dict[PatientID]
            ground_truth_boxes = ground_truth_boxes_dict[PatientID]

            if predict_df_boxes.empty:
                predict_df_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                   'nodule_class': [], 'prob': [], 'Mask': []})
            else:
                predict_df_boxes = predict_df_boxes.reset_index(drop=True)

            if ground_truth_boxes.empty:
                ground_truth_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                   'nodule_class': [], 'prob': [], 'Mask': []})
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
    def __init__(self, gt_anno_dir, conf_thres = 1., result_save_dir = os.path.join(os.getcwd(), 'FindNodulesEvaluation_result'),
                 xlsx_name = 'FindNodulesEvaluation', json_name = 'FindNodulesEvaluation', algorithm = 'find_nodules_new', if_nodule_cls = True,
                 same_box_threshold_gt = config.FIND_NODULES.SAME_BOX_THRESHOLD_GT, score_threshold_gt = config.FIND_NODULES.SCORE_THRESHOLD_GT,
                 z_threshold_gt = config.CLASS_Z_THRESHOLD_GT):
        assert os.path.isdir(gt_anno_dir), 'must initialize it with a valid directory of annotation data'
        self.gt_anno_dir = gt_anno_dir
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.patient_list = []
        self.cls_name = config.CLASSES
        self.conf_thresh = conf_thres
        self.algorithm = algorithm
        self.result_df = pd.DataFrame(
            columns=['patientid', 'algorithm', 'gt_nodule_count', 'nodule_count', 'threshold', 'adjusted_rand_index', 'adjusted_mutual_info_score',
                     'normalized_mutual_info_score', 'homogeneity_completeness_v_measure', 'fowlkes_mallows_score', 'silhouette_score'])
        self.nodule_count_df = pd.DataFrame(
            columns=['patientid', 'nodule_count'])
        # whether the ground truth labels include nodule class or not
        self.if_nodule_cls = if_nodule_cls
        self.nodule_cls_weights = config.CLASS_WEIGHTS
        self.same_box_threshold_gt = same_box_threshold_gt
        self.score_threshold_gt = score_threshold_gt
        self.z_threshold_gt = z_threshold_gt


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
                     'nodule_class': [], 'prob': [], 'Mask': []})

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
                        post_find_nodules_label.append(box_lst.iloc[i]['nodule'])
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
                     'nodule_class': [], 'prob': [], 'Mask': []})

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
                # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于1000个层，故我们在这里选择1000
                ground_truth_boxes, ground_truth_boxes_all_slice= xml_to_boxeslist_with_nodule_num(ground_truth_path, 1000)
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
            # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于1000个层，故我们在这里选择1000
                ground_truth_boxes = xml_to_boxeslist_without_nodule_cls(ground_truth_path, 1000)
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
            # 对于ground truth boxes,我们直接读取其xml标签。因为几乎所有CT图像少于1000个层，故我们在这里选择1000
                ground_truth_boxes, ground_truth_boxes_all_slice= xml_to_boxeslist_with_nodule_num_without_nodule_cls(ground_truth_path, 1000)
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

def df_2_box_list(df):
    raise NotImplemented

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
                        default='./excel_result',
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
                        help='name of xlsx',
                        default='result.xlsx',
                        type=str)
    parser.add_argument('--gt_anno_dir',
                        help='ground truth anno stored dir',
                        default='./anno',
                        type=str)
    parser.add_argument('--multi_class',
                        help='multi-class evaluation', action='store_true')
    parser.add_argument('--score_type',
                        help='type of model score',
                        default='F_score',
                        type=str)
    args = parser.parse_args()
    return args





