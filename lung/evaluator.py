# -- coding: utf-8 --
import json
import pandas as pd
import numpy as np
import os
import argparse
import shutil
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.custom_metric import ClassificationMetric, cls_avg

from lung.get_df_nodules import init_df_boxes
from lung.xml_tools import xml_to_boxeslist
from lung.config import config
from lung.post_process import df_to_cls_label
from lung.get_df_nodules import get_nodule_stat

# from common.utils import generate_df_nodules_2_json

class LungNoduleEvaluatorOffline(object):
    '''
    this class is designed for evaluation of our CT lung model offline. It can read anchor boxes from a selection of format (.json/.npy)
    and generate spreadsheets of statistics (mAP etc.) for each nodule class under customized range of classification (softmax) probability

    threshold, which can be used for plotting ROC curve and calculating AUC.
    :param data_dir: 存储预测出的框的信息的数据路径，我们读入数据的路径
    :param data_type: 存储预测出的框的信息的数据格式，默认为.npy，我们读入数据的格式。对于FRCNN,我们将clean_box_new输出的框存成.npy供读取
    :param anno_dir: 存储对应CT数据标记(annotation)的路径
    :param cls_name: 包含预测所有类别的列表，默认为config.CLASSES, 包含'__background__'类
    :param cls_dict: 包含'rcnn/classname_labelname_mapping.xls'中label_name到class_name映射的字典，不包含'__background__'类
    :param opt_thresh: 存储最终使得各类别模型预测结果最优的概率阈值及其对应tp,fp,score等信息的字典，index为预测的类别。每个index对应一个类似于
    self.count_df的字典，最终存储在model_evaluation.xlsx/optimal threshold中
    :param score_type: 模型评分的函数类型，默认为F_score
    :param same_box_threshold: 当预测出的框被输入get_nodule_stat，会调用韦人的find_nodules()进行结节匹配，其中对于相似度高于某个same_box_threshold
    的结节（任意类别）会只保留一个。在做多分类时我们必须设置其为1.0,因为相同结节的NMS已经在Predict类中做过了
    :param count_df: 初始化的pandas.DataFrame,用于存储最终输出的model evaluation结果
    :param xlsx_save_dir:　存储输出.xlsx结果的路径
    :param xlsx_name: 存储输出.xlsx文件的名字
    :param conf_thresh:　自定义的概率阈值采样点，存在列表中，用于求最优阈值及画ROC曲线

    :param nodule_cls_weights:　不同结节种类对于模型综合评分的权重
    '''
    def __init__(self, data_dir, data_type, anno_dir, score_type = 'fscore',  xlsx_save_dir = os.path.join(os.getcwd(), 'excel_result'),
                 xlsx_name = 'LungModelEvaluation.xlsx', if_generate_nodule_json = True,
                 conf_thresh = np.linspace(0.1, 0.85, num=16).tolist() + np.linspace(0.9, 0.975, num=4).tolist()\
                           + np.linspace(0.99, 0.9975, num=4).tolist() + np.linspace(0.999, 0.99975, num=4).tolist(),
                 nodule_cls_weights = config.CLASS_WEIGHTS):
                 # nodule_cls_weights =  {'3-6nodule': 3,
                 #                    '6-10nodule': 5,
                 #                    'pleural nodule': 1,
                 #                    '10-30nodule': 10,
                 #                    'calcific nodule': 1,
                 #                    '0-5GGN': 5,
                 #                    '5GGN': 8,
                 #                    '0-3nodule': 1,
                 #                    'mass': 10,
                 #                    'not_mass': 0,
                 #                    }):
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
            columns=['nodule_class', 'threshold','tp_count', 'fp_count', 'fn_count', 'accuracy', 'recall', 'precision', 'fp/tp', self.score_type])

        self.if_generate_nodule_json = if_generate_nodule_json
        self.xlsx_save_dir = xlsx_save_dir
        self.xlsx_name = xlsx_name
        # customized confidence threshold for plotting ROC curve
        self.conf_thresh = conf_thresh
        self.nodule_cls_weights = nodule_cls_weights
        self.patient_dict = []
        self.cls_weight = []
        self.cls_value = {'accuracy': [], 'recall': [], 'precision': [], self.score_type: []}

    # 多分类模型评分,每次只选取单类别的检出框，把其余所有类别作为负样本
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
                for index, key in enumerate(predict_df_boxes_dict):
                    self.patient_dict.append(key)
                    predict_df_boxes = predict_df_boxes_dict[key]
                    ground_truth_boxes = ground_truth_boxes_dict[key]

                    print ('processing %s' %key)

                    #　筛选probability超过规定阈值且预测为规定类别的框输入get_nodule_stat
                    if not predict_df_boxes_dict[key].empty:
                        filtered_predict_boxes = predict_df_boxes[predict_df_boxes["nodule_class"] == cls]
                        filtered_predict_boxes = filtered_predict_boxes[filtered_predict_boxes["prob"] >= thresh]
                        filtered_predict_boxes = filtered_predict_boxes.reset_index(drop=True)
                    else:
                        filtered_predict_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                 'nodule_class': [], 'prob': [], 'Mask': []})

                    if not ground_truth_boxes_dict[key].empty:
                        filtered_gt_boxes = ground_truth_boxes[ground_truth_boxes["nodule_class"] == cls]
                        filtered_gt_boxes = filtered_gt_boxes[filtered_gt_boxes["prob"] >= thresh]
                        filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                    else:
                        filtered_gt_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                 'nodule_class': [], 'prob': [], 'Mask': []})

                    #　将模型预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                    print "predict:"
                    _, cls_predict_df = get_nodule_stat(dicom_names=None,
                                             hu_img_array=None,
                                             return_boxes=filtered_predict_boxes,
                                             img_spacing=None,
                                             prefix=key,
                                             classes=self.cls_name,
                                             same_box_threshold=config.FIND_NODULES.SAME_BOX_THRESHOLD_PRED,
                                             score_threshold=config.FIND_NODULES.SCORE_THRESHOLD_PRED,
                                             z_threshold=config.FIND_NODULES.Z_THRESHOLD_PRED,
                                             if_dicom=False,
                                             focus_priority_array=None,
                                             skip_init=True)
                    print "ground truth:"

                    _, cls_gt_df = get_nodule_stat(dicom_names=None,
                                                     hu_img_array=None,
                                                     return_boxes=filtered_gt_boxes,
                                                     img_spacing=None,
                                                     prefix=key,
                                                     classes=self.cls_name,
                                                     same_box_threshold=config.FIND_NODULES.SAME_BOX_THRESHOLD_GT,
                                                     score_threshold=config.FIND_NODULES.SCORE_THRESHOLD_GT,
                                                     z_threshold=config.FIND_NODULES.Z_THRESHOLD_GT,
                                                     if_dicom=False,
                                                     focus_priority_array=None,
                                                     skip_init=True)


                    cls_predict_df = cls_predict_df.reset_index(drop=True)
                    cls_predict_df_list.append(json_df_2_df(cls_predict_df))

                    cls_gt_df = cls_gt_df.reset_index(drop=True)
                    cls_gt_df_list.append(json_df_2_df(cls_gt_df))

                # convert pandas dataframe to list of class labels
                cls_pred_labels, cls_gt_labels = df_to_cls_label(cls_predict_df_list, cls_gt_df_list, self.cls_name)

                # initialize ClassificationMetric class and update with ground truth/predict labels
                cls_metric = ClassificationMetric(cls_num=i_cls)

                # convert labels list to np.ndarray to fit in the metric class
                # cls_gt_labels = np.asarray(cls_gt_labels)
                # cls_pred_labels = np.asarray(cls_pred_labels)
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
                self.cls_value[self.score_type].append(cls_metric.get_fscore(beta=1.))

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

        if not os.path.exists(self.xlsx_save_dir):
            os.makedirs(self.xlsx_save_dir)
        print ("saving %s" %os.path.join(self.xlsx_save_dir, self.xlsx_name))

        # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件

        if os.path.isfile(os.path.join(self.xlsx_save_dir, self.xlsx_name)):
            os.remove(os.path.join(self.xlsx_save_dir, self.xlsx_name))
        writer = pd.ExcelWriter(os.path.join(self.xlsx_save_dir, self.xlsx_name))
        self.count_df.to_excel(writer, 'multi-class evaluation', index=False)

        opt_thresh = opt_thresh.reset_index(drop=True)
        # print opt_thresh
        opt_thresh.to_excel(writer, 'optimal threshold')
        writer.save()




    # 二分类（检出）模型统计,将所有正样本类别统计在一起
    def binary_class_evaluation(self):

        predict_df_boxes_dict, gt_df_boxes_dict = self.load_data()

        # 为了画ROC曲线做模型评分，我们取0.1到1的多个阈值并对predict_df_boxes做筛选
        for thresh in self.conf_thresh:
            predict_df_list = []
            gt_df_list = []
            for index, key in enumerate(predict_df_boxes_dict):
                self.patient_dict.append(key)
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
                    print gt_df_boxes
                    filtered_gt_boxes = gt_df_boxes[gt_df_boxes["prob"] >= thresh]
                    filtered_gt_boxes = filtered_gt_boxes.reset_index(drop=True)
                else:
                    filtered_gt_boxes = pd.DataFrame(
                        {'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                         'nodule_class': [], 'prob': [], 'Mask': []})

                # 　将预测出来的框(filtered_predict_boxes)与标记的ground truth框(filtered_gt_boxes)输入get_nodule_stat进行结节匹配
                print "generating predicted nodules:"
                _, predict_df = get_nodule_stat(dicom_names=None,
                                                    hu_img_array=None,
                                                    return_boxes=filtered_predict_boxes,
                                                    img_spacing=None,
                                                    prefix=key,
                                                    classes=self.cls_name,
                                                    same_box_threshold=config.FIND_NODULES.SAME_BOX_THRESHOLD_PRED,
                                                    score_threshold=config.FIND_NODULES.SCORE_THRESHOLD_PRED,
                                                    z_threshold=config.FIND_NODULES.Z_THRESHOLD_PRED,
                                                    if_dicom=False,
                                                    focus_priority_array=None,
                                                    skip_init=True)
                print "generating ground truth nodules:"
                _, gt_df = get_nodule_stat(dicom_names=None,
                                               hu_img_array=None,
                                               return_boxes=filtered_gt_boxes,
                                               img_spacing=None,
                                               prefix=key,
                                               classes=self.cls_name,
                                               same_box_threshold=config.FIND_NODULES.SAME_BOX_THRESHOLD_GT,
                                               score_threshold=config.FIND_NODULES.SCORE_THRESHOLD_GT,
                                               z_threshold=config.FIND_NODULES.Z_THRESHOLD_GT,
                                               if_dicom=False,
                                               focus_priority_array=None,
                                               skip_init=True)

                predict_df = predict_df.reset_index(drop=True)
                predict_df_list.append(json_df_2_df(predict_df))

                gt_df = gt_df.reset_index(drop=True)
                gt_df_list.append(json_df_2_df(gt_df))

            # convert pandas dataframe to list of class labels
            cls_pred_labels, cls_gt_labels = df_to_cls_label(predict_df_list, gt_df_list, self.cls_name)

            # initialize ClassificationMetric class and update with ground truth/predict labels
            cls_metric = ClassificationMetric(cls_num=1, if_binary=True)


            cls_metric.update(cls_gt_labels, cls_pred_labels)

            self.count_df = self.count_df.append({'nodule_class': 'nodule',
                                                  'threshold': thresh,
                                                  'tp_count': cls_metric.tp,
                                                  'fp_count': cls_metric.fp,
                                                  'fn_count': cls_metric.fn,
                                                  'accuracy': cls_metric.get_acc(),
                                                  'recall': cls_metric.get_rec(),
                                                  'precision': cls_metric.get_prec(),
                                                  'fp/tp': cls_metric.fp / cls_metric.tp,
                                                  self.score_type: cls_metric.get_fscore(beta=1.)},
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
        if not os.path.exists(self.xlsx_save_dir):
            os.makedirs(self.xlsx_save_dir)
        print ("saving %s" % os.path.join(self.xlsx_save_dir, self.xlsx_name))

        #　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件

        if os.path.isfile(os.path.join(self.xlsx_save_dir, self.xlsx_name)):
            os.remove(os.path.join(self.xlsx_save_dir, self.xlsx_name))
        writer = pd.ExcelWriter(os.path.join(self.xlsx_save_dir, self.xlsx_name))
        self.count_df.to_excel(writer, 'binary-class evaluation', index=False)
        print self.opt_thresh
        opt_thresh = pd.DataFrame.from_dict(self.opt_thresh, orient='index')
        opt_thresh = opt_thresh.reset_index(drop=True)
        print opt_thresh
        opt_thresh.to_excel(writer, 'optimal threshold')
        writer.save()

    # 读入预测结果数据

    def load_data(self):
        """

        :return:

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
            elif self.data_type == 'npy':
                predict_npy_path = os.path.join(self.data_dir, PatientID, PatientID + '_predict.npy')
                try:
                    predict_boxes = np.load(predict_npy_path)
                except:
                    print ("broken directory structure, maybe no prediction npy file found: %s" % predict_npy_path)
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

    # 计算多分类average precision
    def get_mAP(self, dataframe):
        tot_weight = 0
        mAP = 0
        for key in self.nodule_cls_weights:
            tot_weight += self.nodule_cls_weights[key]
            mAP += self.nodule_cls_weights[key] * dataframe[key]['precision']
        return  float(mAP) / tot_weight



    # 由predict出的框和ground truth anno生成_nodules.json和_gt.json
    def generate_df_nodules_2_json(self):
        """
        读入_predict.json及gt annotation文件，经过get_nodule_stat转换为json文件并存储到指定目录
        :return:
        """

        predict_df_boxes_dict, ground_truth_boxes_dict = self.load_data()

        # 将所有预测病人的json/npy文件(包含所有层面所有种类的框)转换为DataFrame
        for PatientID in os.listdir(self.data_dir):
            predict_df_boxes = predict_df_boxes_dict[PatientID]
            ground_truth_boxes = ground_truth_boxes_dict[PatientID]

            if predict_df_boxes.empty:
                predict_df_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                   'nodule_class': [], 'prob': [], 'Mask': []})
            if ground_truth_boxes.empty:
                ground_truth_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                                                   'nodule_class': [], 'prob': [], 'Mask': []})
            print "prediction:"
            # predict_df_boxes = predict_df_boxes.reset_index(drop=True)
            _, predict_df = get_nodule_stat(dicom_names=None,
                                            hu_img_array=None,
                                            return_boxes=predict_df_boxes,
                                            img_spacing=None,
                                            prefix=PatientID,
                                            classes=self.cls_name,
                                            same_box_threshold=self.same_box_threshold,
                                            score_threshold=self.score_threshold_predict,
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
                                       same_box_threshold=np.array([0., 0.]),
                                       score_threshold=self.score_threshold_gt,
                                       if_dicom=False,
                                       focus_priority_array=None,
                                       skip_init=True)
            str_nodules = predict_df.T.to_json()
            str_gt = gt_df.T.to_json()
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            json_patient_auto_test_dir = os.path.join(self.data_dir, PatientID)
            print json_patient_auto_test_dir
            if not os.path.exists(json_patient_auto_test_dir):
                os.mkdir(json_patient_auto_test_dir)
            with open(os.path.join(json_patient_auto_test_dir, PatientID + '_nodule.json'), "w") as fp:
                js_nodules = json.loads(str_nodules, "utf-8")
                json.dump(js_nodules, fp)
            with open(os.path.join(json_patient_auto_test_dir, PatientID + '_gt.json'), "w") as fp:
                js_gt = json.loads(str_gt, "utf-8")
                json.dump(js_gt, fp)

    # 筛选一定层厚以上的最终输出的结节（降假阳实验）
    # def nodule_thickness_filter(self, thickness_thresh):
    #     assert type(thickness_thresh) == int, "input thickness_thresh should be an integer, not %s" %thickness_thresh
    #     for PatientID in os.listdir(self.data_dir):
    #         if self.data_type == 'json':
    #             predict_json_path = os.path.join(self.data_dir, PatientID, PatientID + '_nodule.json')
    #         try:
    #             # print predict_json_path
    #             predict_df_boxes = pd.read_json(predict_json_path).T
    #         except:
    #             raise ("broken directory structure, maybe no prediction json file found: %s" % predict_json_path)
    #         # print predict_df_boxes, predict_df_boxes.index
    #         drop_list = []
    #         for i, row in predict_df_boxes.iterrows():
    #             if len(row['SliceRange']) <= thickness_thresh:
    #                 # print row['SliceRange'], i
    #                 drop_list.append(i)
    #         predict_df_boxes = predict_df_boxes.drop(drop_list)
    #         # print predict_df_boxes
    #         generate_df_nodules_2_json(predict_df_boxes, 'json_for_auto_test', PatientID + "_nodule%s.json" %(thickness_thresh))


# def get_F_score(tp, fp, fn, recall, precision, precision_thresh=0.):
#     if recall is np.nan or precision is np.nan:
#         return np.nan
#     else:
#         if precision < precision_thresh:
#             return 0.0
#         else:
#             return 2*precision*recall/(precision + recall)




# def save_json_2_df(json_path):
#     """
#     将json文件转换为DataFrame
#     :param json_path: 具体json文件的路径
#     :return: pandas的DataFrame
#             'bbox'字段：bbox的坐标，[xmax,ymax,xmin,ymin]，list类型
#             'pid'字段：病人的PatientID，unicode类型
#             'slice':这个bbox所在的slice，float类型
#             'nodule_class':这个结节的分类，unicode类型
#             'nodule_id':归属于哪一个结节，属于同一结节的记录nodule_id相同，float类型
#     """
#     df = pd.read_json(json_path)
#     df = df.T
#     ret_df = pd.DataFrame({'bbox': [], 'pid': [], 'slice': [], 'nodule_class': [], 'nodule_id': []})
#     for index, row in df.iterrows():
#         df_add_row = {'bbox': [bbox for bbox in row['Bndbox List']],
#                       'pid': row['Pid'],
#                       'slice': row['SliceRange'],
#                       'nodule_class': row['Type'],
#                       'nodule_id': row['Nodule Id'],
#                       'prob': row['prob']}
#         ret_df = ret_df.append(df_add_row, ignore_index=True)
#    return ret_df

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

def df_2_box_list(df):
    raise NotImplemented

def parse_args():
    parser = argparse.ArgumentParser(description='Infervision auto test')
    parser.add_argument('--data_dir',
                        help='predict result stored dir, .npy by default',
                        default='./npy_for_auto_test',
                        type=str)
    parser.add_argument('--data_type',
                        help='type of data that store prediction result',
                        default='npy',
                        type=str
                        )
    parser.add_argument('--xlsx_save_dir',
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

if __name__ == '__main__':
    args = parse_args()

    print config.CLASSES
    model_eval = LungNoduleEvaluatorOffline(data_dir=args.data_dir,
                                  data_type=args.data_type,
                                  anno_dir=args.gt_anno_dir,
                                  score_type=args.score_type,
                                  xlsx_save_dir=args.xlsx_save_dir,
                                  xlsx_name=args.xlsx_name)
                                  #conf_thresh=np.linspace(0.7,0.7,num=1).tolist())
    # model_eval.generate_df_nodules_2_json()
    if args.multi_class:
        model_eval.multi_class_evaluation()
        print model_eval.opt_thresh


    else:
        model_eval.binary_class_evaluation()

    # model_eval.ROC_plot(os.path.join(os.getcwd(), 'excel_result'), 'multi_class_evaluation_FJDH.xlsx', 'multi-class evaluation')
    # thickness_thresh = [1., 2.]
    # #
    # for i in thickness_thresh:
    #     model_eval.nodule_thickness_filter(thickness_thresh=i)
    #
    #     auto_test(json_dir='./json_for_auto_test',
    #                             xlsx_save_dir='./excel_result',
    #                             image_dir=args.image_dir,
    #                             image_save_dir='./auto_test/NMS%s' %i,
    #                             if_dicom=args.dicom,
    #                             if_normalization=config.NORMALIZATION,
    #                             xlsx_name=args.xlsx_name,
    #                             json_name=('_nodule%s.json'%i),
    #                             if_save_image=True)

    # df = pd.read_excel(os.path.join(args.xlsx_save_dir, 'model_evaluation.xlsx'), sheet_name='optimal threshold')
    # mAP = Model_evaluation.get_mAP(df)
    # multi_class_score = Model_evaluation.multi_class_model_score(df)
    # print ('mAP for oprimal threshold is %f' %(mAP))
    # print ('multi_class_score for optimal threshold is %f' %(multi_class_score))
