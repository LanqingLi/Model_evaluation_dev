# -- coding: utf-8 --
import pandas as pd
import os
import argparse
import json
import numpy as np
import cv2
import shutil
import copy
from common_metrics import AnchorMetric


def df_to_cls_label(cls_predict_df_list, cls_gt_df_list, cls_list, thresh, dim=2):
    """Convert list of pandas dataframe to list of list of cls labels, as input for custom_metric.ClassificationMetric

    :param cls_predict_df_list:
    :param cls_gt_df_list:
    :param cls_list:
    :return:

    Examples
    --------
    predicts = [np.array([1, 1, 1])]
    labels = [np.array([0, 1, 1])]
    acc = metric.Accuracy()
    acc.update(preds = predicts, labels = labels)
    print acc.get()
    ('accuracy', 0.666666666666666666)
    """
    cls_predict_labels = []
    cls_gt_labels = []
    for cls_predict_df, cls_gt_df in zip(cls_predict_df_list, cls_gt_df_list):
        cls_predict_label = []
        cls_gt_label = []
        # No ground truth objects, all fp
        if cls_gt_df is None:
            for index, row in cls_predict_df.iterrows():
                cls_gt_label.append(0)
                # print cls_list.index(row['object_class'])
                cls_predict_label.append(cls_list.index(row['class']))
        # No predicted objects, all fn
        elif cls_predict_df is None:
            for index, row in cls_gt_df.iterrows():
                cls_predict_label.append(0)

                cls_gt_label.append(cls_list.index(row['class']))
        # non-zero gt and predicted objects
        else:
            pred_object = []
            gt_object = []

            # generate interpolated object list
            for i, row_pred in cls_predict_df.iterrows():
                predict_slices, predict_bboxs = object_slice_interpolate_pred(row_pred, z_threshold=1)
                pred_object.append((predict_slices, predict_bboxs))

            for j, row_gt in cls_gt_df.iterrows():
                gt_slices, gt_bboxs = object_slice_interpolate_gt(row_gt)
                gt_object.append((gt_slices, gt_bboxs))

            # construct covariance matrix to account for the equivalence between the gt and predicted objects
            cov_matrix = np.zeros([len(cls_gt_df), len(cls_predict_df)])
            for i, row_gt in cls_gt_df.iterrows():
                # To account for all samples, we compare each detected object with gt, if it doesnt belong to gt, we add a new one and label it as fp
                # Note that for object detection, we didnt account for the background class when running the prediction code, so tn = 0.
                cls_gt_label.append(cls_list.index(row_gt['class']))
                for j, row_pred in cls_predict_df.iterrows():
                    if object_compare(predict_slices=pred_object[j][0], ground_truth_slices=gt_object[i][0],
                                      predict_bboxs=pred_object[j][1], ground_truth_bboxs=gt_object[i][1], thresh=thresh, dim=dim):
                        cov_matrix[i , j] = 1
                if np.sum(cov_matrix[i, :]) > 0:
                    cls_predict_label.append(cls_list.index(row_gt['class']))
                else:
                    cls_predict_label.append(0)
            for i, row_pred in cls_predict_df.iterrows():
                if np.sum(cov_matrix[:, i]) == 0:
                    cls_predict_label.append(cls_list.index(row_pred['class']))
                    cls_gt_label.append(0)
        cls_gt_labels.append(cls_gt_label)
        cls_predict_labels.append(cls_predict_label)

    return cls_predict_labels, cls_gt_labels


def df_to_xlsx_file(cls_predict_df_list, cls_gt_df_list,thresh, dim=2):
    """Convert list of pandas dataframe to list of list of cls labels, as input for custom_metric.ClassificationMetric

    :param cls_predict_df_list:
    :param cls_gt_df_list:
    :param cls_list:
    :return:

    Examples
    --------
    predicts = [np.array([1, 1, 1])]
    labels = [np.array([0, 1, 1])]
    acc = metric.Accuracy()
    acc.update(preds = predicts, labels = labels)
    print acc.get()
    ('accuracy', 0.666666666666666666)
    """
    fn_list = []
    fp_list = []
    tp_list = []
    for cls_predict_df, cls_gt_df in zip(cls_predict_df_list, cls_gt_df_list):
        # No ground truth objects, all fp
        if cls_gt_df is None:
            for index, row in cls_predict_df.iterrows():
                fp_list.append([row['pid'], row['slice'], row['bbox'], np.nan, np.nan, 'FP',
                                row['class'],np.nan,row['prob']
                                ])
        # No predicted objects, all fn
        elif cls_predict_df is None:
            for index, row in cls_gt_df.iterrows():
                fn_list.append([row['pid'],np.nan,np.nan,row['slice'],row['bbox'],'FN',np.nan,row['class'],row['prob']])
        # non-zero gt and predicted objects
        else:
            pred_object = []
            gt_object = []

            # generate interpolated object list
            for i, row_pred in cls_predict_df.iterrows():
                predict_slices, predict_bboxs = object_slice_interpolate_pred(row_pred, z_threshold=1)
                pred_object.append((predict_slices, predict_bboxs))

            for j, row_gt in cls_gt_df.iterrows():
                gt_slices, gt_bboxs = object_slice_interpolate_gt(row_gt)
                gt_object.append((gt_slices, gt_bboxs))

            # construct covariance matrix to account for the equivalence between the gt and predicted objects
            cov_matrix = np.zeros([len(cls_gt_df), len(cls_predict_df)])
            for i, row_gt in cls_gt_df.iterrows():
                for j, row_pred in cls_predict_df.iterrows():

                    if object_compare(predict_slices=pred_object[j][0], ground_truth_slices=gt_object[i][0],
                                      predict_bboxs=pred_object[j][1], ground_truth_bboxs=gt_object[i][1], thresh=thresh, dim=dim):
                        cov_matrix[i , j] = 1


            gt_count,pred_count=cov_matrix.shape

            for index in range(pred_count):
                if np.sum(cov_matrix[:,index])==0:
                    fp_list.append([cls_predict_df.iloc[index]['pid'], cls_predict_df.iloc[index]['slice'], cls_predict_df.iloc[index]['bbox'], np.nan, np.nan, 'FP',
                                    cls_predict_df.iloc[index]['class'],np.nan,cls_predict_df.iloc[index]['prob']])

            for index in range(gt_count):
                if np.sum(cov_matrix[index])==0:
                    fn_list.append([cls_gt_df.iloc[index]['pid'],np.nan,np.nan,cls_gt_df.iloc[index]['slice'],cls_gt_df.iloc[index]['bbox'],'FN',
                                    np.nan,cls_gt_df.iloc[index]['class'],cls_gt_df.loc[index,'prob'], cls_gt_df.loc[index, 'diameter'], cls_gt_df.loc[index, 'ct_value']])
                else:
                    print 'cls_predict_df'
                    print cls_predict_df
                    pred_class = list(cls_predict_df.loc[cov_matrix[index].astype(np.bool), 'class'])
                    pred_slice = list(cls_predict_df.loc[cov_matrix[index].astype(np.bool), 'slice'])
                    gt_slice = cls_gt_df.loc[index, 'slice']
                    pre_bbox = list(cls_predict_df.loc[cov_matrix[index].astype(np.bool), 'bbox'])
                    gt_bbox = cls_gt_df.loc[index, 'bbox']
                    pre_prob = max(cls_predict_df.loc[cov_matrix[index].astype(np.bool), 'prob'])
                    gt_class=cls_gt_df.loc[index,'class']
                    gt_diameter = cls_gt_df.loc[index, 'diameter']
                    gt_hu = cls_gt_df.loc[index, 'ct_value']

                    tp_list.append(
                        [cls_gt_df.loc[index,'pid'], pred_slice, pre_bbox, gt_slice, gt_bbox, 'TP', pred_class,
                         gt_class, pre_prob, gt_diameter, gt_hu])

    columns_gt = ['PatientID', 'PreSlices', 'Prebbox', 'GtSlices',
               'Gtbbox', 'Result', 'predict_class', 'ground_truth_class','Prob', 'Diameter', 'CT_value']
    columns_pred = ['PatientID', 'PreSlices', 'Prebbox', 'GtSlices',
               'Gtbbox', 'Result', 'predict_class', 'ground_truth_class','Prob']
    ret_df = pd.DataFrame(columns=columns_gt)
    ret_df = ret_df.append(pd.DataFrame(fp_list, columns=columns_pred))
    ret_df = ret_df.append(pd.DataFrame(fn_list, columns=columns_gt))
    ret_df = ret_df.append(pd.DataFrame(tp_list, columns=columns_gt))
    return ret_df


def object_slice_interpolate_pred(predict_df_record, z_threshold=1):
    """
    :param predict_df_record:一条预测记录,代表一个结节，DataFrame

    :param z_threshold:上下多找多少张

    """
    print predict_df_record['pid']
    predict_slices = range(predict_df_record['slice'][0] - z_threshold,
                           predict_df_record['slice'][-1] + 1 + z_threshold)

    predict_bboxs = []

    # 补全缺失层面的bbox
    for predict_slice in predict_slices:
        if not predict_slice in predict_df_record['slice']:
            predict_bboxs.append([None, None, None, None])
        else:
            i = predict_df_record['slice'].index(predict_slice)
            predict_bboxs.append(predict_df_record['bbox'][i][:4])

    # 用插值法补全bbox
    predict_bboxs = np.array(pd.DataFrame(predict_bboxs).interpolate(limit_direction='both'))


    return predict_slices, predict_bboxs

def object_slice_interpolate_gt(ground_truth_df_record):
    """
    :param ground_truth_df_record:一条ground truth记录，代表一个结节，DataFrame
    """
    print ground_truth_df_record['pid']
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

def object_compare(predict_slices, ground_truth_slices, predict_bboxs, ground_truth_bboxs, thresh, dim):
    """
    将一个预测结节和一个实际结节比较，给出是否匹配
    :param predict_df_record:一条预测记录,代表一个结节，DataFrame
    :param ground_truth_df_record:一条ground truth记录，代表一个结节，DataFrame
    :param z_threshold:上下多找多少张
    :return:比较后的的结果，bool
    """
    # 计算交叉层面bbox的DICE
    cross_slices = list(set(predict_slices) & set(ground_truth_slices))
    # 无交叉返回0
    if len(cross_slices) == 0:
        return False
    # 有交叉，只要有一对满足标准，就算对
    ret_if_same_bbox = False
    for slice in cross_slices:
        predict_index = predict_slices.index(slice)
        ground_truth_index = ground_truth_slices.index(slice)
        # score = find_objects.calcDICE(predict_bboxs[predict_index], ground_truth_bboxs[ground_truth_index])
        if_same_bbox = cal_same_bbox_cdi(np.asarray([ground_truth_bboxs[ground_truth_index]]), np.asarray([predict_bboxs[predict_index]]), thresh=thresh, dim=dim)
        # 只要有一对满足标准，就算对
        ret_if_same_bbox = ret_if_same_bbox or if_same_bbox
        if ret_if_same_bbox:
            break
    return ret_if_same_bbox

def cal_same_bbox_iou(bbox_gt, bbox_pt, thresh, dim):
    anchor_metric = AnchorMetric(dim=dim)
    return anchor_metric.iou(bbox_gt, bbox_pt) > thresh

def cal_same_bbox_cdi(bbox_gt, bbox_pt, thresh, dim):
    """
    determine whether two bounding boxes are the same according to AnchorMetric.center_deviation_iou metric

    :param bbox_gt:
    :param bbox_pt:
    :param thresh:
    :param dim:
    :return:
    """
    anchor_metric = AnchorMetric(dim=dim)
    return np.all(anchor_metric.center_deviation_iou(bbox_gt, bbox_pt) < thresh)

