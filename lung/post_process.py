# -- coding: utf-8 --
import pandas as pd
import os
import argparse
import json
import numpy as np
import cv2
import shutil
import copy
import find_nodules_new
import config

from scipy import stats
import math
def df_to_cls_label(cls_predict_df_list, cls_gt_df_list, cls_list):
    """Convert list of pandas dataframe to list of list of cls labels, as input for custom_metric.ClassificationMetric

    :param cls_predict_df_list:
    :param cls_gt_df_list:
    :param cls_list:
    :return:

    Examples
    --------
    predicts = [np.array([[1], [1], [1]])]
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
        # No ground truth nodules, all fp
        if cls_gt_df is None:
            for index, row in cls_predict_df.iterrows():
                cls_gt_label.append(0)
                # print cls_list.index(row['nodule_class'])
                cls_predict_label.append(cls_list.index(row['nodule_class']))
        # No predicted nodules, all fn
        elif cls_predict_df is None:
            for index, row in cls_gt_df.iterrows():
                cls_predict_label.append(0)

                # print cls_list.index(row['nodule_class'])
                cls_gt_label.append(cls_list.index(row['nodule_class']))
        # non-zero gt and predicted nodules
        else:
            cov_matrix = np.zeros([len(cls_predict_df), len(cls_gt_df)])
            for i, row_pred in cls_predict_df.iterrows():
                # print cls_list.index(row_pred['nodule_class'])
                cls_predict_label.append(cls_list.index(row_pred['nodule_class']))
                for j, row_gt in cls_gt_df.iterrows():
                    if nodule_compare(row_pred, row_gt):
                        cov_matrix[i , j] = 1
                if np.sum(cov_matrix[i, :]) > 0:
                    cls_gt_label.append(cls_list.index(row_pred['nodule_class']))
                else:
                    cls_gt_label.append(0)
            for i, row_gt in cls_gt_df.iterrows():
                if np.sum(cov_matrix[:, i]) == 0:
                    # print cls_list.index(row_gt['nodule_class'])
                    cls_gt_label.append(cls_list.index(row_gt['nodule_class']))
                    cls_predict_label.append(0)
        cls_gt_labels.append(cls_gt_label)
        cls_predict_labels.append(cls_predict_label)
    print "cls_pred"
    print cls_predict_labels
    print "cls_gt"
    print cls_gt_labels
    return cls_predict_labels, cls_gt_labels




def nodule_compare(predict_df_record, ground_truth_df_record, z_threshold=1):
    """
    将一个预测结节和一个实际结节比较，给出是否匹配
    :param predict_df_record:一条预测记录,代表一个结节，DataFrame
    :param ground_truth_df_record:一条ground truth记录，代表一个结节，DataFrame
    :param z_threshold:上下多找多少张
    :return:比较后的的结果，bool
    """
    predict_slices = range(predict_df_record['slice'][0] - z_threshold,
                           predict_df_record['slice'][-1] + 1 + z_threshold)
    ground_truth_slices = range(ground_truth_df_record['slice'][0],
                                ground_truth_df_record['slice'][-1] + 1)
    predict_bboxs = []
    ground_truth_bboxs = []
    # 补全缺失层面的bbox
    for predict_slice in predict_slices:
        if not predict_slice in predict_df_record['slice']:
            predict_bboxs.append([None, None, None, None])
        else:
            i = predict_df_record['slice'].index(predict_slice)
            predict_bboxs.append(predict_df_record['bbox'][i][:4])
    for ground_truth_slice in ground_truth_slices:
        if not ground_truth_slice in ground_truth_df_record['slice']:
            ground_truth_bboxs.append([None, None, None, None])
        else:
            i = ground_truth_df_record['slice'].index(ground_truth_slice)
            ground_truth_bboxs.append(ground_truth_df_record['bbox'][i][:4])
    # 用插值法补全bbox
    predict_bboxs = np.array(pd.DataFrame(predict_bboxs).interpolate(limit_direction='both'))
    ground_truth_bboxs = np.array(pd.DataFrame(ground_truth_bboxs).interpolate(limit_direction='both'))

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
        # score = find_nodules.calcDICE(predict_bboxs[predict_index], ground_truth_bboxs[ground_truth_index])
        if_same_bbox = cal_same_bbox(ground_truth_bboxs[ground_truth_index], predict_bboxs[predict_index])
        # 只要有一对满足标准，就算对
        ret_if_same_bbox = ret_if_same_bbox or if_same_bbox
        if ret_if_same_bbox:
            break
    return ret_if_same_bbox


def cal_same_bbox(bbox_gt, bbox_pt, iou_thresh = 0.5):
    return IOU(bbox_gt, bbox_pt) > iou_thresh

def overlapND(x, y):
    """
    :params x: 2*n np array
    :params y: 2*n np array
    """
    assert (x.shape[0] == 2)
    assert (y.shape[0] == 2)
    res = []
    for i in range(x.shape[1]):
        res.append(overlap1D(x[:, i], y[:, i]))
    return np.vstack(res).T

def overlap1D(x, y):
    """
    Returns the overlap of 1d segment, returns [0, 0] if not overlapped.
    :params x: 1d np array of 2 elements. [st, ed]
    :params y: 1d np array of 2 elements. [st ,ed]
    """
    st = np.max([x[0], y[0]])
    ed = np.min([x[1], y[1]])
    return np.array([st, ed])

def IOU(box1, box2):
    box1 = box1.copy().reshape([2, 2])
    box2 = box2.copy().reshape([2, 2])
    if np.any(box1[1] - box1[0] <= 0) or np.any(box2[1] - box2[0] <= 0):
        raise ValueError(
            "Boxes should be represented as [xmin, ymin, xmax, ymax]. Box1: %s. Box2: %s. " % (str(box1), str(box2)))

    res = overlapND(box1, box2)
    # return res
    if np.any(res[1] - res[0] <= 0):
        return 0.0
    return 2 * float(np.prod(res[1] - res[0])) / (np.prod(box1[1] - box1[0]) + np.prod(box2[1] - box2[0]) - np.prod(res[1] - res[0]))