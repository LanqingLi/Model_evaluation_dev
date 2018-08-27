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
                # Note that for object detection, we didnt account for the background class when run the prediction code, so tn = 0.
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

def object_slice_interpolate_pred(predict_df_record, z_threshold=1):
    """
    :param predict_df_record:一条预测记录,代表一个结节，DataFrame

    :param z_threshold:上下多找多少张

    """
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
        if_same_bbox = cal_same_bbox(np.asarray([ground_truth_bboxs[ground_truth_index]]), np.asarray([predict_bboxs[predict_index]]), thresh=thresh, dim=dim)
        # 只要有一对满足标准，就算对
        ret_if_same_bbox = ret_if_same_bbox or if_same_bbox
        if ret_if_same_bbox:
            break
    return ret_if_same_bbox


def cal_same_bbox(bbox_gt, bbox_pt, thresh, dim):
    anchor_metric = AnchorMetric(dim=dim)
    return anchor_metric.iou(bbox_gt, bbox_pt) > thresh

