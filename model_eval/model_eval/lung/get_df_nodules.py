# coding:utf-8
import sys
import numpy as np
import pandas as pd


from objmatch.find_objects import find_objects

# nodule_class = config.CLASSES
PI = 3.141592654
ABNORMAL_VAL = -9999


def mask_cor_to_img_cor(xmin, xmax, xmask):
    '''
    从mask矩阵坐标映射回图片像素坐标
    :param xmin: 图片像素坐标下边界
    :param xmax: 图片像素坐标上边界
    :param xmask: mask矩阵坐标
    :return: 图片像素坐标
    '''
    return int(round(xmin + (xmax - xmin) * xmask / 63))



def init_df_boxes(return_boxes, classes):
    '''
    初始化一个bndbox的DataFrame
    :param dicom_names: dicom序列，获取起始instanceNumber
    :param return_boxes: 模型返回格式的boxes_list
    :param classes: 结节种类表
    :return: 初始化好的bndbox的DataFrame，和起始层面的instanceNumber
    '''
    flag_dcm_ins_num = False
    df_boxes = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
                             'class': [], 'prob': [], 'mask': []})
    df_boxes['instanceNumber'] = df_boxes['instanceNumber'].astype(np.int64)
    for i_slice in range(len(return_boxes)):
        return_boxes_slice = return_boxes[i_slice]
        if len(return_boxes_slice) == 0:
            continue
        else:
            if flag_dcm_ins_num:
                pass
            else:
                instance_number = i_slice + 1
            for i_cls in range(len(return_boxes_slice)):
                return_boxes_cls = return_boxes_slice[i_cls]
                for bndbox in return_boxes_cls:
                    # 尚未加入mask功能，暂存空，未来添加
                    mask = []
                    df_add_row = {'sliceId': i_slice, 'instanceNumber': instance_number,
                                  'xmin': bndbox[0], 'ymin': bndbox[1], 'xmax': bndbox[2], 'ymax': bndbox[3],
                                  'class': classes[i_cls + 1], 'prob': bndbox[4], 'mask': mask}
                    df_boxes = df_boxes.append(df_add_row, ignore_index=True)
    return df_boxes



def calc_series_minus_priority(list_name, focus_priority_array=None):
    if focus_priority_array is None:
        focus_priority_array = {"mass": 6,
                                "calcific nodule": 5,
                                "solid nodule": 4,
                                "GGN": 3,
                                "0-3nodule": 2,
                                "nodule": 1}
    list_priority = []
    for name in list_name:
        if name in focus_priority_array:
            list_priority.append(-focus_priority_array[name])
        else:
            list_priority.append(0)
    return list_priority


def get_density(max_weights):
    '''
    根据排序好的Hu值list计算密度
    从大到小取值，取到3个负值停止或取到超过列表长度的一半停止
    计算平均值
    :param max_weights: 排序好的Hu值列表
    :return: 密度值
    '''
    l = len(max_weights)
    if l == 0:
        return -9999
    MAX_MINUS = 2
    MAX_LEN = int(l / 2 + 1)
    sum_weight = 0
    cnt_minus = 0
    i = 0
    while i < l:
        hu = max_weights[i]
        if hu < 0:
            cnt_minus += 1
        if cnt_minus > MAX_MINUS:
            break
        sum_weight += hu
        i += 1
        if i > MAX_LEN:
            break
    return sum_weight / i


def add_nodule_to_df(df_add_nodule, df_nodules, bndbox_list, slice_range, nodule_prob_list,
                     nodule_type):
    '''
    把统计好的一个结节，加入结节DataFrame
    :param df_add_nodule: 一个结节行
    :param df_nodules: 结节表
    :param bndbox_list: 统计好的结节对应的bndbox_list
    :param slice_range: 统计好的结节对应的层面数
    :return: 添加好的结节DataFrame
    '''
    nodule_prob = np.max(nodule_prob_list)

    df_add_nodule['Bndbox List'] = bndbox_list
    df_add_nodule['SliceRange'] = slice_range
    df_add_nodule['Prob'] = nodule_prob
    df_add_nodule['Type'] = nodule_type
    return df_nodules.append(df_add_nodule, ignore_index=True)


def get_nodule_stat(dicom_names, return_boxes, prefix, classes, z_threshold, same_box_threshold=np.array([0.8, 0.8]), hu_img_array=None, img_spacing=None, if_dicom=True,
                    focus_priority_array=None, skip_init=False, score_threshold = 0.8, nodule_cls_weights = {}):
    '''
    调用find_nodules,把结节信息统计进DataFrame
    :param dicom_names: dicom序列路径，用于获取起始instanceNumber
    :param hu_img_array: Honus Value Array，在进行密度计算时会使用
    :param return_boxes: bndbox list
    :param img_spacing: 从dicom中读出来的spacing
    :param prefix: patient_id，实际上是新定义的txid
    :param classes: 结节种类表
    :return: df_boxes，各个bndbox和相应的信息；df_nodules，统计出来的结节DataFrame
    '''
    # 初始化框层面DataFrame
    if focus_priority_array == None:
        focus_priority_array = {"mass": 6,
                                "calcific nodule": 5,
                                "solid nodule": 4,
                                "GGN": 3,
                                "0-3nodule": 2,
                                 "nodule": 1}

    if skip_init:
        df_boxes = return_boxes
    else:
        df_boxes = init_df_boxes(dicom_names, return_boxes, classes, if_dicom)

    # 调用find_nodules计算结节和获取结节编号
    bbox_info, nodule_list = find_objects(df_boxes, SAME_BOX_THRESHOLD=same_box_threshold, Z_THRESHOLD=z_threshold,
                                          SCORE_THRESHOLD=score_threshold, object_cls_weights=nodule_cls_weights)
    # 结节编号排序
    #如果df_boxes已有结节信息，例如ssd的数据，则需要先删掉'nodule'这一列才能添加find_nodules生成的结节信息,对于'minusNamePriority', 'minusProb'亦同理
    try:
        df_boxes = df_boxes.drop(columns=['object'])
    except:
        print ("no 'object' in df_boxes")
    try:
        df_boxes = df_boxes.drop(columns=['minusNamePriority'])
    except:
        print ("no 'minusNamePriority' in df_boxes")
    try:
        df_boxes = df_boxes.drop(columns=['minusProb'])
    except:
        print ("no 'minusProb' in df_boxes")

    df_boxes.insert(0, 'object', bbox_info['object'])

    list_name = list(df_boxes["class"])

    series_minus_priority = calc_series_minus_priority(list_name)
    series_minus_prob = -df_boxes["prob"]
    df_boxes.insert(0, 'minusNamePriority', series_minus_priority)
    df_boxes.insert(0, 'minusProb', series_minus_prob)
    df_boxes = df_boxes.sort_values(['object', 'instanceNumber', 'minusNamePriority', 'minusProb'], ascending=True)
    df_boxes = df_boxes.reset_index(drop=True)

    # 初始化结节DataFrame
    df_nodules = pd.DataFrame({'Bndbox List': [], 'Object Id': [], 'Pid': prefix, 'Type': [],
                               'SliceRange': [], 'Prob': []})
    df_add_nodule = {}
    bndbox_list = []
    slice_range = []
    nodule_prob_list = []
    i_row = 0
    len_df = len(df_boxes)
    last_nodule = -1

    while i_row <= len_df:
        # 判断存储
        if i_row != len_df:
            row = df_boxes.iloc[i_row]
            now_nodule = row["object"]
        else:
            now_nodule = ABNORMAL_VAL

        if now_nodule == -1:
            i_row += 1
            continue
        elif (now_nodule != last_nodule or i_row == len_df):
            # 新结节
            # 添加结节信息之前还要进行一些检查
            if (not bndbox_list) or (not slice_range):
                # print "something is empty ..."
                pass
            elif last_nodule >= 0:
                df_nodules = add_nodule_to_df(df_add_nodule,
                                              df_nodules,
                                              bndbox_list,
                                              slice_range,
                                              nodule_prob_list,
                                              nodule_type)
            # 新结节统计信息初始化，但要保证不是最后一行
            if i_row == len_df:
                break
            bndbox_list = []
            slice_range = []
            nodule_prob_list = []
            now_ins_num = ABNORMAL_VAL
            nodule_priority = ABNORMAL_VAL
            nodule_type = "NoType"
            df_add_nodule = {'Object Id': now_nodule, 'Bndbox List': [], 'Type': ABNORMAL_VAL,
                             'Pid': prefix}
        # 跳过条件
        skip_flag = False
        # 当前非结节正常id
        if now_nodule < 0:
            skip_flag = True
        # 如果同一结节，层面相同，则跳过, 排序就是为了这一步有效
        if (now_nodule == last_nodule) and (row["instanceNumber"] == now_ins_num):
            skip_flag = True

        if skip_flag:
            i_row += 1
            nodule_prob_list.append(row["prob"])
            continue

        # 非跳过，结节统计操作，包括第一个结节，不管是不是新结节，在不跳过的情况下，以下会被执行
        # 层面优先级小于结节优先级 minusNamePriority minusProb
        row_priority = -row["minusNamePriority"]
        if row_priority > nodule_priority:  # 清空prob，如果priority更新
            nodule_priority = row_priority
            nodule_type = row["class"]
            nodule_prob_list = []
        if row_priority == nodule_priority:
            nodule_prob_list.append(row["prob"])
        now_ins_num = row["instanceNumber"]
        bndbox_list.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"], row["prob"]])
        slice_range.append(int(row["sliceId"]) + 1)

        last_nodule = now_nodule
        i_row += 1

    return df_boxes, df_nodules
