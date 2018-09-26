# coding:utf-8
import numpy as np
import pandas as pd
import sys

from model_eval.tools.data_preprocess import get_instance_number

from objmatch.find_objects import find_objects

# nodule_class = config.CLASSES
PI = 3.141592654
ABNORMAL_VAL = -9999

def init_df_objects(slice_object_list, key_list, class_key):
    '''
    初始化一个object(anchor)的DataFrame
    :param slice_object_list: 一个以层面数排列的列表，每个元素是对应层面的所有object
    :param return_boxes: 模型返回格式的boxes_list
    :param classes: 结节种类表
    :return: 初始化好的bndbox的DataFrame，和起始层面的instanceNumber
    '''
    # print slice_object_list
    df_objects = pd.DataFrame()
    assert class_key in key_list, 'key_list must contain the object class keyword %s!' %class_key
    for i_slice, slice_object in enumerate(slice_object_list):
        if len(slice_object) > 0:
            for object in slice_object:
                assert set(key_list) == set(key_list).intersection(set(object.attr_dict.keys())), 'object keyword list must contain the designated(input) keyword list'
                df_add_row = pd.DataFrame([object.attr_dict])
                df_objects = df_objects.append(df_add_row, ignore_index=True)
    df_objects = df_objects.rename(index=str, columns={class_key: 'class', 'sliceId': 'instanceNumber'})
    df_objects['instanceNumber'] += 1
    df_objects['instanceNumber'] = df_objects['instanceNumber'].astype(np.int64)
    return df_objects

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


def add_object_to_df(df_objects, object_dict):
    '''
    把统计好的一个物体，加入物体DataFrame
    :param df_objects: 物体DataFrame
    :param object_dict: 存储新的物体信息的字典
    :return: 添加好的物体DataFrame
    '''
    new_df = pd.DataFrame([object_dict])
    df_objects = df_objects.append(new_df, ignore_index=True)
    return df_objects


def get_object_stat(slice_object_list, prefix, classes, z_threshold, key_list, class_key, matched_key_list,
                    same_box_threshold=np.array([0.8, 0.8]), hu_img_array=None, img_spacing=None, if_dicom=True,
                    focus_priority_array=None, skip_init=False, score_threshold = 0.8, nodule_cls_weights = {}):
    '''
    调用find_nodules,把结节信息统计进DataFrame
    :param hu_img_array: Honus Value Array，在进行密度计算时会使用
    :param return_boxes: bndbox list
    :param img_spacing: 从dicom中读出来的spacing
    :param prefix: patient_id，实际上是新定义的txid
    :param classes: 结节种类表
    :return: df_boxes，各个bndbox和相应的信息；df_nodules，统计出来的结节DataFrame
    '''

    # 初始化框层面DataFrame
    if skip_init:
        df_objects = slice_object_list
    else:
        df_objects = init_df_objects(slice_object_list, key_list, class_key)

    # 调用find_nodules计算结节和获取结节编号
    object_info, object_list = find_objects(df_objects, SAME_BOX_THRESHOLD=same_box_threshold, Z_THRESHOLD=z_threshold,
                                          SCORE_THRESHOLD=score_threshold, object_cls_weights=nodule_cls_weights)
    # 结节编号排序
    #如果df_boxes已有结节信息，例如ssd的数据，则需要先删掉'nodule'这一列才能添加find_nodules生成的结节信息,对于'minusNamePriority', 'minusProb'亦同理
    try:
        df_objects = df_objects.drop(columns=['object'])
    except:
        print ("no 'object' in df_objects")
    try:
        df_objects = df_objects.drop(columns=['minusNamePriority'])
    except:
        print ("no 'minusNamePriority' in df_objects")
    try:
        df_objects = df_objects.drop(columns=['minusProb'])
    except:
        print ("no 'minusProb' in df_objects")

    df_objects.insert(0, 'object', object_info['object'])

    list_name = list(df_objects["class"])

    series_minus_priority = calc_series_minus_priority(list_name, focus_priority_array)
    series_minus_prob = -df_objects["prob"]
    df_objects.insert(0, 'minusNamePriority', series_minus_priority)
    df_objects.insert(0, 'minusProb', series_minus_prob)
    df_objects = df_objects.sort_values(['object', 'instanceNumber', 'minusNamePriority', 'minusProb'], ascending=True)
    df_objects = df_objects.reset_index(drop=True)

    # 初始化结节DataFrame
    df_matched_objects = pd.DataFrame(columns=matched_key_list)
    matched_object_dict = {'Object Id': [], 'Bndbox List': [], 'Type': [], 'Pid': [], 'SliceRange': [], 'Prob': [],
                           'Diameter': [], 'CT_value': []}
    assert set(matched_object_dict.keys()) == set(matched_object_dict.keys()).intersection(
        set(matched_key_list)), 'matched object keyword list must contain the matched object dict keys'
    bndbox_list = []
    slice_range = []
    object_prob_list = []
    object_diameter_list = []
    object_hu_list = []
    i_row = 0
    len_df = len(df_objects)
    last_object = -1

    while i_row <= len_df and len_df > 0:
        if i_row == len_df:
            try:
                df_matched_objects = add_object_to_df(df_objects=df_matched_objects,
                                                      object_dict={'Object Id': cur_object,
                                                                   'Bndbox List': bndbox_list,
                                                                   'SliceRange': slice_range,
                                                                   'Pid': prefix, 'Prob': np.max(object_prob_list),
                                                                   'Type': object_type,
                                                                   'Diameter': object_diameter_list[0],
                                                                   'CT_value': object_hu_list[0]})
            except:
                df_matched_objects = add_object_to_df(df_objects=df_matched_objects,
                                                      object_dict={'Object Id': cur_object,
                                                                   'Bndbox List': bndbox_list,
                                                                   'SliceRange': slice_range,
                                                                   'Pid': prefix, 'Prob': np.max(object_prob_list),
                                                                   'Type': object_type})
            break
        # 判断存储
        row = df_objects.iloc[i_row]
        cur_object = row["object"]

        if cur_object == -1:
            i_row += 1
            continue
        elif (cur_object != last_object or i_row == len_df):
            # 新结节
            # 添加结节信息之前还要进行一些检查
            if (not bndbox_list) or (not slice_range):
                # print "something is empty ..."
                pass
            elif last_object >= 0:
                assert len(set(object_diameter_list)) <= 1, 'all slices of a single object must have the same diameter'
                assert len(set(object_hu_list)) <= 1, 'all slices of a single object must have the same hu value'

                try:
                    df_matched_objects = add_object_to_df(df_objects=df_matched_objects,
                                                      object_dict={'Object Id': cur_object-1, 'Bndbox List': bndbox_list, 'SliceRange': slice_range,
                                                      'Pid': prefix, 'Prob': np.max(object_prob_list), 'Type': object_type,
                                                      'Diameter': object_diameter_list[0] , 'CT_value': object_hu_list[0]})
                except:
                    df_matched_objects = add_object_to_df(df_objects=df_matched_objects,
                                                          object_dict={'Object Id': cur_object-1,
                                                                       'Bndbox List': bndbox_list,
                                                                       'SliceRange': slice_range,
                                                                       'Pid': prefix, 'Prob': np.max(object_prob_list),
                                                                       'Type': object_type})
            # 新结节统计信息初始化，但要保证不是最后一行
            if i_row == len_df:
                break
            bndbox_list = []
            slice_range = []
            object_prob_list = []
            object_diameter_list = []
            object_hu_list = []
            now_ins_num = ABNORMAL_VAL
            object_priority = ABNORMAL_VAL
            object_type = "NoType"

        # 跳过条件
        skip_flag = False
        # 当前非结节正常id
        if cur_object < 0:
            skip_flag = True
        # 如果同一结节，层面相同，则跳过, 排序就是为了这一步有效
        if (cur_object == last_object) and (row["instanceNumber"] == now_ins_num):
            skip_flag = True

        if skip_flag:
            i_row += 1
            object_prob_list.append(row["prob"])
            try:
                object_diameter_list.append(row["Diameter"])
                object_hu_list.append(row["CT_value"])
            except:
                pass
            continue

        # 非跳过，结节统计操作，包括第一个结节，不管是不是新结节，在不跳过的情况下，以下会被执行
        # 层面优先级小于结节优先级 minusNamePriority minusProb
        row_priority = -row["minusNamePriority"]
        if row_priority > object_priority:  # 清空prob，如果priority更新
            object_priority = row_priority
            object_type = row["class"]
            object_prob_list = []
            object_diameter_list = []
            object_hu_list = []
        if row_priority == object_priority:
            object_prob_list.append(row["prob"])
            try:
                object_diameter_list.append(row["Diameter"])
                object_hu_list.append(row["CT_value"])
            except:
                pass
        now_ins_num = row["instanceNumber"]
        bndbox_list.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"], row["prob"]])
        slice_range.append(int(row["instanceNumber"]))

        last_object = cur_object
        i_row += 1

    return df_objects, df_matched_objects
