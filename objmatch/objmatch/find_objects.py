# encoding: utf-8
import pandas as pd
import numpy as np
import networkx as nx
from common_metrics import AnchorMetric
# this script was written based on networkx version <= 2.0, and cannot be applied to networkx version later than 2.0
# one but not the only significant difference is that, networkx.algorithms.max_weight_matching <= 2.0 returns a dict,
# whereas networkx.algorithms.max_weight_matching > 2.0 returns a set

def get_bounding_box_nparray(bbox):
    return np.array([[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]])

def find_parent(id1, union_find_set):
    # 并查集用，查找功能
    if union_find_set[id1] == id1:
        return id1
    else:
        return find_parent(union_find_set[id1], union_find_set)

def union(id1, id2, union_find_set):
    # 并查集用，合并功能
    #if find_parent(id1, union_find_set) != find_parent(id2, union_find_set):
    union_find_set[find_parent(id1, union_find_set)] = find_parent(id2, union_find_set)

def add_object(objects, cur_object):
    # 增加一个新物体
    objects.append({"objectList": [cur_object]})
    objects[-1]["id"] = len(objects)

def norm_weight(weight_list):
    # normalize model weights
    weight_array = np.array(weight_list)
    assert weight_array.dtype == 'float', "input weights must have type 'float'"
    return list(weight_array / np.sum(weight_array))

def find_objects(bboxInfo, Z_THRESHOLD, SAME_BOX_THRESHOLD=np.array([1.6, 1.6]), SCORE_THRESHOLD=0.6,
                 object_cls_weights={}, same_cls_boost = 2.):

    """将boundingbox转换为结节。
    :param bboxInfo: Bounding Box表格（pandas.DataFrame类型）,index必须为自然顺序。
                     每一行代表一个预测出来的bounding box，包含下面的列：
                     * 'instanceNumber'列：表示当前bounding box所在的层面的InstanceNumber，编号从1开始。
                     * 'xmin'列：表示当前bounding box的左上角x坐标（高度[H]方向）。
                     * 'ymin'列：表示当前bounding box的左上角y坐标（左右[W]方向）。
                     * 'xmax'列：表示当前bounding box的右下角x坐标（高度[H]方向）。
                     * 'ymax'列：表示当前bounding box的右下角y坐标（左右[W]方向）。
                     * 'class'列：表示当前bounding box的预测类别（如'object', 'mass'等）。
                     * 'prob'里：表示当前bounding box的预测概率。
    :param Z_THRESHOLD： 每个新层面向前做贪心匹配时往前找的最大层面数
    :param SAME_BOX_THRESHOLD: 判断同一层面两个框是否为等价类的中心点偏移阈值
    :param SCORE_THRESHOLD: 判断不同层面两个框是否匹配的sim_metric_3d阈值
    :param object_cls_weights:　检出框的类别的权重
    :param same_cls_boost: 当不同层面两个框匹配时，如果类别相同的奖励系数
    :return objectInfo: 在bnd上附加一列'object'，取值为-1, 1..n。
                        -1代表当前Bounding box不属于任何一个object;
                        1..n代表当前bounding box所属的结节编号。
                        我们不允许一个结节在同一个层面内存在多个重合的bounding box(一个等价类中没有匹配上的框设为-1)。
    ":return objects: 结节信息的列表，每个元素为一个字典
    """
    # 初始化匹配anchor的度规类AnchorMetric:
    anchor_metric = AnchorMetric(dim=2)
    # 首先计算同一层面内dice coefficient比较高的， 并且认为这些bounding box标记了同一个结节
    bboxInfo = bboxInfo.copy()
    # 初始化并查集
    unionFindSet = bboxInfo.index.tolist()
    objectSlices = {}
    for i in bboxInfo["instanceNumber"].unique():
        lst = bboxInfo.query("instanceNumber == @i")
        if len(lst) > 1:
            for j1 in range(len(lst.index)):
                for j2 in range(j1 + 1, len(lst.index)):
                    # 如果两个box中心点相对位移小于SAME_BOX_THRESHOLD，那么认为这两个box表示同一个结节
                    iou = anchor_metric.iou(get_bounding_box_nparray(lst.iloc[j1]), get_bounding_box_nparray(lst.iloc[j2]))
                    if iou > 0 and np.all(anchor_metric.center_deviation_iou(get_bounding_box_nparray(lst.iloc[j1]),
                                  get_bounding_box_nparray(lst.iloc[j2])) < SAME_BOX_THRESHOLD):
                        # 将两个box插入一颗等价类树
                        union(lst.index[j1], lst.index[j2], unionFindSet)
        objectSlices[i] = lst.index.tolist()

    for i in range(len(unionFindSet)):
        unionFindSet[i] = find_parent(unionFindSet[i], unionFindSet)
    # unionFindSet保存了当前bounding box与本层面内其他bounding box的归属关系
    # 对于bounding box i，如果bboxInfo.loc[i]["unionFindSet"] != i，
    # 那么表明它与bboxInfo.loc[i]["unionFindSet"]表示同一个结节
    bboxInfo["unionFindSet"] = unionFindSet
    # 在不同层面间对结节进行匹配，并且获得结节列表
    objects = []
    BOXID_VALUE = 10000
    for curZ in sorted(objectSlices.keys()):
        # 枚举之前的所有结节，检查最后一个层面，如果与当前层面的instanceNumber差值在Z_THRESHOLD之内，那么作为备选结节加入lastBoxes
        lastBoxes = [{"objectID": k["id"],
                      "bndbox": get_bounding_box_nparray(bboxInfo.loc[k["objectList"][-1]]),
                      "class": bboxInfo.loc[k["objectList"][-1]]["class"],
                      "prob": bboxInfo.loc[k["objectList"][-1]]["prob"]}
                      for k in objects if 0 < curZ - bboxInfo.loc[k["objectList"][-1]]["instanceNumber"] <= \
                            Z_THRESHOLD[bboxInfo.loc[k["objectList"][-1]]["class"]]]
        # 枚举本层面所有bounding box，每个框的评分为类别评分*置信度概率
        curBoxes = [{"matched": False,
                     "boxID": k,
                     "bndbox": get_bounding_box_nparray(bboxInfo.loc[k]),
                     "class": bboxInfo.loc[k]["class"],
                     "prob": bboxInfo.loc[k]["prob"],
                     "score": object_cls_weights[bboxInfo.loc[k]["class"]] * bboxInfo.loc[k]["prob"]}
                     for k in objectSlices[curZ]]
        # 对于有多个box表示一个结节的，只选择其中一个
        # （选择bboxInfo.loc[i]["unionFindSet"] == i的那个，即等价类树的根节点）
        curBoxes_root = \
                    [{"boxID": k,
                     "bndbox": get_bounding_box_nparray(bboxInfo.loc[k]),
                     "class": bboxInfo.loc[k]["class"],
                     "prob": bboxInfo.loc[k]["prob"]}
                     for k in objectSlices[curZ] if bboxInfo.loc[k]["unionFindSet"] == k]
        # 选取每个等价类中评分最高的框，"matched"记录该框是否与前层结节相匹配（最大权匹配），若没有则视为新结节插入。
        curBoxes_union = []
        for i in curBoxes_root:
            Boxes_union = [j for j in curBoxes if bboxInfo.loc[j["boxID"]]["unionFindSet"] == i["boxID"]]
            Boxes_union = sorted(Boxes_union, key=lambda  k: k["score"])
            curBoxes_union.append(Boxes_union[-1])
        #　如果在之前层面（Z_THRESHOLD以内）没有结节，将新的等价类插入作为新结节的开始
        if len(lastBoxes) == 0:
            for k in curBoxes_union:
                add_object(objects, k["boxID"])
            continue
        # 建立二分图
        g = nx.Graph()
        g.add_nodes_from([i["boxID"] + BOXID_VALUE for i in curBoxes])
        g.add_nodes_from([i["objectID"] for i in lastBoxes])
        for i in lastBoxes:
            for j in curBoxes:
                # 定义不同层面的3D中心点相对偏移（sim_metric），因为结节在３D上移动很少，只对中心点相对偏移在一定阈值内的两个结节做匹配
                evalScore = anchor_metric.center_deviation_sqrt(i["bndbox"], j["bndbox"])
                if evalScore < SCORE_THRESHOLD:
                    if i["class"] not in object_cls_weights or j["class"] not in object_cls_weights:
                        print "object class not found in objects_cls_weights"
                        raise KeyError
                    object_weight = object_cls_weights[i["class"]] * i["prob"]
                    box_weight = object_cls_weights[j["class"]] * j["prob"]
                    # we suppress 3d matching for misaligned boxes, only valid for matching lung objects
                    misalign_suppress = np.exp(-evalScore)
                    # 定义不同层面两个bounding box的分数作为边权
                    if i["class"] == j["class"]:
                        matchingScore = same_cls_boost * object_weight * box_weight * misalign_suppress
                    else:
                        matchingScore = object_weight * box_weight * misalign_suppress
                    g.add_weighted_edges_from([[i["objectID"], j["boxID"] + BOXID_VALUE, matchingScore]])
        # 求出最大权匹配,networkx2.0之后nx.algorithms.max_weight_matching返回set,之前版本都是字典（本代码默认用较低版本运行）
        matchRes = nx.algorithms.max_weight_matching(g)
        matched_object_list = []
        reduced_matched_object_list = []
        for i in matchRes.keys():
            if i < BOXID_VALUE:
                matched_object_list.append(i)
                reduced_matched_object_list.append(i)

        if len(matched_object_list) == 0:
            for i in curBoxes_union:
                # 对于没有匹配上的bounding box，认为是一个新结节的开始
                if i["matched"] == False:
                    add_object(objects, i["boxID"])
            continue

        elif len(matched_object_list) == 1:
            box_union = bboxInfo.loc[matchRes[matched_object_list[0]] - BOXID_VALUE]["unionFindSet"]
            box_union_index = [i for i in range(len(curBoxes_union)) if
                               bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"] == box_union]
            if len(box_union_index) != 1:
                print ('there should be one and only one box for the same equivalent class in curBoxes_union')
                raise IndexError
            curBoxes_union[box_union_index[0]]["matched"] = True
        # 检查最大权匹配出的框有没有在一个等价类中，如果有，舍弃边权较低的那条边
        else:
            for j in range(len(matched_object_list)):
                if j < len(matched_object_list) - 1:
                    for k in range(len(matched_object_list)-j-1):
                        # 两个匹配了的框的等价类ID
                        box_union1 = bboxInfo.loc[matchRes[matched_object_list[j]]-BOXID_VALUE]["unionFindSet"]
                        box_union2 = bboxInfo.loc[matchRes[matched_object_list[j+k+1]]-BOXID_VALUE]["unionFindSet"]

                        box_union_index1 = [i for i in range(len(curBoxes_union)) if bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"]== box_union1]
                        box_union_index2 = [i for i in range(len(curBoxes_union)) if bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"]== box_union2]

                        if len(box_union_index1) != 1 or len(box_union_index2) != 1:
                            print ('there should be one and only one box for the same equivalent class in curBoxes_union')
                            raise  IndexError
                        curBoxes_union[box_union_index1[0]]["matched"] = True
                        curBoxes_union[box_union_index2[0]]["matched"] = True
                        if box_union1 == box_union2:
                            if g[matched_object_list[j]][matchRes[matched_object_list[j]]]['weight'] <= \
                                    g[matched_object_list[j+k+1]][matchRes[matched_object_list[j+k+1]]]['weight']:
                                if matched_object_list[j] in reduced_matched_object_list:
                                    reduced_matched_object_list.remove(matched_object_list[j])
                            elif matched_object_list[j+k+1] in reduced_matched_object_list:
                                reduced_matched_object_list.remove(matched_object_list[j+k+1])

        # 对于已经匹配上的bounding box，加入对应的objectList中
        for i in reduced_matched_object_list:

            objects[i - 1]["objectList"].append(matchRes[i] - BOXID_VALUE)

        for i in curBoxes_union:
            # 对于没有匹配上的bounding box，认为是一个新结节的开始
            if i["matched"] == False:
                add_object(objects, i["boxID"])

    object_result = [-1] * len(bboxInfo)
    for i in objects:
        for j in i["objectList"]:
            object_result[j] = i["id"]

    bboxInfo["object"] = object_result
    return bboxInfo[["instanceNumber", "xmin", "ymin", "xmax", "ymax", "class", "prob", "object"]], objects

def find_objects_ensemble(bboxInfo_list, MODEL_WEIGHT_LIST, MODEL_CONF_LIST, Z_THRESHOLD, OBJ_FREQ_THRESH=None,
                          SAME_BOX_THRESHOLD=np.array([1.6, 1.6]), SCORE_THRESHOLD=0.6, object_cls_weights={}, same_cls_boost = 2.):

    """将boundingbox转换为结节。
    :param bboxInfo_List: 一个列表，每个元素对应多模型中一个模型预测出的Bounding Box表格（pandas.DataFrame类型）的列表,index必须为自然顺序。
                     每一行代表一个预测出来的bounding box，包含下面的列：
                     * 'instanceNumber'列：表示当前bounding box所在的层面的InstanceNumber，编号从1开始。
                     * 'xmin'列：表示当前bounding box的左上角x坐标（高度[H]方向）。
                     * 'ymin'列：表示当前bounding box的左上角y坐标（左右[W]方向）。
                     * 'xmax'列：表示当前bounding box的右下角x坐标（高度[H]方向）。
                     * 'ymax'列：表示当前bounding box的右下角y坐标（左右[W]方向）。
                     * 'class'列：表示当前bounding box的预测类别（如'object', 'mass'等）。
                     * 'prob'里：表示当前bounding box的预测概率。
    :param Z_THRESHOLD： 每个新层面向前做贪心匹配时往前找的最大层面数
    :param MODEL_WEIGHT_LIST: 存储各模型权重的列表，用于模型结果的加权
    :param MODEL_CONF_LIST:　存储各模型置信度概率的列表，用于估计各模型各检出框为真阳的概率
    :param OBJECT_FREQ_THRESH: 匹配时判断是否保留各等价类框的频率阈值
    :param SAME_BOX_THRESHOLD: 判断同一层面两个框是否为等价类的中心点偏移阈值
    :param SCORE_THRESHOLD: 判断不同层面两个框是否匹配的sim_metric_3d阈值
    :param object_cls_weights:　检出框的类别的权重
    :param same_cls_boost: 当不同层面两个框匹配时，如果类别相同的奖励系数
    :return objectInfo: 在bnd上附加一列'object'，取值为-1, 1..n。
                        -1代表当前Bounding box不属于任何一个object;
                        1..n代表当前bounding box所属的结节编号。
                        我们不允许一个结节在同一个层面内存在多个重合的bounding box(一个等价类中没有匹配上的框设为-1)。
    ":return objects: 结节信息的列表，每个元素为一个字典
    """
    norm_weight_list = norm_weight(MODEL_WEIGHT_LIST)
    if OBJ_FREQ_THRESH==None:
        OBJ_FREQ_THRESH = np.sum(np.array(MODEL_CONF_LIST))
        print 'object frequency threshold =%f' %(OBJ_FREQ_THRESH)

    print 'bboxInfo_list:'
    print bboxInfo_list
    bboxInfo = pd.DataFrame()
    for i, bboxInfo_df in enumerate(bboxInfo_list):
        bboxInfo_i = bboxInfo_df.copy()
        bboxInfo_i['model_idx'] = i
        bboxInfo_i['prob'] = bboxInfo_i['prob'] * norm_weight_list[i]
        bboxInfo = bboxInfo.append(bboxInfo_i, ignore_index=True)

    assert len(bboxInfo_list) == len(MODEL_WEIGHT_LIST), 'bounding box list and model weight list must contain the same' \
                                                         'number of models'
    # 初始化匹配anchor的度规类AnchorMetric:
    anchor_metric = AnchorMetric(dim=2)
    # 初始化并查集
    unionFindSet = bboxInfo.index.tolist()
    objectSlices = {}

    print 'bboxInfo:'
    print bboxInfo
    for i in bboxInfo["instanceNumber"].unique():
        lst = bboxInfo.query("instanceNumber == @i")
        if len(lst) > 1:
            for j1 in range(len(lst.index)):
                for j2 in range(j1 + 1, len(lst.index)):
                    # 如果两个box中心点相对位移小于SAME_BOX_THRESHOLD，那么认为这两个box表示同一个结节
                    iou = anchor_metric.iou(get_bounding_box_nparray(lst.iloc[j1]), get_bounding_box_nparray(lst.iloc[j2]))
                    if iou > 0 and np.all(anchor_metric.center_deviation_iou(get_bounding_box_nparray(lst.iloc[j1]),
                                  get_bounding_box_nparray(lst.iloc[j2])) < SAME_BOX_THRESHOLD):
                        # 将两个box插入一颗等价类树
                        union(lst.index[j1], lst.index[j2], unionFindSet)
        objectSlices[i] = lst.index.tolist()

    for i in range(len(unionFindSet)):
        unionFindSet[i] = find_parent(unionFindSet[i], unionFindSet)
    # unionFindSet保存了当前bounding box与本层面内其他bounding box的归属关系
    # 对于bounding box i，如果bboxInfo.loc[i]["unionFindSet"] != i，
    # 那么表明它与bboxInfo.loc[i]["unionFindSet"]表示同一个结节
    bboxInfo["unionFindSet"] = unionFindSet
    # 在不同层面间对结节进行匹配，并且获得结节列表
    objects = []
    BOXID_VALUE = 10000
    for curZ in sorted(objectSlices.keys()):
        # 枚举之前的所有结节，检查最后一个层面，如果与当前层面的instanceNumber差值在Z_THRESHOLD之内，那么作为备选结节加入lastBoxes
        lastBoxes = [{"objectID": k["id"],
                      "bndbox": get_bounding_box_nparray(bboxInfo.loc[k["objectList"][-1]]),
                      "class": bboxInfo.loc[k["objectList"][-1]]["class"],
                      "prob": bboxInfo.loc[k["objectList"][-1]]["prob"]}
                      for k in objects if 0 < curZ - bboxInfo.loc[k["objectList"][-1]]["instanceNumber"] <= \
                            Z_THRESHOLD[bboxInfo.loc[k["objectList"][-1]]["class"]]]
        # 枚举本层面所有bounding box，每个框的评分为类别评分*置信度概率
        curBoxes = [{"matched": False,
                     "boxID": k,
                     "model_idx": bboxInfo.loc[k]["model_idx"],
                     "bndbox": get_bounding_box_nparray(bboxInfo.loc[k]),
                     "class": bboxInfo.loc[k]["class"],
                     "prob": bboxInfo.loc[k]["prob"],
                     "score": object_cls_weights[bboxInfo.loc[k]["class"]] * bboxInfo.loc[k]["prob"]}
                     for k in objectSlices[curZ]]
        # 对于有多个box表示一个结节的，只选择其中一个
        # （选择bboxInfo.loc[i]["unionFindSet"] == i的那个，即等价类树的根节点）
        curBoxes_root = \
                    [{"boxID": k,
                     "bndbox": get_bounding_box_nparray(bboxInfo.loc[k]),
                     "class": bboxInfo.loc[k]["class"],
                     "prob": bboxInfo.loc[k]["prob"]}
                     for k in objectSlices[curZ] if bboxInfo.loc[k]["unionFindSet"] == k]
        # 首先找到每个等价类中包括的框，如果这些框来自于n>=OBJECT_FREQ_THRESH个模型，则保留该等价类否则视为假阳不保留．
        # 第二步，选取每个等价类中评分最高的框，"matched"记录该框是否与前层结节相匹配（最大权匹配），若没有则视为新结节插入。
        curBoxes_union = []
        for i in curBoxes_root:
            Boxes_union = [j for j in curBoxes if bboxInfo.loc[j["boxID"]]["unionFindSet"] == i["boxID"]]
            Boxes_union_model_idx = [j['model_idx'] for j in Boxes_union]
            if len(set(Boxes_union_model_idx)) >= OBJ_FREQ_THRESH:
                Boxes_union = sorted(Boxes_union, key=lambda  k: k["score"])
                curBoxes_union.append(Boxes_union[-1])

        # 根据保留后的等价类枚举本层面所有bounding box，每个框的评分为类别评分*置信度概率
        curBoxes_kept = []
        for i in curBoxes_union:
            curBoxes_kept += [{"matched": False,
                         "boxID": k,
                         "bndbox": get_bounding_box_nparray(bboxInfo.loc[k]),
                         "class": bboxInfo.loc[k]["class"],
                         "prob": bboxInfo.loc[k]["prob"],
                         "score": object_cls_weights[bboxInfo.loc[k]["class"]] * bboxInfo.loc[k]["prob"]}
                         for k in objectSlices[curZ] if bboxInfo.loc[k]["unionFindSet"] == bboxInfo.loc[i["boxID"]]["unionFindSet"]]

        #　如果在之前层面（Z_THRESHOLD以内）没有结节，将新的等价类插入作为新结节的开始
        if len(lastBoxes) == 0:
            for k in curBoxes_union:
                add_object(objects, k["boxID"])
            continue
        # 建立二分图
        g = nx.Graph()
        g.add_nodes_from([i["boxID"] + BOXID_VALUE for i in curBoxes_kept])
        g.add_nodes_from([i["objectID"] for i in lastBoxes])
        for i in lastBoxes:
            for j in curBoxes_kept:
                # 定义不同层面的3D中心点相对偏移（sim_metric），因为结节在３D上移动很少，只对中心点相对偏移在一定阈值内的两个结节做匹配
                evalScore = anchor_metric.center_deviation_sqrt(i["bndbox"], j["bndbox"])
                if evalScore < SCORE_THRESHOLD:
                    if i["class"] not in object_cls_weights or j["class"] not in object_cls_weights:
                        print "object class not found in objects_cls_weights"
                        raise KeyError
                    object_weight = object_cls_weights[i["class"]] * i["prob"]
                    box_weight = object_cls_weights[j["class"]] * j["prob"]
                    # we suppress 3d matching for misaligned boxes, only valid for matching lung objects
                    misalign_suppress = np.exp(-evalScore)
                    # 定义不同层面两个bounding box的分数作为边权
                    if i["class"] == j["class"]:
                        matchingScore = same_cls_boost * object_weight * box_weight * misalign_suppress
                    else:
                        matchingScore = object_weight * box_weight * misalign_suppress
                    g.add_weighted_edges_from([[i["objectID"], j["boxID"] + BOXID_VALUE, matchingScore]])
        # 求出最大权匹配,networkx2.0之后nx.algorithms.max_weight_matching返回set,之前版本都是字典（本代码默认用较低版本运行）
        matchRes = nx.algorithms.max_weight_matching(g)
        matched_object_list = []
        reduced_matched_object_list = []
        for i in matchRes.keys():
            if i < BOXID_VALUE:
                matched_object_list.append(i)
                reduced_matched_object_list.append(i)

        if len(matched_object_list) == 0:
            for i in curBoxes_union:
                # 对于没有匹配上的bounding box，认为是一个新结节的开始
                if i["matched"] == False:
                    add_object(objects, i["boxID"])
            continue

        elif len(matched_object_list) == 1:
            box_union = bboxInfo.loc[matchRes[matched_object_list[0]] - BOXID_VALUE]["unionFindSet"]
            box_union_index = [i for i in range(len(curBoxes_union)) if
                               bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"] == box_union]
            if len(box_union_index) != 1:
                print ('there should be one and only one box for the same equivalent class in curBoxes_union')
                raise IndexError
            curBoxes_union[box_union_index[0]]["matched"] = True
        # 检查最大权匹配出的框有没有在一个等价类中，如果有，舍弃边权较低的那条边
        else:
            for j in range(len(matched_object_list)):
                if j < len(matched_object_list) - 1:
                    for k in range(len(matched_object_list)-j-1):
                        # 两个匹配了的框的等价类ID
                        box_union1 = bboxInfo.loc[matchRes[matched_object_list[j]]-BOXID_VALUE]["unionFindSet"]
                        box_union2 = bboxInfo.loc[matchRes[matched_object_list[j+k+1]]-BOXID_VALUE]["unionFindSet"]

                        box_union_index1 = [i for i in range(len(curBoxes_union)) if bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"]== box_union1]
                        box_union_index2 = [i for i in range(len(curBoxes_union)) if bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"]== box_union2]

                        if len(box_union_index1) != 1 or len(box_union_index2) != 1:
                            print ('there should be one and only one box for the same equivalent class in curBoxes_union')
                            raise  IndexError
                        curBoxes_union[box_union_index1[0]]["matched"] = True
                        curBoxes_union[box_union_index2[0]]["matched"] = True
                        if box_union1 == box_union2:
                            if g[matched_object_list[j]][matchRes[matched_object_list[j]]]['weight'] <= \
                                    g[matched_object_list[j+k+1]][matchRes[matched_object_list[j+k+1]]]['weight']:
                                if matched_object_list[j] in reduced_matched_object_list:
                                    reduced_matched_object_list.remove(matched_object_list[j])
                            elif matched_object_list[j+k+1] in reduced_matched_object_list:
                                reduced_matched_object_list.remove(matched_object_list[j+k+1])

        # 对于已经匹配上的bounding box，加入对应的objectList中
        for i in reduced_matched_object_list:

            objects[i - 1]["objectList"].append(matchRes[i] - BOXID_VALUE)

        for i in curBoxes_union:
            # 对于没有匹配上的bounding box，认为是一个新结节的开始
            if i["matched"] == False:
                add_object(objects, i["boxID"])

    object_result = [-1] * len(bboxInfo)
    for i in objects:
        for j in i["objectList"]:
            object_result[j] = i["id"]

    bboxInfo["object"] = object_result
    return bboxInfo[["instanceNumber", "xmin", "ymin", "xmax", "ymax", "class", "prob", "object"]], objects
