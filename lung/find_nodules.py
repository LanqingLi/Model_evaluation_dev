# encoding: utf-8
import pandas as pd
import numpy as np
import networkx as nx


def overlap1D(x, y):
    """
    Returns the overlap of 1d segment, returns [0, 0] if not overlapped.
    :params x: 1d np array of 2 elements. [st, ed]
    :params y: 1d np array of 2 elements. [st ,ed]
    """
    st = np.max([x[0], y[0]])
    ed = np.min([x[1], y[1]])
    return np.array([st, ed])


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


def calcDICE(box1, box2):
    box1 = box1.copy().reshape([2, 2])
    box2 = box2.copy().reshape([2, 2])
    if np.any(box1[1] - box1[0] < 0) or np.any(box2[1] - box2[0] < 0):
        raise ValueError(
            "Boxes should be represented as [xmin, ymin, xmax, ymax]. Box1: %s. Box2: %s. " % (str(box1), str(box2)))
    box1[1] += 1
    box2[1] += 1
    res = overlapND(box1, box2)
    if np.any(res[1] - res[0] <= 0):
        return 0.0
    return 2 * float(np.prod(res[1] - res[0])) / (np.prod(box1[1] - box1[0]) + np.prod(box2[1] - box2[0]))


def get_bounding_box_nparray(bbox):
    return np.array([[bbox["xmin"], bbox["ymin"]], [bbox["xmax"], bbox["ymax"]]])


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


def add_nodule(nodules, cur_nodule):
    # 增加一个新结节
    nodules.append({"noduleList": [cur_nodule]})
    nodules[-1]["id"] = len(nodules)


def find_nodules(bboxInfo, Z_THRESHOLD=3, SAME_BOX_THRESHOLD=0.7, SCORE_THRESHOLD=0.6):
    """将boundingbox转换为结节。
    :param bboxInfo: Bounding Box表格（pandas.DataFrame类型）。
                     每一行代表一个预测出来的bounding box，包含下面的列：
                     * 'instanceNumber'列：表示当前bounding box所在的层面的InstanceNumber，编号从1开始。
                     * 'xmin'列：表示当前bounding box的左上角x坐标（高度[H]方向）。
                     * 'ymin'列：表示当前bounding box的左上角y坐标（左右[W]方向）。
                     * 'xmax'列：表示当前bounding box的右下角x坐标（高度[H]方向）。
                     * 'ymax'列：表示当前bounding box的右下角y坐标（左右[W]方向）。
                     * 'nodule_class'列：表示当前bounding box的预测类别（如'nodule', 'mass'等）。
                     * 'prob'里：表示当前bounding box的预测概率。
    :return noduleInfo: 在bnd上附加一列'nodule'，取值为-1, 1..n。
                        -1代表当前Bounding box不属于任何一个nodule;
                        1..n代表当前bounding box所属的结节编号。
                        我们允许一个结节在同一个层面内存在多个重合的bounding box。
    """
    # 首先计算同一层面内dice coefficient比较高的， 并且认为这些bounding box标记了同一个结节
    bboxInfo = bboxInfo.copy()
    # 初始化并查集
    unionFindSet = bboxInfo.index.tolist()
    noduleSlices = {}
    for i in bboxInfo["instanceNumber"].unique():
        lst = bboxInfo.query("instanceNumber == @i")
        #print lst
        #print lst['prob'][lst.index[0]]
        if len(lst) > 1:
            for j1 in range(len(lst.index)):
                for j2 in range(j1 + 1, len(lst.index)):
                    if calcDICE(get_bounding_box_nparray(lst.iloc[j1]),
                                get_bounding_box_nparray(lst.iloc[j2])) > SAME_BOX_THRESHOLD:
                        # 如果两个box相似程度高于SAME_BOX_THRESHOLD，那么认为这两个box表示同一个结节,并保留概率更大的那个
                        # if lst['prob'][lst.index[j1]] < lst['prob'][lst.index[j2]]:
                        #     union(lst.index[j1], lst.index[j2], unionFindSet)
                        # else:
                        #     union(lst.index[j2], lst.index[j1], unionFindSet)
                        union(lst.index[j1], lst.index[j2], unionFindSet)
        noduleSlices[i] = lst.index.tolist()

    for i in range(len(unionFindSet)):
        unionFindSet[i] = find_parent(unionFindSet[i], unionFindSet)
    # unionFindSet保存了当前bounding box与本层面内其他bounding box的归属关系
    # 对于bounding box i，如果bboxInfo.loc[i]["unionFindSet"] != i，
    # 那么表明它与bboxInfo.loc[i]["unionFindSet"]表示同一个结节
    bboxInfo["unionFindSet"] = unionFindSet
    # 在不同层面间对结节进行匹配，并且获得结节列表
    nodules = []
    BOXID_VALUE = 10000
    for curZ in sorted(noduleSlices.keys()):
        # 枚举之前的所有结节，检查最后一个层面，如果与当前层面的instanceNumber差值在Z_THRESHOLD之内，那么作为备选结节加入lastBoxes
        lastBoxes = [{"noduleID": k["id"],
                      "bndbox": get_bounding_box_nparray(bboxInfo.loc[k["noduleList"][-1]])}
                     for k in nodules if 0 < curZ - bboxInfo.loc[k["noduleList"][-1]]["instanceNumber"] <= Z_THRESHOLD]
        # 枚举本层面所有bounding box，对于有多个box表示一个结节的，只选择其中一个
        # （选择bboxInfo.loc[i]["unionFindSet"] == i的那个）
        curBoxes = [{"boxID": k,
                     "bndbox": get_bounding_box_nparray(bboxInfo.loc[k]),
                     "nodule_class": bboxInfo.loc[k]["nodule_class"]}
                    for k in noduleSlices[curZ] if bboxInfo.loc[k]["unionFindSet"] == k]
        if len(lastBoxes) == 0:
            for k in curBoxes:
                add_nodule(nodules, k["boxID"])
            continue
        # 建立二分图
        g = nx.Graph()
        g.add_nodes_from([i["boxID"] + BOXID_VALUE for i in curBoxes])
        g.add_nodes_from([i["noduleID"] for i in lastBoxes])
        for i in lastBoxes:
            for j in curBoxes:
                # 使用两个bounding box的DICE分数作为边权
                evalScore = calcDICE(i["bndbox"], j["bndbox"])
                if evalScore > SCORE_THRESHOLD:
                    #  只有DICE超过SCORE_THRESHOLD的会被作为备选匹配加入二分图中
                    g.add_weighted_edges_from([[i["noduleID"], j["boxID"] + BOXID_VALUE, evalScore]])
        # 求出最大权匹配
        matchRes = nx.algorithms.max_weight_matching(g)
        unmatched = set([k for k in noduleSlices[curZ] if bboxInfo.loc[k]["unionFindSet"] == k])
        for i in matchRes.keys():
            if i < BOXID_VALUE:
                print curZ, i, matchRes[i]
                # 对于已经匹配上的bounding box，加入对应的noduleList中
                nodules[i - 1]["noduleList"].append(matchRes[i] - BOXID_VALUE)
                unmatched.remove(matchRes[i] - BOXID_VALUE)
        for i in unmatched:
            # 对于没有匹配上的bounding box，认为是一个新结节的开始
            add_nodule(nodules, i)

    nodule_result = [-1] * len(bboxInfo)
    for i in nodules:
        for j in i["noduleList"]:
            nodule_result[j] = i["id"]
    for i in bboxInfo.index:
        if bboxInfo.loc[i]["unionFindSet"] != i:
            nodule_result[i] = nodule_result[bboxInfo.loc[i]["unionFindSet"]]  # 如果需要把重复的box设成-1，把这里改成-1
    bboxInfo["nodule"] = nodule_result
    return bboxInfo[["instanceNumber", "xmin", "ymin", "xmax", "ymax", "nodule_class", "prob", "nodule"]], nodules