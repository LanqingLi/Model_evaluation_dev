# encoding: utf-8
import pandas as pd
import numpy as np
import networkx as nx
# this script was written based on networkx version <= 2.0, and cannot be applied to networkx version later than 2.0
# one but not the only significant difference is that, networkx.algorithms.max_weight_matching <= 2.0 returns a dict,
# whereas networkx.algorithms.max_weight_matching > 2.0 returns a set


def overlap1D(x, y):
    """
    Returns the overlap of 1d segment, returns [0, 0] if not overlapped.
    :params x: 1d np array of 2 elements. [st, ed]
    :params y: 1d np array of 2 elements. [st ,ed]
    """
    lower_end = np.max([x[0], y[0]])
    higher_end = np.min([x[1], y[1]])
    if lower_end >= higher_end:
        return [0, 0]
    else:
        return np.array([lower_end, higher_end])

def overlapND(x, y):
    """
    Returns the overlap of n-d segment, returns [0, 0] in any dimension where x, y do not overlap
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
    """
    Returns the dice between two boxes, defined as 2* union(A, B) / (area(A) + area(B))
    :params x: [xmin, ymin, xmax, ymax] np array
    :params y: [xmin, ymin, xmax, ymax] np array
    """
    box1 = box1.copy().reshape([2, 2])
    box2 = box2.copy().reshape([2, 2])
    if np.any(box1[1] - box1[0] <= 0) or np.any(box2[1] - box2[0] <= 0):
        raise ValueError(
            "Boxes should be represented as [xmin, ymin, xmax, ymax]. Box1: %s. Box2: %s. " % (str(box1), str(box2)))
    res = overlapND(box1, box2)
    if np.any(res[1] - res[0] <= 0):
        return 0.0
    return 2 * float(np.prod(res[1] - res[0])) / (np.prod(box1[1] - box1[0]) + np.prod(box2[1] - box2[0]))

def get_center(box):
    box = (box.copy().reshape([2, 2])).astype('float32')
    return np.mean(box, axis=0)

def sim_metric_2d(box1, box2):
    """
    Returns the similarity metric between two boxes in a 2d ct slice, defined as the box center displacement divided by
    the greatest half-width of the two boxes in each dimension (x and y)
    :params x: [xmin, ymin, xmax, ymax] np array
    :params y: [xmin, ymin, xmax, ymax] np array
    """
    box1 = (box1.copy().reshape([2, 2])).astype('float32')
    box2 = (box2.copy().reshape([2, 2])).astype('float32')

    if np.any(box1[1] - box1[0] < 0) or np.any(box2[1] - box2[0] < 0):
        raise ValueError(
            "Boxes should be represented as [xmin, ymin, xmax, ymax]. Box1: %s. Box2: %s. " % (str(box1), str(box2)))
    size1 = (box1[1] - box1[0]) / 2
    size2 = (box2[1] - box2[0]) / 2
    center1 = get_center(box1)
    center2 = get_center(box2)
    return np.absolute(center1 - center2) / np.maximum(size1, size2)

def sim_metric_3d(box1, box2):
    """
    Returns the similarity metric between two boxes in different ct slices, defined as the Euclidean distance between
    the box centers divided by the greatest half-width of the two boxes in each dimension (x and y)
    :params x: [xmin, ymin, xmax, ymax] np array
    :params y: [xmin, ymin, xmax, ymax] np array
    """
    box1 = (box1.copy().reshape([2, 2])).astype('float32')
    box2 = (box2.copy().reshape([2, 2])).astype('float32')
    if np.any(box1[1] - box1[0] < 0) or np.any(box2[1] - box2[0] < 0):
        raise ValueError(
            "Boxes should be represented as [xmin, ymin, xmax, ymax]. Box1: %s. Box2: %s. " % (str(box1), str(box2)))
    size1 = (box1[1] - box1[0]) / 2
    size2 = (box2[1] - box2[0]) / 2
    center1 = get_center(box1)
    center2 = get_center(box2)
    rela_disp = (np.absolute(center1 - center2) / np.maximum(size1, size2)) ** 2
    return np.sqrt(np.sum(rela_disp))

def get_bounding_box_nparray(bbox):
    return np.array([[bbox["xmin"], bbox["ymin"]], [bbox["xmax"], bbox["ymax"]]])

def find_parent(id1, union_find_set):
    # 并查集用，查找功能
    # print id1
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

def find_nodules(bboxInfo, Z_THRESHOLD, SAME_BOX_THRESHOLD=np.array([1.6, 1.6]), SCORE_THRESHOLD=0.6,
                 nodule_cls_weights={}, same_cls_boost = 2.):

    """将boundingbox转换为结节。
    :param bboxInfo: Bounding Box表格（pandas.DataFrame类型）,index必须为自然顺序。
                     每一行代表一个预测出来的bounding box，包含下面的列：
                     * 'instanceNumber'列：表示当前bounding box所在的层面的InstanceNumber，编号从1开始。
                     * 'xmin'列：表示当前bounding box的左上角x坐标（高度[H]方向）。
                     * 'ymin'列：表示当前bounding box的左上角y坐标（左右[W]方向）。
                     * 'xmax'列：表示当前bounding box的右下角x坐标（高度[H]方向）。
                     * 'ymax'列：表示当前bounding box的右下角y坐标（左右[W]方向）。
                     * 'nodule_class'列：表示当前bounding box的预测类别（如'nodule', 'mass'等）。
                     * 'prob'里：表示当前bounding box的预测概率。
    :param Z_THRESHOLD： 每个新层面向前做贪心匹配时往前找的最大层面数
    :param SAME_BOX_THRESHOLD: 判断同一层面两个框是否为等价类的中心点偏移阈值
    :param SCORE_THRESHOLD: 判断不同层面两个框是否匹配的sim_metric_3d阈值
    :param nodule_cls_weights:　检出框的类别的权重
    :param same_cls_boost: 当不同层面两个框匹配时，如果类别相同的奖励系数
    :return noduleInfo: 在bnd上附加一列'nodule'，取值为-1, 1..n。
                        -1代表当前Bounding box不属于任何一个nodule;
                        1..n代表当前bounding box所属的结节编号。
                        我们不允许一个结节在同一个层面内存在多个重合的bounding box(一个等价类中没有匹配上的框设为-1)。
    ":return nodules: 结节信息的列表，每个元素为一个字典
    """
    # 首先计算同一层面内dice coefficient比较高的， 并且认为这些bounding box标记了同一个结节
    bboxInfo = bboxInfo.copy()
    # 初始化并查集
    unionFindSet = bboxInfo.index.tolist()
    noduleSlices = {}
    for i in bboxInfo["instanceNumber"].unique():
        lst = bboxInfo.query("instanceNumber == @i")
        if len(lst) > 1:
            for j1 in range(len(lst.index)):
                for j2 in range(j1 + 1, len(lst.index)):
                    # 如果两个box中心点相对位移小于SAME_BOX_THRESHOLD，那么认为这两个box表示同一个结节
                    iou = calcDICE(get_bounding_box_nparray(lst.iloc[j1]), get_bounding_box_nparray(lst.iloc[j2]))
                    if iou > 0 and np.all(sim_metric_2d(get_bounding_box_nparray(lst.iloc[j1]),
                                  get_bounding_box_nparray(lst.iloc[j2]))/iou < SAME_BOX_THRESHOLD):
                        # 将两个box插入一颗等价类树
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
                      "bndbox": get_bounding_box_nparray(bboxInfo.loc[k["noduleList"][-1]]),
                      "nodule_class": bboxInfo.loc[k["noduleList"][-1]]["nodule_class"],
                      "prob": bboxInfo.loc[k["noduleList"][-1]]["prob"]}
                      for k in nodules if 0 < curZ - bboxInfo.loc[k["noduleList"][-1]]["instanceNumber"] <= \
                            Z_THRESHOLD[bboxInfo.loc[k["noduleList"][-1]]["nodule_class"]]]
        # 枚举本层面所有bounding box，每个框的评分为类别评分*置信度概率
        curBoxes = [{"matched": False,
                     "boxID": k,
                     "bndbox": get_bounding_box_nparray(bboxInfo.loc[k]),
                     "nodule_class": bboxInfo.loc[k]["nodule_class"],
                     "prob": bboxInfo.loc[k]["prob"],
                     "score": nodule_cls_weights[bboxInfo.loc[k]["nodule_class"]] * bboxInfo.loc[k]["prob"]}
                     for k in noduleSlices[curZ]]
        # 对于有多个box表示一个结节的，只选择其中一个
        # （选择bboxInfo.loc[i]["unionFindSet"] == i的那个，即等价类树的根节点）
        curBoxes_root = \
                    [{"boxID": k,
                     "bndbox": get_bounding_box_nparray(bboxInfo.loc[k]),
                     "nodule_class": bboxInfo.loc[k]["nodule_class"],
                     "prob": bboxInfo.loc[k]["prob"]}
                     for k in noduleSlices[curZ] if bboxInfo.loc[k]["unionFindSet"] == k]
        # 选取每个等价类中评分最高的框，"matched"记录该框是否与前层结节相匹配（最大权匹配），若没有则视为新结节插入。
        curBoxes_union = []
        for i in curBoxes_root:
            Boxes_union = [j for j in curBoxes if bboxInfo.loc[j["boxID"]]["unionFindSet"] == i["boxID"]]
            Boxes_union = sorted(Boxes_union, key=lambda  k: k["score"])
            curBoxes_union.append(Boxes_union[-1])
        #　如果在之前层面（Z_THRESHOLD以内）没有结节，将新的等价类插入作为新结节的开始
        if len(lastBoxes) == 0:
            for k in curBoxes_union:
                add_nodule(nodules, k["boxID"])
            continue
        # 建立二分图
        g = nx.Graph()
        g.add_nodes_from([i["boxID"] + BOXID_VALUE for i in curBoxes])
        g.add_nodes_from([i["noduleID"] for i in lastBoxes])
        for i in lastBoxes:
            for j in curBoxes:
                # 定义不同层面的3D中心点相对偏移（sim_metric），因为结节在３D上移动很少，只对中心点相对偏移在一定阈值内的两个结节做匹配
                evalScore = sim_metric_3d(i["bndbox"], j["bndbox"])
                if evalScore < SCORE_THRESHOLD:
                    if i["nodule_class"] not in nodule_cls_weights or j["nodule_class"] not in nodule_cls_weights:
                        print "nodule class not found in nodules_cls_weights"
                        raise KeyError
                    nodule_weight = nodule_cls_weights[i["nodule_class"]] * i["prob"]
                    box_weight = nodule_cls_weights[j["nodule_class"]] * j["prob"]
                    # we suppress 3d matching for misaligned boxes, only valid for matching lung nodules
                    misalign_suppress = np.exp(-evalScore)
                    # 定义不同层面两个bounding box的分数作为边权
                    if i["nodule_class"] == j["nodule_class"]:
                        matchingScore = same_cls_boost * nodule_weight * box_weight * misalign_suppress
                    else:
                        matchingScore = nodule_weight * box_weight * misalign_suppress
                    g.add_weighted_edges_from([[i["noduleID"], j["boxID"] + BOXID_VALUE, matchingScore]])
        # 求出最大权匹配,networkx2.0之后nx.algorithms.max_weight_matching返回set,之前版本都是字典（本代码默认用较低版本运行）
        matchRes = nx.algorithms.max_weight_matching(g)
        matched_nodule_list = []
        reduced_matched_nodule_list = []
        for i in matchRes.keys():
            if i < BOXID_VALUE:
                matched_nodule_list.append(i)
                reduced_matched_nodule_list.append(i)

        if len(matched_nodule_list) == 0:
            for i in curBoxes_union:
                # 对于没有匹配上的bounding box，认为是一个新结节的开始
                if i["matched"] == False:
                    add_nodule(nodules, i["boxID"])
            continue

        elif len(matched_nodule_list) == 1:
            box_union = bboxInfo.loc[matchRes[matched_nodule_list[0]] - BOXID_VALUE]["unionFindSet"]
            box_union_index = [i for i in range(len(curBoxes_union)) if
                               bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"] == box_union]
            if len(box_union_index) != 1:
                print ('there should be one and only one box for the same equivalent class in curBoxes_union')
                raise IndexError
            curBoxes_union[box_union_index[0]]["matched"] = True
        # 检查最大权匹配出的框有没有在一个等价类中，如果有，舍弃边权较低的那条边
        else:
            for j in range(len(matched_nodule_list)):
                if j < len(matched_nodule_list) - 1:
                    for k in range(len(matched_nodule_list)-j-1):
                        # 两个匹配了的框的等价类ID
                        box_union1 = bboxInfo.loc[matchRes[matched_nodule_list[j]]-BOXID_VALUE]["unionFindSet"]
                        box_union2 = bboxInfo.loc[matchRes[matched_nodule_list[j+k+1]]-BOXID_VALUE]["unionFindSet"]

                        box_union_index1 = [i for i in range(len(curBoxes_union)) if bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"]== box_union1]
                        box_union_index2 = [i for i in range(len(curBoxes_union)) if bboxInfo.loc[curBoxes_union[i]["boxID"]]["unionFindSet"]== box_union2]

                        if len(box_union_index1) != 1 or len(box_union_index2) != 1:
                            print ('there should be one and only one box for the same equivalent class in curBoxes_union')
                            raise  IndexError
                        curBoxes_union[box_union_index1[0]]["matched"] = True
                        curBoxes_union[box_union_index2[0]]["matched"] = True
                        if box_union1 == box_union2:
                            if g[matched_nodule_list[j]][matchRes[matched_nodule_list[j]]]['weight'] <= \
                                    g[matched_nodule_list[j+k+1]][matchRes[matched_nodule_list[j+k+1]]]['weight']:
                                if matched_nodule_list[j] in reduced_matched_nodule_list:
                                    reduced_matched_nodule_list.remove(matched_nodule_list[j])
                            elif matched_nodule_list[j+k+1] in reduced_matched_nodule_list:
                                reduced_matched_nodule_list.remove(matched_nodule_list[j+k+1])

        # 对于已经匹配上的bounding box，加入对应的noduleList中
        for i in reduced_matched_nodule_list:

            nodules[i - 1]["noduleList"].append(matchRes[i] - BOXID_VALUE)

        for i in curBoxes_union:
            # 对于没有匹配上的bounding box，认为是一个新结节的开始
            if i["matched"] == False:
                add_nodule(nodules, i["boxID"])

    nodule_result = [-1] * len(bboxInfo)
    for i in nodules:
        for j in i["noduleList"]:
            nodule_result[j] = i["id"]

    bboxInfo["nodule"] = nodule_result
    return bboxInfo[["instanceNumber", "xmin", "ymin", "xmax", "ymax", "nodule_class", "prob", "nodule"]], nodules