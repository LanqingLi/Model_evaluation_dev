# encoding: utf-8
import pandas as pd
import numpy as np
import networkx as nx


def edge_weight_metric(i, j, nodule_cls_weights, same_cls_boost):
    '''
    :param i: bbox1
    :param j: bbox2
    :param nodule_cls_weights: classification weights, hyperparameter
    :param same_cls_boost: hyperparameter
    :return: weight of the edge between node i and node j. The more likely they belong to the same nodule, the greater
    the value should be.
    This is the most important metric of this file since the whole task is to summarize the edge weight in this graph.
    '''
    evalScore = sim_metric_3d(get_bounding_box_nparray(i), get_bounding_box_nparray(j))
    nodule_weight = nodule_cls_weights[i["nodule_class"]] * i["prob"]
    box_weight = nodule_cls_weights[j["nodule_class"]] * j["prob"]
    # we suppress 3d matching for misaligned boxes, only valid for matching lung nodules
    misalign_suppress = np.exp(-evalScore)
    if i["nodule_class"] == j["nodule_class"]:
        matchingScore = same_cls_boost * nodule_weight * box_weight * misalign_suppress
    else:
        matchingScore = nodule_weight * box_weight * misalign_suppress
    return matchingScore


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
    if np.any(box1[1] - box1[0] <= 0) or np.any(box2[1] - box2[0] <= 0):
        raise ValueError(
            "Boxes should be represented as [xmin, ymin, xmax, ymax]. Box1: %s. Box2: %s. " % (str(box1), str(box2)))
    # box1[1] += 1
    # box2[1] += 1
    res = overlapND(box1, box2)
    # return res
    if np.any(res[1] - res[0] <= 0):
        return 0.0
    return 2 * float(np.prod(res[1] - res[0])) / (np.prod(box1[1] - box1[0]) + np.prod(box2[1] - box2[0]))


def get_center(box):
    box = (box.copy().reshape([2, 2])).astype('float32')
    return np.mean(box, axis=0)


def sim_metric_2d(box1, box2):
    box1 = (box1.copy().reshape([2, 2])).astype('float32')
    box2 = (box2.copy().reshape([2, 2])).astype('float32')
    # print box1
    # print box2
    # print box1[1] - box1[0]
    # print box2[1] - box2[0]
    if np.any(box1[1] - box1[0] < 0) or np.any(box2[1] - box2[0] < 0):
        raise ValueError(
            "Boxes should be represented as [xmin, ymin, xmax, ymax]. Box1: %s. Box2: %s. " % (str(box1), str(box2)))
    size1 = (box1[1] - box1[0]) / 2
    size2 = (box2[1] - box2[0]) / 2
    center1 = get_center(box1)
    center2 = get_center(box2)
    # print np.absolute(center1 - center2), np.maximum(size1, size2)
    return np.absolute(center1 - center2) / np.maximum(size1, size2)


def sim_metric_3d(box1, box2):
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
        union_find_set[id1] = find_parent(union_find_set[id1], union_find_set)
        return union_find_set[id1]


def union(id1, id2, union_find_set, ufrank):
    # 并查集用，合并功能
    #if find_parent(id1, union_find_set) != find_parent(id2, union_find_set):
    x = find_parent(id1, union_find_set)
    y = find_parent(id2, union_find_set)
    if ufrank[x] > ufrank[y]:
        union_find_set[y] = x
    else:
        union_find_set[x] = y
        if ufrank[x] == ufrank[y]:
            ufrank[y] += 1


def add_nodule(nodules, cur_nodule):
    # 增加一个新结节
    nodules.append({"noduleList": [cur_nodule]})
    nodules[-1]["id"] = len(nodules)


def calc_sum_edges(g, dict):
    '''
    :param g: graph
    :param dict: dictionary returned by networkx.max_weight_matching
    :return: total weight sum of edges in networkx.max_weight_matching
    '''
    res = 0.0
    for p, q in dict.iteritems():
        res += g[p][q]['weight']
    res = res / 2.0
    return res


def find_nodules(bboxInfo, Z_THRESHOLD=3., SAME_BOX_THRESHOLD=np.array([1., 1.]), SCORE_THRESHOLD=1.,
                 nodule_cls_weights={
                          'solid nodule': 1.,
                          'calcific nodule': 1.,
                          'GGN': 1.,
                          '0-3nodule': 1.,
                          'mass': 1.,
                          'not_mass': 0,
                          }, same_cls_boost = 2.):

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
    :return noduleInfo: 在bnd上附加一列'nodule'，取值为-1, 1..n。
                        -1代表当前Bounding box不属于任何一个nodule;
                        1..n代表当前bounding box所属的结节编号。
                        我们不允许一个结节在同一个层面内存在多个重合的bounding box(一个等价类中没有匹配上的框设为-1)。
    """
    # 首先计算同一层面内dice coefficient比较高的， 并且认为这些bounding box标记了同一个结节
    bboxInfo = bboxInfo.copy()
    bboxInfo = bboxInfo.sort_index()
    # 初始化并查集
    unionFindSet = bboxInfo.index.tolist()
    bbox_num = len(unionFindSet)
    ufrank = [0] * bbox_num
    noduleSlices = {}
    for i in bboxInfo["instanceNumber"].unique():
        lst = bboxInfo.query("instanceNumber == @i")
        if len(lst) > 1:
            for j1 in range(len(lst.index)):
                for j2 in range(j1 + 1, len(lst.index)):
                    # 如果两个box中心点相对位移小于SAME_BOX_THRESHOLD，那么认为这两个box表示同一个结节
                    if np.all(sim_metric_2d(get_bounding_box_nparray(lst.iloc[j1]),
                                            get_bounding_box_nparray(lst.iloc[j2])) <= SAME_BOX_THRESHOLD):
                        # 将两个box插入一颗等价类树
                        union(lst.index[j1], lst.index[j2], unionFindSet, ufrank)
        noduleSlices[i] = lst.index.tolist()
    # unionFindSet保存了当前bounding box与本层面内其他bounding box的归属关系
    # 对于bounding box i，如果bboxInfo.loc[i]["unionFindSet"] != i，
    # 那么表明它与bboxInfo.loc[i]["unionFindSet"]表示同一个结节
    # equivalent_cls contains all boxes that belong to the same equivalent class
    equivalent_cls = {}
    for i in range(len(unionFindSet)):
        unionFindSet[i] = find_parent(unionFindSet[i], unionFindSet)
        if unionFindSet[i] in equivalent_cls:
            equivalent_cls[unionFindSet[i]].append(i)
        else:
            equivalent_cls[unionFindSet[i]] = [i]
    bboxInfo["unionFindSet"] = unionFindSet
    # Create a graph with root of each unionfind set as two nodes, one sending edges to boxes below it, one receiving
    # edges from boxes above it.
    g = nx.Graph()
    incoming_value = 10000
    # These nodes only have outcoming edges, i.e. they only make edges with boxes with bigger layer number
    g.add_nodes_from([k for k in equivalent_cls])
    # These nodes only have incoming edges, i.e. they only make edges with boxes with smaller layer number
    g.add_nodes_from([k + incoming_value for k in equivalent_cls])
    for root1_index1 in equivalent_cls:
        for root2_index1 in equivalent_cls:
            if root2_index1 < root1_index1:
                continue
            root1_index = root1_index1
            root2_index = root2_index1
            root1 = bboxInfo.loc[root1_index]
            root2 = bboxInfo.loc[root2_index]
            if root2["instanceNumber"] < root1["instanceNumber"]:
                root1, root2 = root2, root1
                root1_index, root2_index = root2_index, root1_index
            # If two equivalent classes are within Z_THRESHOLD of layers, try to find if there can be an edge.
            if root1["instanceNumber"] != root2["instanceNumber"] and abs(root1["instanceNumber"] - root2["instanceNumber"]) <= Z_THRESHOLD:
                max_score = -1
                node1 = None
                node2 = None
                # Try every two possible boxes to make edge.
                for box1_index in equivalent_cls[root1_index]:
                    for box2_index in equivalent_cls[root2_index]:
                        box1 = bboxInfo.loc[box1_index]
                        box2 = bboxInfo.loc[box2_index]
                        evalScore = sim_metric_3d(get_bounding_box_nparray(box1), get_bounding_box_nparray(box2))
                        # Try to add edge only if two boxes are close enough spatially.
                        if evalScore < SCORE_THRESHOLD:
                            matchingScore = edge_weight_metric(box1, box2, nodule_cls_weights, same_cls_boost)
                            if matchingScore > max_score:
                                max_score = matchingScore
                                node1 = box1_index
                                node2 = box2_index
                # If there can be an edge, add an edge
                if max_score > -1:
                    g.add_weighted_edges_from([[root1_index, root2_index + incoming_value, max_score]])
                    g[root1_index][root2_index + incoming_value]['couple'] = [node1, node2]
    # Graph construction is completed.
    # Calculate max matching.
    matchRes = nx.algorithms.max_weight_matching(g)
    sum_edges = calc_sum_edges(g, matchRes)
    # Record if a box has been selected, 0 means not selected, 1 means selected as upper box in a matching, 2 means
    # selected as lower box in a matching, 3 means selected as both.
    box_status = [0] * bbox_num
    # Use unionfind to find nodules.
    # mildly_fuckedup counts the number of equivalent classes that match different boxes.
    # wildly_fuckedup counts the number of equivalent classes that match two boxes of different label.
    # topdown_count counts the number of equivalent classes that are matched both up and down.
    topdown_count = 0
    mildly_fuckedup = 0
    wildly_fuckedup = 0
    equi_nodule = bboxInfo.index.tolist()
    uf2rank = [0] * bbox_num
    right_box = {}
    for p, q in matchRes.iteritems():
        if p > q:
            continue
        q = q - incoming_value
        union(p, q, equi_nodule, uf2rank)
        upper, lower = g[p][q + incoming_value]['couple']
        box_status[upper] += 1
        if p in right_box:
            topdown_count += 1
        if p in right_box and upper != right_box[p][0]:
            right_box[p].append(upper)
            mildly_fuckedup += 1
            if bboxInfo.loc[upper]['nodule_class'] != bboxInfo.loc[right_box[p][0]]['nodule_class']:
                wildly_fuckedup += 1
        else:
            right_box[p] = [upper]
        box_status[lower] += 2
        if q in right_box:
            topdown_count += 1
        if q in right_box and lower != right_box[q][0]:
            right_box[q].append(lower)
            mildly_fuckedup += 1
            if bboxInfo.loc[lower]['nodule_class'] != bboxInfo.loc[right_box[q][0]]['nodule_class']:
                wildly_fuckedup += 1
        else:
            right_box[q] = [lower]

    nodule_num = 0
    nodule_dict = {}
    nodule_list = []
    nodule = [-1] * bbox_num
    # Generate nodule list.
    for root in equivalent_cls:
        if not root in right_box:
            max_confidence = 0
            for box_index in equivalent_cls[root]:
                prob_here = bboxInfo.loc[box_index]['prob']
                if prob_here > max_confidence:
                    max_confidence = prob_here
                    right_box[root] = [box_index]
        if len(right_box[root]) > 1:
            print bboxInfo.loc[root]['instanceNumber'], right_box[root]
        ulti_root = find_parent(root, equi_nodule)
        if ulti_root not in nodule_dict:
            nodule_list.append({})
            nodule_list[nodule_num]['id'] = nodule_num + 1
            nodule_list[nodule_num]['noduleList'] = []
            nodule_dict[ulti_root] = nodule_num
            nodule_index = nodule_num
            nodule_num += 1
        else:
            nodule_index = nodule_dict[ulti_root]
        if len(right_box[root]) > 2:
            raise ValueError('Root', root, 'on layer', bboxInfo.loc[root]['instanceNumber'], 'has',
                             len(right_box[root]), 'boxes in max matching. There should be less than three')
        nodule[right_box[root][0]] = nodule_index + 1
        nodule_list[nodule_index]['noduleList'].append(right_box[root][0])
    bboxInfo['nodule'] = nodule
    return bboxInfo[["instanceNumber", "xmin", "ymin", "xmax", "ymax", "nodule_class", "prob", "nodule"]], nodule_list
