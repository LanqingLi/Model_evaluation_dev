# -- coding: utf-8 --
import numpy as np
import math
from model_eval.common.metric import EvalMetric, check_label_shapes
from sklearn import metrics

########################
# CLASSIFICATION METRICS
########################

class ClassificationMetric(EvalMetric):

    def __init__(self, name='ClsMetric', cls_num=1, allow_extra_outputs=False, if_binary=True, pos_cls_fusion=True):
        super(ClassificationMetric, self).__init__(name)
        self._allow_extra_outputs = allow_extra_outputs
        self.cls_num = cls_num
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.sum = 0.0
        self.label_pos = 0.0
        self.pred_pos = 0.0
        # keep track of the number of positive samples in gt and pred labels, for cerebrum segmentation, this is approximately
        # the intracerebral hemorrhage volume
        self.gt_vol = 0.0
        self.pred_vol = 0.0
        # whether it is binary classification
        self.if_binary = if_binary
        # In the context of binary classification, whether to regard all positive samples as the same class, or regard all classes (positive
        # and negative samples) but a single positive class as background (negative class). The former is usually adapted
        # for detection of positive samples (检出), the latter is usually used for detection of a specific class
        self.pos_cls_fusion = pos_cls_fusion

    def update(self, labels, preds):
        """
        update the metric with ground truth and predicted labels
        :param labels: [[0, 1, 2]] each item of the list is a list of ground truth class numbers for a single patient
        :param preds: [[1, 0, 1]] each item of the list is a list of predicted class numbers for a single patient

        :return:
        """
        check_label_shapes(labels, preds)

        for pred, label in zip(preds, labels):
            check_label_shapes(label, pred)
            pred_label = np.asarray(pred)
            label = np.asarray(label)
            check_label_shapes(label, pred_label)

            ind = self.cls_num
            if self.if_binary:
                # for binary classification, we either regard all labels that are not ind as negative (0) or regard all labels
                # that are not negative as positive (the same class)
                if self.pos_cls_fusion:
                    label[label > 0] = 1
                    pred_label[pred_label > 0] = 1
                    ind = 1
                else:
                    label[label != ind] = 0
                    pred_label[pred_label != ind] = 0


            pred_pos = (pred_label == ind)
            label_pos = (label == ind)
            pred_neg = (pred_label == 0)
            label_neg = (label == 0)

            self.label_pos += np.asscalar(np.sum(label_pos))
            self.pred_pos += np.asscalar(np.sum(pred_pos))
            tp = np.asscalar(np.sum(pred_pos * label_pos))
            tn = np.asscalar(np.sum(pred_neg * label_neg))

            self.tp += tp
            self.tn += tn
            self.fp += np.asscalar(np.sum(pred_pos) - tp)
            self.fn += np.asscalar(np.sum(label_pos) - tp)
            self.sum += np.asscalar(np.sum(pred_pos) + np.sum(label_pos))

    def reset(self):
        """Resets the internal evaluation result to initial state.  """
        self.tp = 0.0
        self.sum = 0.0
        self.fp=0.0
        self.fn = 0.0
        self.tn = 0.0
        self.label_pos = 0.0
        self.pred_pos = 0.0
        self.gt_vol = 0.0
        self.pred_vol = 0.0
        self.if_binary = True

    def get_acc(self):
        """Get the evaluated accuracy

        :return: accuracy = (tp + tn) / (tp + fp + tn + fn)
        """
        tot_sample = self.tp + self.fp + self.tn + self.fn
        if tot_sample > 0:
            return (self.tp + self.tn) / tot_sample
        else:
            print ZeroDivisionError
            return np.nan

    def get_rec(self):
        """Get the evaluated recall.

        :return: recall
        """

        if self.tp + self.fn > 0.:
            return self.tp / (self.tp + self.fn)
        else:
            print ZeroDivisionError
            return np.nan

    def get_prec(self):
        """Get the evaluated precision

        :return: precision
        """

        if self.tp + self.fp > 0.:
            return self.tp / (self.tp + self.fp)
        else:
            print ZeroDivisionError
            return np.nan

    def get_fscore(self, beta=1.):
        """Get the evaluated f score

        The F score is equivalent to weighted average of the precision and recall.
        where the best value is 1.0 and the worst value is 0.0. The formula for F score is:

        F_{beta} = (1 + \beta^2) * (precision * recall) / (\beta^2 * precision + recall)


        :param beta: float number accounts for the extent to which precision and recall are weighted differently, 1. by default
        :return: f score
        """
        if type(beta) is float:
            # if either precision or recall is nan, return nan
            if math.isnan(self.get_prec()) or math.isnan(self.get_rec()):
                return np.nan

            elif beta ** 2 * self.get_prec() + self.get_rec() > 0.:
                return (1 + beta ** 2) * (self.get_rec() * self.get_prec()) / (beta ** 2 * self.get_prec() + self.get_rec())

            # if both precision and recall are zero, return zero
            else:
                print ZeroDivisionError
                return 0.
        else:
            print ("fscore requires a float beta as input, not {}".format(beta))
            return ValueError

    def get_gt_vol(self):
        return self.label_pos

    def get_pred_vol(self):
        return self.pred_pos

def cls_avg(cls_weight, cls_value):
    assert len(cls_weight) == len(cls_value), 'weight and value list should contain the same number of classes'

    weight_sum = 0.
    value_sum = 0.

    # weight = np.asarray(cls_weight)
    # value = np.asarray(cls_value)
    for i in range(len(cls_weight)):
        # we ignore nan result (typically caused by ZeroDivisionError)
        if not math.isnan(cls_value[i]):
            weight_sum += cls_weight[i]
            value_sum += cls_value[i] * cls_weight[i]

    if weight_sum > 0.:
        return value_sum / weight_sum
    # if all cls_value are nan, return nan
    else:
        print ZeroDivisionError
        return np.nan

########################
# CLUSTERING METRICS
########################

class ClusteringMetric(EvalMetric):
    """The notion of clustering applies to problems of our concern such as nodule-matching. The algorithm (find_nodules)
    which groups 2D boxes embedded in slices of CT scans as 3D objects (nodules) is in its essence a clustering algorithm.
    Hence we developed the clustering metric for evaluation and testing of such tasks. One particular advantage of find_nodules
    testing is that we have the knowledge of the ground truth classes (each gt nodule is regarded as a class), which makes
    it possible for one to compute indexes such as adjusted rand index, adjusted mutual information score, homogeneity,
    completeness, etc. This metric is believed to be useful for evaluation of any future tasks that involves clustering
    and unsupervised learning.

    Relevant measures were adapted from http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    and https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """

    def __init__(self, name='ClusMetric', allow_extra_outputs=False):
        super(ClusteringMetric, self).__init__(name)
        self._allow_extra_outputs = allow_extra_outputs

        self.labels_gt = []
        self.labels_pred = []

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for pred, label in zip(preds, labels):
            check_label_shapes(pred, label)
            self.labels_gt.append(label)
            self.labels_pred.append(pred)

    def reset(self):
        self.labels_gt = []
        self.labels_pred = []

    def get_adjusted_rand_index(self):
        adjusted_rand_score = []
        for label, pred in zip(self.labels_gt, self.labels_pred):
            adjusted_rand_score.append(metrics.adjusted_rand_score(label, pred))
        return adjusted_rand_score

    def get_adjusted_mutual_info_score(self):
        adjusted_mutual_infor_score = []
        for label, pred in zip(self.labels_gt, self.labels_pred):
            adjusted_mutual_infor_score.append(metrics.adjusted_mutual_info_score(label, pred))
        return adjusted_mutual_infor_score

    def get_normalized_mutual_info_score(self):
        normalized_mutual_infor_score = []
        for label, pred in zip(self.labels_gt, self.labels_pred):
            normalized_mutual_infor_score.append(metrics.normalized_mutual_info_score(label, pred))
        return normalized_mutual_infor_score

    def get_homogeneity_completeness_v_measure(self):
        homogeneity_completeness_v_measure = []
        for label, pred in zip(self.labels_gt, self.labels_pred):
            homogeneity_completeness_v_measure.append(metrics.homogeneity_completeness_v_measure(label, pred))
        return homogeneity_completeness_v_measure

    def get_fowlkes_mallows_score(self):
        fowlkes_mallows_score = []
        for label, pred in zip(self.labels_gt, self.labels_pred):
            fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(label, pred))
        return fowlkes_mallows_score

    def get_silhouette_score(self):
        print NotImplementedError("silhouette_score is defined for situation when the ground truth labels are not known and the distance metric between samples are explicitly defined for clustering")
        silhouette_score = []
        for label, pred in zip(self.labels_gt, self.labels_pred):
            silhouette_score.append(np.nan)
        return silhouette_score


########################
# REGRESSION METRICS
########################

