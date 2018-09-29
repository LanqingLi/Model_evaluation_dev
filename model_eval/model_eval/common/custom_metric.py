# -- coding: utf-8 --
import numpy as np
import math
import SimpleITK as sitk
from metric import EvalMetric, check_label_shapes
from sklearn import metrics


########################
# CLASSIFICATION METRICS
########################

class ClassificationMetric(EvalMetric):

    def __init__(self, name='ClsMetric', cls_num=1, allow_extra_outputs=False, if_binary=True, pos_cls_fusion=True):
        # number of positive classes for the metric to account for
        self._cls_num = cls_num
        super(ClassificationMetric, self).__init__(name)
        self._allow_extra_outputs = allow_extra_outputs

        assert self._cls_num >= 1 and type(self._cls_num) == int, "the number of positive classes must be a positive integer!"
        self.tp = [0.0 for _ in range(self._cls_num)]
        self.fp = [0.0 for _ in range(self._cls_num)]
        self.fn = [0.0 for _ in range(self._cls_num)]
        self.tn = [0.0 for _ in range(self._cls_num)]
        self.sum = [0.0 for _ in range(self._cls_num)]
        self.label_pos = [0.0 for _ in range(self._cls_num)]
        self.pred_pos = [0.0 for _ in range(self._cls_num)]
        # keep track of the number of positive samples in gt and pred labels, for cerebrum segmentation, this is approximately
        # the intracerebral hemorrhage volume
        self.gt_vol = [0.0 for _ in range(self._cls_num)]
        self.pred_vol = [0.0 for _ in range(self._cls_num)]
        # whether it is binary classification
        self.if_binary = if_binary
        # In the context of binary classification, whether to regard all positive samples as the same class, or regard all classes (positive
        # and negative samples) but a single positive class as background (negative class). The former is usually adapted
        # for detection of positive samples (检出), the latter is usually used for detection of a specific class
        self.pos_cls_fusion = pos_cls_fusion
        self.cls_label = 0

    @property
    def get_cls_num(self):
        '''
        :return: The number of positive classes in this metric
        '''
        return self._cls_num

    def update(self, labels, preds, cls_label=1):
        """
        update the metric with ground truth and predicted labels
        :param labels: [[0, 1, 2]] each item of the list is a list of ground truth class numbers for a single patient
        :param preds: [[1, 0, 1]] each item of the list is a list of predicted class numbers for a single patient

        :return:
        """
        check_label_shapes(labels, preds)
        # the current postive class label to account for, if self.cls_label = 0 (default value), do nothing
        self.cls_label = cls_label
        assert self.cls_label >= 1 and type(self.cls_label) == int, "cls_label must be a positive integer!"

        for pred, label in zip(preds, labels):
            check_label_shapes(label, pred)
            pred_label = np.asarray(pred)
            label = np.asarray(label)
            check_label_shapes(label, pred_label)

            ind = self.cls_label
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

            self.label_pos[self.cls_label - 1] += np.asscalar(np.sum(label_pos))
            self.pred_pos[self.cls_label - 1] += np.asscalar(np.sum(pred_pos))
            tp = np.asscalar(np.sum(pred_pos * label_pos))
            tn = np.asscalar(np.sum(pred_neg * label_neg))

            self.tp[self.cls_label - 1] += tp
            self.tn[self.cls_label - 1] += tn
            self.fp[self.cls_label - 1] += np.asscalar(np.sum(pred_pos) - tp)
            self.fn[self.cls_label - 1] += np.asscalar(np.sum(label_pos) - tp)
            self.sum[self.cls_label - 1] += np.asscalar(np.sum(pred_pos) + np.sum(label_pos))

    def reset(self):
        """Resets the internal evaluation result to initial state.  """
        self.tp = [0.0 for _ in range(self._cls_num)]
        self.fp = [0.0 for _ in range(self._cls_num)]
        self.fn = [0.0 for _ in range(self._cls_num)]
        self.tn = [0.0 for _ in range(self._cls_num)]
        self.sum = [0.0 for _ in range(self._cls_num)]
        self.label_pos = [0.0 for _ in range(self._cls_num)]
        self.pred_pos = [0.0 for _ in range(self._cls_num)]
        self.gt_vol = [0.0 for _ in range(self._cls_num)]
        self.pred_vol = [0.0 for _ in range(self._cls_num)]
        self.if_binary = True

    def get_acc(self, cls_label):
        """Get the evaluated accuracy

        :return: accuracy = (tp + tn) / (tp + fp + tn + fn)
        """
        assert cls_label >= 1 and type(cls_label) == int, "cls_label must be a positive integer!"
        tot_sample = self.tp[cls_label-1] + self.fp[cls_label-1] + self.tn[cls_label-1] + self.fn[cls_label-1]
        if tot_sample > 0:
            return (self.tp[cls_label-1] + self.tn[cls_label-1]) / tot_sample
        else:
            print ZeroDivisionError
            return np.nan

    def get_rec(self, cls_label):
        """Get the evaluated recall.

        :return: recall
        """
        assert cls_label >= 1 and type(cls_label) == int, "cls_label must be a positive integer!"
        if self.tp[cls_label-1] + self.fn[cls_label-1] > 0.:
            return self.tp[cls_label-1] / (self.tp[cls_label-1] + self.fn[cls_label-1])
        else:
            print ZeroDivisionError
            return np.nan

    def get_prec(self, cls_label):
        """Get the evaluated precision

        :return: precision
        """
        assert cls_label >= 1 and type(cls_label) == int, "cls_label must be a positive integer!"
        if self.tp[cls_label-1] + self.fp[cls_label-1] > 0.:
            return self.tp[cls_label-1] / (self.tp[cls_label-1] + self.fp[cls_label-1])
        else:
            print ZeroDivisionError
            return np.nan

    def get_dice(self, cls_label):
        """Get the evaluated dice (2*tp/(2*tp + fp + fn))

        :param cls_label:
        :return: dice
        """
        assert cls_label >= 1 and type(cls_label) == int, "cls_label must be a positive integer!"
        if (2*self.tp[cls_label-1] + self.fp[cls_label-1] + self.fn[cls_label-1]) > 0.:
            return 2*self.tp[cls_label-1] / (2*self.tp[cls_label-1] + self.fp[cls_label-1] + self.fn[cls_label-1])
        else:
            print ZeroDivisionError
            return np.nan


    def get_fscore(self, cls_label, beta=1.):
        """Get the evaluated f score

        The F score is equivalent to weighted average of the precision and recall. The greater the beta, the more significant
        the recall. The best value is 1.0 and the worst value is 0.0. The formula for F score is:

        F_{beta} = (1 + \beta^2) * (precision * recall) / (\beta^2 * precision + recall)


        :param beta: float number accounts for the extent to which precision and recall are weighted differently, 1. by default
        :return: f score
        """
        if type(beta) is float:
            # if either precision or recall is nan, return nan
            if math.isnan(self.get_prec(cls_label)) or math.isnan(self.get_rec(cls_label)):
                return np.nan

            elif beta ** 2 * self.get_prec(cls_label) + self.get_rec(cls_label) > 0.:
                return (1 + beta ** 2) * (self.get_rec(cls_label) * self.get_prec(cls_label)) / (beta ** 2 * self.get_prec(cls_label) + self.get_rec(cls_label))

            # if both precision and recall are zero, return zero
            else:
                print ZeroDivisionError
                return 0.
        else:
            print ("fscore requires a float beta as input, not {}".format(beta))
            return ValueError

    def get_gt_vol(self, cls_label):
        return self.label_pos[cls_label-1]

    def get_pred_vol(self, cls_label):
        return self.pred_pos[cls_label-1]

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
# CUSTOMIZED SCORE
########################

# coronary artery calcium score
class CAC_Score(EvalMetric):
    '''


    '''
    def __init__(self, name='CACScore', allow_extra_outputs=False):
        super(CAC_Score, self).__init__(name)
        self._allow_extra_outputs = allow_extra_outputs

    def default_calcium_cate(hu_value):
        assert hu_value >= 130, 'hu_value must be greater than 130'
        if hu_value >= 400:
            return 4
        elif hu_value >= 300:
            return 3
        elif hu_value >= 200:
            return 2
        elif hu_value >= 130:
            return 1

    def default_risk_cate(self, score):
        if not type(score) == float or score < 0:
            return TypeError('input argument %s of type %s must be a positive float!' %(score, type(score)))
        elif score == 0:
            return 'Zero'
        elif score < 100:
            return 'Mild'
        elif score < 300:
            return 'Moderate'
        else:
            return 'Severe'

    def check_binary_mask(self, binary_mask):
        num_zero = np.sum(binary_mask == 0)
        num_one = np.sum(binary_mask == 1)
        return num_one + num_zero == np.prod(binary_mask.shape)

    def get_CAC_score(self, image, binary_mask, voxel_volume, calcium_cate=default_calcium_cate, thresh=130, dim=2):
        '''
        :param image: raw image data, shape = [H, W, D]
        :param binary_mask: binary mask for coronary artery calcium (CAC) region , shape = [H, W, D]
        :param calcium_cate: function for determining calcium category (score) for each calcified region
        :param voxel_volume: voxel volume computed from voxel spacing of the image array
        :param thresh: threshold for filtering calcium region
        :param dim: dimension param for calcium score integral (whether to integrate slice by slice or plaque by plaque)
        :return: CAC score for the raw image data given binary mask
        '''
        calcium_score = 0.
        assert self.check_binary_mask(binary_mask), 'binary_mask must only contain 0 or 1'

        if dim == 3:
            masked_data = sitk.GetImageFromArray(binary_mask)
            img_data = sitk.GetImageFromArray(image)
            sitk_binary_mask = sitk.Cast(masked_data, sitk.sitkInt16)
            masked_img_array = img_data * sitk_binary_mask

            label = sitk.ConnectedComponent(masked_img_array>= thresh)
            stat = sitk.LabelIntensityStatisticsImageFilter()
            stat.Execute(label, img_data)

            for i in stat.GetLabels():
                size = stat.GetSum(i) / stat.GetMean(i) * voxel_volume
                print ("Calcified Plaque: {0} -> Mean: {1} Size: {2} Max: {3}".format(i, stat.GetMean(i), size, stat.GetMaximum(i)))
                calcium_score += calcium_cate(stat.GetMaximum(i)) * size / 3.
            return calcium_score

        elif dim == 2:
            for i in range(image.shape[-1]):
                binary_mask_slice = binary_mask[:, :, i]
                img_slice = image[:, :, i]
                masked_img_slice = binary_mask_slice * img_slice
                masked_data_slice = sitk.GetImageFromArray(masked_img_slice)
                img_data_slice = sitk.GetImageFromArray(img_slice)

                label = sitk.ConnectedComponent(masked_data_slice >= thresh)
                stat = sitk.LabelIntensityStatisticsImageFilter()
                stat.Execute(label, img_data_slice)

                for j in stat.GetLabels():
                    size = stat.GetSum(j) / stat.GetMean(j) * voxel_volume
                    print ("Calcified Plaque: {0} -> Mean: {1} Size: {2} Max: {3}".format(j, stat.GetMean(j), size, stat.GetMaximum(j)))
                    calcium_score += calcium_cate(stat.GetMaximum(j)) * size / 3.
            return calcium_score





########################
# REGRESSION METRICS
########################

