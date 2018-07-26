import numpy as np
from common.metric import EvalMetric, check_label_shapes


########################
# CLASSIFICATION METRICS
########################

class ClassificationMetric(EvalMetric):

    def __init__(self, name='ClsMetric', cls_num=1, allow_extra_outputs=False):
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


    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for pred, label in zip(preds, labels):
            pred_label = pred
            check_label_shapes(label, pred_label)
            ind = self.cls_num
            print "ind"
            print ind
            print "pred_label"
            print pred_label
            print "label"
            print label

            pred_pos = (pred_label == ind)
            label_pos = (label == ind)
            pred_neg = (pred_label == 0.)
            label_neg = (label == 0.)
            print "pred_pos"
            print pred_pos
            print "label_pos"
            print label_pos
            print "pred_neg"
            print pred_neg
            print "label_neg"
            print label_neg
            self.label_pos += np.asscalar(np.sum(label_pos))
            self.pred_pos += np.asscalar(np.sum(pred_pos))
            tp = np.asscalar(np.sum(pred_pos * label_pos))
            tn = np.asscalar(np.sum(pred_neg * label_neg))

            print tp
            print tn
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
        self.label_pos = 0.0
        self.pred_pos = 0.0

    def get_acc(self):
        """Get the evaluated accuracy

        :return:
        """
        tot_sample = self.tp + self.fp + self.tn + self.fn
        if tot_sample > 0:
            return (self.tp + self.tn) / tot_sample
        else:
            print ZeroDivisionError
            return np.nan

    def get_rec(self):
        """Get the evaluated recall.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """

        if self.tp + self.fn > 0.:
            return self.tp / (self.tp + self.fn)
        else:
            print ZeroDivisionError
            return np.nan

    def get_prec(self):
        """Get the evaluated precision

        :return:
        """

        if self.tp + self.fp > 0.:
            return self.tp / (self.tp + self.fp)
        else:
            print ZeroDivisionError
            return np.nan

    def get_fscore(self, beta=1.):
        """GEt the evaluated f score

        The F score is equivalent to weighted average of the precision and recall.
        where the best value is 1.0 and the worst value is 0.0. The formula for F score is:

        F_{beta} = (1 + \beta^2) * (precision * recall) / (\beta^2 * precision + recall)


        :param beta: float number accounts for the extent to which precision and recall are weighted differently, 1. by default
        :return:
        """
        if type(beta) is float:
            return (1 + beta ** 2) * (self.get_rec() * self.get_prec()) / (beta ** 2 * self.get_prec() + self.get_rec())
        else:
            print ("fscore requires a float beta as input, not {}".format(beta))
            return ValueError



########################
# REGRESSION METRICS
########################

