from __future__ import absolute_import
from collections import OrderedDict
import numpy as np
from model_eval.common import registry

def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    """
    def __init__(self, name, output_names=None, label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        label: OrderedDict of str -> ndarray
            name to array mapping for labels

        pred: list of ndarray
            name to array mapping of predicted outputs
        """
        if self.output_names is not None:
            preds = [pred[name] for name in self.output_names]
        else:
            preds = list(pred.values())

        if self.label_names is not None:
            labels = [label[name] for name in self.label_names]
        else:
            labels = list(label.values())

        self.update(labels, preds)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels: list of ndarray
            The labels of the data.

        preds: list of ndarray
            Predicted values.
        """
        raise NotImplementedError

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        names: list of str
            Name of the metrics.
        values: list of float
            Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

# pylint: disable=invalid-name
register = registry.get_register_func(EvalMetric, 'metric')
alias = registry.get_alias_func(EvalMetric, 'metric')
_create = registry.get_create_func(EvalMetric, 'metric')
# pylint: enable=invalid-name

def create(metric, *args, **kwargs):
    """Creates evaluation metric from metric names or instances of EvalMetric or a custom metric function.

    Parameters
    ----------
    metric: str or callable
        Specifies the metric to create.
        This argument must be one of the below:

        - Name of a metric.
        - An instance of 'EvalMetric'.
        - A list, each element of which is a metric or a metric name.
        - An evaluation function that computes custom metric for a given batch of labels and predictions.

    *args: list
        Additional arguments to metric constructor.
        Only used when metric is str.
    **kwargs: dict
        Additional arguments to metric constructor.
        Only used when metric is str

    Examples
    --------
    def custom_metric(label, pred):
        return np.mean(np.abs(label - pred))

    metric1 = metric.create('acc')
    metric2 = metric.create(custom_metric)
    metric3 = metric.create([metric1, metric2, 'rmse'])
    """
    if callable(metric):
        return CustomMetric(metric, *args, **kwargs)
    elif isinstance(metric, list):
        composite_metric = CompositeEvalMetric()
        for child_metric in metric:
            composite_metric.add(create(child_metric, *args, **kwargs))
        return composite_metric

    return _create(metric, *args, **kwargs)

@register
@alias('composite')
class CompositeEvalMetric(EvalMetric):
    """Manages multiple evaluation metrics.

    Parameters
    ----------
    metrics : list of EvalMetric
        List of child metrics.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    predicts = [np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    labels   = [np.array([0, 1, 1])]
    eval_metrics_1 = metric.Accuracy()
    eval_metrics_2 = metric.F1()
    eval_metrics = metric.CompositeEvalMetric()
    for child_metric in [eval_metrics_1, eval_metrics_2]:
        eval_metrics.add(child_metric)
        eval_metrics.update(labels = labels, preds = predicts)
        print eval_metrics.get()
    (['accuracy', 'f1'], [0.6666666666666666, 0.8])
    """

    def __init__(self, metrics=None, name='composite', output_names=None, label_names=None):
        super(CompositeEvalMetric, self).__init__(
            'composite', output_names=output_names, label_names=label_names)
        if metrics is None:
            metrics = []
        self.metrics = [create(i) for i in metrics]

    def add(self, metric):
        """Adds a child metric.

        Parameters
        ----------
        metric
            A metric instance.
        """
        self.metrics.append(create(metric))

    def get_metric(self, index):
        """Returns a child metric.

        Parameters
        ----------
        index: int
            Index of child metric in the list of metrics.
        """
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(index, len(self.metrics)))

    def update_dict(self, labels, preds):
        if self.label_names is not None:
            labels = OrderedDict([i for i in labels.items()
                                  if i[0] in self.label_names])
        if self.output_names is not None:
            preds = OrderedDict([i for i in preds.items()
                                 if i[0] in self.output_names])

        for metric in self.metrics:
            metric.update_dict(labels, preds)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels: list of ndarray
            The labels fo the data.

        preds: list of ndarray
            Predicted values.
        """
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        """Returns the current evaluation result.

        Returns
        -------
        names: list of str
            Names of the metrics.
        values: list of float
            Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get()
            if isinstance(name, basestring):
                name = [name]
            if isinstance(name, (float, int, long, np.generic)):
                value = [value]
            names.extend(name)
            values.extend(value)
        return (names, values)

    def get_config(self):
        config = super(CompositeEvalMetric, self).get_config()
        config.update({'metrics': [i.get_config for i in self.metrics]})
        return config

@register
class CustomMetric(EvalMetric):
    """Computes a customized evaluation metric.

    The `feval` function can return a `tuple` of (sum_metric, num_inst) or return
    an `int` sum_metric.

    Parameters
    ----------
    feval : callable(label, pred)
        Customized evaluation function.
    name : str, optional
        The name of the metric. (the default is None).
    allow_extra_outputs : bool, optional
        If true, the prediction outputs can have extra outputs.
        This is useful in RNN, where the states are also produced
        in outputs for forwarding. (the default is False).
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    predicts = [np.array([3, -0.5, 2, 7]).reshape(4,1))]
    labels = [np.array([2.5, 0.0, 2, 8]).reshape(4,1))]
    feval = lambda x, y : (x + y).mean()
    eval_metrics = metric.CustomMetric(feval=feval)
    eval_metrics.update(labels, predicts)
    print eval_metrics.get()
    ('custom(<lambda>)', 6.0)
    """
    def __init__(self, feval, name=None, allow_extra_outputs=False, output_names=None, label_names=None):
        if name is None:
            name = feval.__name__
            if name.find('<') != -1:
                name = 'custom(%s)' % name
        super(CustomMetric, self).__init__(
            name, feval=feval, allow_extra_outputs=allow_extra_outputs, output_names=output_names, label_names=label_names)
        self._feval = feval
        self._allow_extra_outputs = allow_extra_outputs

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels: list of ndarray
            The labels of the data.

        preds: list of ndarray
            Predicted values.
        """
        if not self._allow_extra_outputs:
            check_label_shapes(labels, preds)

        for pred, label in zip(preds, labels):
            reval = self._feval(label, pred)
            if isinstance(reval, tuple):
                (sum_metric, num_inst) = reval
                self.sum_metric += sum_metric
                self.num_inst += num_inst
            else:
                self.sum_metric += reval
                self.num_inst += 1

    def get_config(self):
        raise NotImplementedError("CustomMetric cannot be serialized")

########################
# CLASSIFICATION METRICS
########################

@register
@alias('acc')
class Accuracy(EvalMetric):
    """Computes accuracy binary classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis: int, default = 1
        The axis that represent classes
    name: str
        Name of this metric instance for display.
    output_names: list of str, or None
        Name of predictions that should be used when updating with update_dict
        By default include all predictions.
    label_names: list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    predicts = [np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    labels = [np.array([0, 1, 1])]
    acc = metric.Accuracy()
    acc.update(preds = predicts, labels = labels)
    print acc.get()
    ('accuracy', 0.666666666666666666)
    """
    def __init__(self, axis=1, name='accuracy', output_names=None, label_names=None):
        super(Accuracy, self).__init__(
            name, axis=axis, output_names=output_names, label_names=label_names)
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        
        Parameters
        ----------
        labels: list of ndarray
            The labels of the data with class indices as values, one per sample.
            
        preds: list of ndarray
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = np.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.astype('int32')
            label = label.astype('int32')

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)



@register
@alias('top_k_accuracy', 'top_k_acc')
class TopKAccuracy(EvalMetric):
    """Computes top k predictions accuracy.

    'TopKAccuracy' differs from Accuracy in that it considers the prediction to be "True" as long as the ground truth
    label is in the top K predicted labels.

    If 'top_k' = "1", then 'TopKAccuracy' is identical to 'Accuracy'.

    Parameters
    ----------
    top_k: int
        Whether targets are in top k predictions.
    name: str
        Name of this metric instance for display.
    output_names: list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    np.random.seed(999)
    top_k = 3
    labels = [np.array([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])]
    predicts = [np.array(np.random.rand(10, 10))]
    acc = metric.TopKAccuracy(top_k=top_k)
    acc.update(labels, predicts)
    print acc.get()
    ('top_k_accuracy', 0.3)
    """

    def __init__(self, top_k=1, name='top_k_accuracy', output_names=None, label_names=None):
        super(TopKAccuracy, self).__init__(name, top_k=top_k, output_names=output_names, label_names=label_names)
        self.top_k = top_k
        assert(self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        self.name += '_%d' % self.top_k

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels: list of ndarray
            The labels of the data.

        preds: list of ndarray
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            assert(len(pred_label.shape) <= 2), 'Predictions should be no more than 2 dims'
            pred_label = np.argsort(pred_label.astype('float32'), axis=1)
            label = label.astype('int32')
            check_label_shapes(label, pred_label)
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    self.sum_metric += (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
            self.num_inst += num_samples

@register
class FScore(EvalMetric):
    """Computes the F score of a binary classification problem

    The F score is equivalent to weighted average of the precision and recall.
    where the best value is 1.0 and the worst value is 0.0. The formula for F score is:

        F_{beta} = (1 + \beta^2) * (precision * recall) / (\beta^2 * precision + recall)

    where \beta accounts for the extent to which precision and recall are weighted differently

    The formula for precision and recall is:

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

    .. note:

        This F score only supports binary classification.

    Parameters
    ----------
    name: str
        Name of this metric instance for display.
    outpu_names: list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names: list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    predicts = [np.array([[0.3, 0.7], [0., 1.], [0.4, 0.6]])]
    labels = [np.array([0., 1., 1.])]
    beta = 1.
    acc = metric.FScore()
    acc.update(preds = predicts, labels = labels)
    print acc.get()
    ('fscore', 0.8)
    """

    def __init__(self, name='fscore', output_names=None, label_names=None):
        super(FScore, self).__init__(
            name, output_names=output_names, label_names=label_names
        )

    def update(self, beta, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels: list of ndarray
            The labels of the data

        preds: list of ndarray
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.astype('int32')
            pred_label = np.argmax(pred, axis=1)

            check_label_shapes(label, pred)
            if len(np.unique(label)) > 2:
                raise ValueError("FScore currently only supports binary classification.")

            true_positives, false_positives, false_negatives = 0., 0., 0.

            for y_pred, y_true in zip(pred_label, label):
                if y_pred == 1 and y_true == 1:
                    true_positives += 1.
                elif y_pred == 1 and y_true == 0:
                    false_positives += 1.
                elif y_pred == 0 and y_true == 1:
                    false_negatives += 1.

            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = np.nan

            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = np.nan

            if precision + recall > 0:
                f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
            elif recall == np.nan or precision == np.nan:
                f_score = np.nan
            else:
                f_score = 0.

            self.sum_metric += f_score
            self.sum_inst += 1

#####################
# REGRESSION METRICS
#####################

@register
@alias('ce')
class CrossEntropy(EvalMetric):
    """Computes Cross Entropy loss.

    The cross entropy over a batch of sample size: math: 'N' is given by

    .. math::
        -\\sum_{n=1}^{N}\\sum_{k=1}^{K}t_{nk}\\log (y_{nk}),

    where: math: 't_{nk}=1' if anf only if sample: math: 'n' belongs to class: math: 'k'.
    : math:'y_{nk}' denote the probability of sample: math: 'n' belonging to class: math: 'k'.

    Parameters
    ----------
    eps: float
        Cross Entropy loss is undefined for predicted value 0 or 1, so predicted values are added with the small constant
    name: str
        Name of this metric instance for display
    output_names: list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names: list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    predicts = [mx.nd.array([[0.3, 0.7], [0., 1.], [0.4, 0.6]])]
    labels = [mx.nd.array([0, 1, 1])]
    ce = metric.CrossEntropy()
    ce.update(labels, predicts)
    print ce.get()
    ('cross-entropy', 0.57159948348999023)
    """
    def __init__(self, eps=1e-12, name='cross-entropy', output_names=None, label_names=None):
        super(CrossEntropy, self).__init__(
            name, eps=eps, output_names=output_names, label_names=label_names)
        self.eps = eps

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels: list of ndarray
            The labels of the data.

        preds: list of ndarray
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.ravel
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]