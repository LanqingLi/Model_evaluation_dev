# -- coding: utf-8 --
import numpy as np
import os, json
import argparse
import pandas as pd
#from config import config
from config import BrainConfig
import nrrd, cv2
from model_eval.brain.contour_draw import contour_and_draw, contour_and_draw_rainbow
from model_eval.tools.data_postprocess import save_xlsx_json
from model_eval.tools.data_preprocess import window_convert
from model_eval.common.custom_metric import ClassificationMetric

def default_post_processor(img_array, prob_map_array):
    return prob_map_array

class BrainSemanticSegEvaluatorOnlineIter(object):
    '''
    This class has the same evaluation functionality as BrainSemanticSegEvaluatorOnline. But instead of taking lists
    of input data sorted by patient ID, it takes a data iterator and a predictor function, and iterates through all
    patients by key words. Therefore, it can be smoothly connected to model inference to produce evaluation result (end-to-end).
    Parameters
    ----------
    data_iter: an iterator which takes a list of patient folder names and iterates through each patient (data folder).
    All images of a single patient are considered a batch.
        e.g.
        data_iter = FileIter(
            root_dir=config.validating.valid_dir,
            persons=config.validating.valid_person,
            is_train=False,
            rgb_mean=(0, 0, 0)
        )
        whenever its next method is invoked, it returns a dictionary which contains the following items:
        data =
            {predict_key: image data after post-process, can be fed into predictor to perform inference, shape = (figure number(batch size), 3(RGB channel), 512, 512)
             gt_key: list = [ground truth label/mask, shape = (512, 512, figure num), info of ct scan]
             img_key: raw image, shape = (figure number, 3(RGB channel), 512, 512)
             patient_key: patient ID
            }
    predictor: a predictor function which performs inference, it generates a np.ndarray, with shape = (batch size, num of classes, 512, 512)
    '''
    def __init__(self, cls_label_xls_path, data_iter, predict_key, gt_key, img_key, patient_key, predictor, post_processor=default_post_processor,
                 if_post_process = False, img_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_contour'),
                 mask_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_mask'), if_save_mask = False,
                 score_type='fscore', result_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_result'),
                 xlsx_name='BrainSemanticSegEvaluation.xlsx', json_name='BrainSemanticSegEvaluation',
                 conf_thresh=np.linspace(0.1, 0.9, num=9).tolist(), fscore_beta=1., thresh=0.5):
        config = BrainConfig(cls_label_xls_path=cls_label_xls_path)
        self.data_iter = data_iter
        # reset data_iter when initialized, to make sure we start from the beginning when doing iteration
        self.data_iter.reset()
        self.predict_key = predict_key
        self.gt_key = gt_key
        self.img_key = img_key
        self.patient_key = patient_key
        # a predictor function to read data and generate predicted result
        self.predictor = predictor
        self.post_processor = post_processor
        self.if_post_process = if_post_process
        self.img_save_dir = img_save_dir
        self.mask_save_dir = mask_save_dir
        # config.CLASSES 包含background class,是模型预测的所有类别
        self.cls_name = config.CLASSES
        self.cls_dict = config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.result_df = pd.DataFrame(
            columns=['PatientID', 'class', 'threshold', 'tp_count', 'tn_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision', 'fp/tp', 'dice', 'gt_vol', 'pred_vol', 'gt_phys_vol/mm^3',
                     'pred_phys_vol/mm^3', 'phys_vol_diff/mm^3', self.score_type, 'Patient_Number'])
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.conf_thresh = conf_thresh
        self.cls_weights = config.CLASS_WEIGHTS
        self.fscore_beta = fscore_beta
        self.thresh = thresh
        self.patient_num = 0.
        # save predicted mask as .npy when if_save_mask=True
        if if_save_mask:
            if not os.path.exists(self.mask_save_dir):
                os.mkdir(self.mask_save_dir)
            for data in self.data_iter:
                if data is None:
                    break
                if data[self.predict_key] is None:
                    continue
                if not os.path.exists(os.path.join(self.mask_save_dir, data['pid'])):
                    os.mkdir(os.path.join(self.mask_save_dir, data['pid']))
                predict_data = self.predictor(data[self.predict_key])
                np.save(os.path.join(self.mask_save_dir, data['pid'], data['pid'] + '.npy'), predict_data)
            self.data_iter.reset()


    def binary_class_contour_plot_single_thresh(self):
        self.data_iter.reset()
        if not os.path.exists(self.img_save_dir):
            os.mkdir(self.img_save_dir)

        for data in self.data_iter:
            if data is None:
                break
            if data[self.predict_key] is None:
                continue
            predict_data = self.predictor(data[self.predict_key])
            gt_nrrd = data[self.gt_key]
            # the input raw data (images) have shape (figure number, 3(RGB channel), 512, 512), we need to transpose
            # to shape (512, 512, figure number, 3(RGB channel))
            img_nrrd = data[self.img_key].transpose(2, 3, 0, 1)
            PatientID = data[self.patient_key]
            # dim of the raw image
            dim = [img_nrrd.shape[0], img_nrrd.shape[1]]

            print ('processing PatientID: %s' % PatientID)
            assert predict_data.shape[
                       1] == 2, 'the number of classes %s in predict labels should be 2 for binary classification' % (
                predict_data.shape[1])

            # transpose the predict_data to be shape = [512, 512, figure number]
            # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
            predict_data_cpy = predict_data[:, 1, :, :].copy()
            predict_data_cpy = predict_data_cpy.transpose(1, 2, 0)
            # check if the predict and ground truth labels have the same shape
            if not predict_data_cpy.shape == gt_nrrd[0].shape:
                raise Exception("predict and ground truth labels must have the same shape")

            for fig_num in range(predict_data_cpy.shape[-1]):
                gt_img = img_nrrd[:, :, fig_num]
                gt_img = window_convert(gt_img, 40, 80)
                gt_map = gt_nrrd[0][:, :, fig_num]

                predict_map = predict_data_cpy[:, :, fig_num]
                img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                lab = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                binary_pmap = np.array(predict_map > self.thresh, dtype=np.int)

                if self.if_post_process:
                    # transpose data[self.predict_key] shape = (figure number, 3(RGB channel), 512, 512) to
                    # shape = (512, 512, figure number, 3(RGB channel))
                    img_data = ((data[self.predict_key].copy()).transpose(2, 3, 0, 1))[:, :, fig_num, :]
                    binary_pmap = self.post_processor(img_data, binary_pmap)

                lab, _ = contour_and_draw(lab, gt_map)
                img, _ = contour_and_draw(img, binary_pmap)

                prob = predict_map[:, :, np.newaxis] * 255
                prob = np.array(np.repeat(prob, axis=2, repeats=3), dtype=np.uint8)

                to_show = np.zeros(shape=(dim[0], dim[1]*2, 3), dtype=np.uint8)
                to_show[:, :dim[1], :] = img
                to_show[:, dim[1]:, :] = lab

                cv2.imwrite(os.path.join(self.img_save_dir, PatientID + '-' + str(fig_num).zfill(2) + 'thresh=%s.jpg' %(self.thresh)), to_show)

    def binary_class_contour_plot_multi_thresh(self):
        self.data_iter.reset()
        if not os.path.exists(self.img_save_dir):
            os.mkdir(self.img_save_dir)

        for data in self.data_iter:
            if data is None:
                break
            if data[self.predict_key] is None:
                continue
            predict_data = self.predictor(data[self.predict_key])
            gt_nrrd = data[self.gt_key]
            # the input raw data (images) have shape (figure number, 3(RGB channel), 512, 512), we need to transpose
            # to shape (512, 512, figure number, 3(RGB channel))
            img_nrrd = data[self.img_key].transpose(2, 3, 0, 1)
            PatientID = data[self.patient_key]
            # dim of the raw image
            dim = [img_nrrd.shape[0], img_nrrd.shape[1]]

            print ('processing PatientID: %s' % PatientID)
            assert predict_data.shape[
                       1] == 2, 'the number of classes %s in predict labels should be 2 for binary classification' % (
                predict_data.shape[1])
            # transpose the predict_data to be shape = [512, 512, figure number]
            # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
            predict_data_cpy = predict_data[:, 1, :, :].copy()
            predict_data_cpy = predict_data_cpy.transpose((1, 2, 0))
            # check if the predict and ground truth labels have the same shape
            if not predict_data_cpy.shape == gt_nrrd[0].shape:
                raise Exception("predict and ground truth labels must have the same shape")

            for fig_num in range(predict_data_cpy.shape[-1]):
                gt_img = img_nrrd[:, :, fig_num]
                gt_img = window_convert(gt_img, 40, 80)
                gt_map = gt_nrrd[0][:, :, fig_num]
                predict_map = predict_data_cpy[:, :, fig_num]
                img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                lab = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                lab, _ = contour_and_draw(lab, gt_map)
                # superimpose contour images for multiple thresholds
                for index, thresh in enumerate(self.conf_thresh):

                    binary_pmap = np.array(predict_map > thresh, dtype=np.int)

                    if self.if_post_process:
                        # transpose data[self.predict_key] shape = (figure number, 3(RGB channel), 512, 512) to
                        # shape = (512, 512, figure number, 3(RGB channel))
                        img_data = ((data[self.predict_key].copy()).transpose(2, 3, 0, 1))[:, :, fig_num, :]
                        binary_pmap = self.post_processor(img_data, binary_pmap)

                    img, _ = contour_and_draw_rainbow(img, binary_pmap, color_range=len(self.conf_thresh),
                                                      color_num=index)

                    prob = predict_map[:, :, np.newaxis] * 255
                    prob = np.array(np.repeat(prob, axis=2, repeats=3), dtype=np.uint8)

                    to_show = np.zeros(shape=(dim[0], dim[1]*2, 3), dtype=np.uint8)
                    to_show[:, :dim[1], :] = img
                    to_show[:, dim[1]:, :] = lab

                cv2.imwrite(os.path.join(self.img_save_dir, PatientID + '-' + str(fig_num).zfill(2) + 'multi_thresh.jpg'),
                            to_show)

    # 多分类分割模型通用评分,用阈值筛选正类别，并在其中选取最大值作为one-hot label
    def multi_class_evaluation(self):

        for thresh in self.conf_thresh:
            # reset data iterator, otherwise we cannot perform new iteration
            self.data_iter.reset()
            print ('threshold = %s' % thresh)
            # predict_data.shape() = [figure number, class number, 512, 512], gt_nrrd.shape() = [512, 512, figure number],
            # the pictures are arranged in the same order

            # initialize ClassificationMetric class for total stat (all patients) and update with ground truth/predict labels
            cls_metric = ClassificationMetric(cls_num=len(self.cls_name)-1, if_binary=True, pos_cls_fusion=False)
            gt_tot_phys_vol = [0. for _ in range(len(self.cls_name) - 1)]
            pred_tot_phys_vol = [0. for _ in range(len(self.cls_name) - 1)]
            self.patient_num = 0.

            for data in self.data_iter:
                if data is None:
                    break
                if data[self.predict_key] is None:
                    continue
                self.patient_num += 1.
                predict_data = self.predictor(data[self.predict_key])
                gt_nrrd = data[self.gt_key]
                PatientID = data[self.patient_key]

                print ('processing PatientID: %s' % PatientID)
                # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
                predict_data_cpy = predict_data.copy()
                # transpose predict_data_cpy to shape [512, 512, figure number, class number]
                predict_data_cpy = predict_data_cpy.transpose((2, 3, 0, 1))
                assert predict_data_cpy.shape[-1] == len(self.cls_name), 'the num of classes: %s in predict labels ' \
                                                                         'should equal that defined in brain/classname_labelname_mapping.xls: %s' % (
                                                                             predict_data_cpy.shape[-1],
                                                                             len(self.cls_name))
                predict_data_slice_cpy = predict_data[:, 0, :, :].copy()
                predict_data_slice_cpy = predict_data_slice_cpy.transpose((1, 2, 0))
                # check if the predict and ground truth labels have the same shape
                print (predict_data_slice_cpy.shape)
                print (gt_nrrd[0].shape)
                if not predict_data_slice_cpy.shape == gt_nrrd[0].shape:
                    raise Exception("predict and ground truth labels must have the same shape")

                predict_label_list = []
                gt_label_list = []
                for fig_num in range(predict_data_cpy.shape[-2]):
                    gt_label = gt_nrrd[0][:, :, fig_num]
                    gt_label_list.append(gt_label)
                    predict_score = predict_data_cpy[:, :, fig_num, :].copy()
                    for cls in range(predict_data_cpy.shape[-1]):
                        if self.cls_name[cls] == '__background__':
                            predict_score[:, :, cls] = 0.
                            continue
                        predict_score[predict_data_cpy[:, :, fig_num, cls] <= thresh] = 0.
                    # return the predict label as the index of the maximum value across all classes, the __background__
                    # class should be the first one, so if none of the positive class score is greater than zero, the
                    # pixel is classified as the background
                    predict_label = np.argmax(predict_score, axis=2)
                    predict_label_list.append(predict_label)

                # list-to-array-to-list conversion for calculating the overall stat (each patient)
                predict_list = [np.asarray(predict_label_list)]
                gt_list = [np.asarray(gt_label_list)]

                # calculate physical volume using space directions info stored in .nrrd
                space_matrix = np.zeros((3, 3))
                space_matrix[0] = np.asarray(gt_nrrd[1]['space directions'][0]).astype('float32')
                space_matrix[1] = np.asarray(gt_nrrd[1]['space directions'][1]).astype('float32')
                space_matrix[2] = np.asarray(gt_nrrd[1]['space directions'][2]).astype('float32')
                # calculate voxel volume as the determinant of spacing matrix
                voxel_vol = np.linalg.det(space_matrix)

                # initialize ClassificationMetric class for each patient and update with ground truth/predict labels
                patient_metric = ClassificationMetric(cls_num=len(self.cls_name)-1, if_binary=True, pos_cls_fusion=False)
                for cls_label in range(len(self.cls_name)):
                    if self.cls_name[cls_label] == '__background__':
                        continue

                    cls_metric.update(gt_list, predict_list, cls_label=cls_label)

                    patient_metric.update(gt_list, predict_list, cls_label=cls_label)

                    if patient_metric.tp[cls_label-1] == 0:
                        fp_tp = np.nan
                    else:
                        fp_tp = patient_metric.fp[cls_label-1] / patient_metric.tp[cls_label-1]

                    self.result_df = self.result_df.append({'PatientID': PatientID,
                                                            'class': self.cls_name[cls_label],
                                                            'threshold': thresh,
                                                            'tp_count': patient_metric.tp[cls_label-1],
                                                            'tn_count': patient_metric.tn[cls_label-1],
                                                            'fp_count': patient_metric.fp[cls_label-1],
                                                            'fn_count': patient_metric.fn[cls_label-1],
                                                            'accuracy': patient_metric.get_acc(cls_label),
                                                            'recall': patient_metric.get_rec(cls_label),
                                                            'precision': patient_metric.get_prec(cls_label),
                                                            'fp/tp': fp_tp,
                                                            'dice': patient_metric.get_dice(cls_label),
                                                            'gt_vol': patient_metric.get_gt_vol(cls_label),
                                                            'pred_vol': patient_metric.get_pred_vol(cls_label),
                                                            'gt_phys_vol/mm^3': patient_metric.get_gt_vol(cls_label) * voxel_vol,
                                                            'pred_phys_vol/mm^3': patient_metric.get_pred_vol(cls_label) * voxel_vol,
                                                            'phys_vol_diff/mm^3': patient_metric.get_gt_vol(cls_label) * voxel_vol -
                                                                                  patient_metric.get_pred_vol(cls_label) * voxel_vol,
                                                            self.score_type: patient_metric.get_fscore(
                                                                cls_label=cls_label, beta=self.fscore_beta)},
                                                            ignore_index=True)
                    gt_tot_phys_vol[cls_label-1] += patient_metric.get_gt_vol(cls_label) * voxel_vol
                    pred_tot_phys_vol[cls_label-1] += patient_metric.get_pred_vol(cls_label) * voxel_vol

            for cls_label in range(len(self.cls_name)):
                if self.cls_name[cls_label] == '__background__':
                    continue

                if cls_metric.tp[cls_label-1] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric.fp[cls_label-1] / cls_metric.tp[cls_label-1]

                self.result_df = self.result_df.append({'PatientID': 'total',
                                                        'class': self.cls_name[cls_label],
                                                        'threshold': thresh,
                                                        'tp_count': cls_metric.tp[cls_label-1],
                                                        'tn_count': cls_metric.tn[cls_label-1],
                                                        'fp_count': cls_metric.fp[cls_label-1],
                                                        'fn_count': cls_metric.fn[cls_label-1],
                                                        'accuracy': cls_metric.get_acc(cls_label),
                                                        'recall': cls_metric.get_rec(cls_label),
                                                        'precision': cls_metric.get_prec(cls_label),
                                                        'fp/tp': fp_tp,
                                                        'dice': cls_metric.get_dice(cls_label),
                                                        'gt_vol': cls_metric.get_gt_vol(cls_label),
                                                        'pred_vol': cls_metric.get_pred_vol(cls_label),
                                                        'gt_phys_vol/mm^3': gt_tot_phys_vol[cls_label-1],
                                                        'pred_phys_vol/mm^3': pred_tot_phys_vol[cls_label-1],
                                                        'phys_vol_diff/mm^3': gt_tot_phys_vol[cls_label-1] - pred_tot_phys_vol[cls_label-1],
                                                        self.score_type: cls_metric.get_fscore(
                                                            cls_label=cls_label, beta=self.fscore_beta),
                                                        'Patient_Number': self.patient_num},
                                                       ignore_index=True)
                self.result_df = self.result_df.append({'PatientID': 'average',
                                                        'class': self.cls_name[cls_label],
                                                        'threshold': thresh,
                                                        'tp_count': cls_metric.tp[cls_label - 1]/self.patient_num,
                                                        'tn_count': cls_metric.tn[cls_label - 1]/self.patient_num,
                                                        'fp_count': cls_metric.fp[cls_label - 1]/self.patient_num,
                                                        'fn_count': cls_metric.fn[cls_label - 1]/self.patient_num,
                                                        'accuracy': cls_metric.get_acc(cls_label),
                                                        'recall': cls_metric.get_rec(cls_label),
                                                        'precision': cls_metric.get_prec(cls_label),
                                                        'fp/tp': fp_tp,
                                                        'dice': cls_metric.get_dice(cls_label),
                                                        'gt_vol': cls_metric.get_gt_vol(cls_label)/self.patient_num,
                                                        'pred_vol': cls_metric.get_pred_vol(cls_label)/self.patient_num,
                                                        'gt_phys_vol/mm^3': gt_tot_phys_vol[cls_label - 1]/self.patient_num,
                                                        'pred_phys_vol/mm^3': pred_tot_phys_vol[cls_label - 1]/self.patient_num,
                                                        'phys_vol_diff/mm^3': (gt_tot_phys_vol[cls_label - 1] -
                                                                              pred_tot_phys_vol[cls_label - 1])/self.patient_num,
                                                        self.score_type: cls_metric.get_fscore(
                                                            cls_label=cls_label, beta=self.fscore_beta),
                                                        'Patient_Number': self.patient_num},
                                                       ignore_index=True)

                # find the optimal threshold
                if self.cls_name[cls_label] not in self.opt_thresh:

                    self.opt_thresh[self.cls_name[cls_label]] = self.result_df.iloc[-1]

                    self.opt_thresh[self.cls_name[cls_label]].loc['threshold'] = thresh

                else:
                    # we choose the optimal threshold corresponding to the one that gives the highest model score
                    if self.result_df.iloc[-1][self.score_type] > self.opt_thresh[self.cls_name[cls_label]][
                        self.score_type]:
                        self.opt_thresh[self.cls_name[cls_label]] = self.result_df.iloc[-1]
                        self.opt_thresh[self.cls_name[cls_label]].loc['threshold'] = thresh

        self.result_df = self.result_df.sort_values(['threshold', 'PatientID', 'class'])

        save_xlsx_json(self.result_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'multi-class_evaluation', 'optimal_threshold')


    # 多分类分割模型通用评分轻量级版本，每张图只做一遍预测,用阈值筛选正类别，并在其中选取最大值作为one-hot label
    def multi_class_evaluation_light(self):
        cls_metric_list = []
        gt_tot_phys_vol = []
        pred_tot_phys_vol = []
        self.data_iter.reset()
        for thresh in self.conf_thresh:
            print ('threshold = %s' % thresh)
            # predict_data.shape() = [figure number, class number, 512, 512], gt_nrrd.shape() = [512, 512, figure number],
            # the pictures are arranged in the same order

            # initialize ClassificationMetric class for total stat (all patients) and update with ground truth/predict labels
            cls_metric_list.append(ClassificationMetric(cls_num=len(self.cls_name) - 1, if_binary=True, pos_cls_fusion=False))
            gt_tot_phys_vol.append([0. for _ in range(len(self.cls_name) - 1)])
            pred_tot_phys_vol.append([0. for _ in range(len(self.cls_name) - 1)])

        self.patient_num = 0.

        for data in self.data_iter:
            if data is None:
                break
            if data[self.predict_key] is None:
                continue
            self.patient_num += 1.
            predict_data = self.predictor(data[self.predict_key])
            gt_nrrd = data[self.gt_key]
            PatientID = data[self.patient_key]

            print ('processing PatientID: %s' % PatientID)
            # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
            predict_data_cpy = predict_data.copy()
            # transpose predict_data_cpy to shape [512, 512, figure number, class number]
            predict_data_cpy = predict_data_cpy.transpose((2, 3, 0, 1))
            assert predict_data_cpy.shape[-1] == len(self.cls_name), 'the num of classes: %s in predict labels ' \
                                                                     'should equal that defined in brain/classname_labelname_mapping.xls: %s' % (
                                                                         predict_data_cpy.shape[-1],
                                                                         len(self.cls_name))
            predict_data_slice_cpy = predict_data[:, 0, :, :].copy()
            predict_data_slice_cpy = predict_data_slice_cpy.transpose((1, 2, 0))
            # check if the predict and ground truth labels have the same shape
            print (predict_data_slice_cpy.shape)
            print (gt_nrrd[0].shape)
            if not predict_data_slice_cpy.shape == gt_nrrd[0].shape:
                raise Exception("predict and ground truth labels must have the same shape")


            for index, thresh in enumerate(self.conf_thresh):
                predict_label_list = []
                gt_label_list = []
                for fig_num in range(predict_data_cpy.shape[-2]):
                    gt_label = gt_nrrd[0][:, :, fig_num]
                    gt_label_list.append(gt_label)
                    predict_score = predict_data_cpy[:, :, fig_num, :].copy()
                    for cls in range(predict_data_cpy.shape[-1]):
                        if self.cls_name[cls] == '__background__':
                            predict_score[:, :, cls] = 0.
                            continue
                        predict_score[predict_data_cpy[:, :, fig_num, cls] <= thresh] = 0.
                    # return the predict label as the index of the maximum value across all classes, the __background__
                    # class should be the first one, so if none of the positive class score is greater than zero, the
                    # pixel is classified as the background
                    predict_label = np.argmax(predict_score, axis=2)
                    predict_label_list.append(predict_label)

                # list-to-array-to-list conversion for calculating the overall stat (each patient)
                predict_list = [np.asarray(predict_label_list)]
                gt_list = [np.asarray(gt_label_list)]

                # calculate physical volume using space directions info stored in .nrrd
                space_matrix = np.zeros((3, 3))
                space_matrix[0] = np.asarray(gt_nrrd[1]['space directions'][0]).astype('float32')
                space_matrix[1] = np.asarray(gt_nrrd[1]['space directions'][1]).astype('float32')
                space_matrix[2] = np.asarray(gt_nrrd[1]['space directions'][2]).astype('float32')
                # calculate voxel volume as the determinant of spacing matrix
                voxel_vol = np.linalg.det(space_matrix)

                # initialize ClassificationMetric class for each patient and update with ground truth/predict labels
                patient_metric = ClassificationMetric(cls_num=len(self.cls_name) - 1, if_binary=True,
                                                      pos_cls_fusion=False)
                for cls_label in range(len(self.cls_name)):
                    if self.cls_name[cls_label] == '__background__':
                        continue

                    cls_metric_list[index].update(gt_list, predict_list, cls_label=cls_label)

                    patient_metric.update(gt_list, predict_list, cls_label=cls_label)

                    if patient_metric.tp[cls_label - 1] == 0:
                        fp_tp = np.nan
                    else:
                        fp_tp = patient_metric.fp[cls_label - 1] / patient_metric.tp[cls_label - 1]

                    self.result_df = self.result_df.append({'PatientID': PatientID,
                                                            'class': self.cls_name[cls_label],
                                                            'threshold': thresh,
                                                            'tp_count': patient_metric.tp[cls_label - 1],
                                                            'tn_count': patient_metric.tn[cls_label - 1],
                                                            'fp_count': patient_metric.fp[cls_label - 1],
                                                            'fn_count': patient_metric.fn[cls_label - 1],
                                                            'accuracy': patient_metric.get_acc(cls_label),
                                                            'recall': patient_metric.get_rec(cls_label),
                                                            'precision': patient_metric.get_prec(cls_label),
                                                            'fp/tp': fp_tp,
                                                            'dice': patient_metric.get_dice(cls_label),
                                                            'gt_vol': patient_metric.get_gt_vol(cls_label),
                                                            'pred_vol': patient_metric.get_pred_vol(cls_label),
                                                            'gt_phys_vol/mm^3': patient_metric.get_gt_vol(
                                                                                cls_label) * voxel_vol,
                                                            'pred_phys_vol/mm^3': patient_metric.get_pred_vol(
                                                                                cls_label) * voxel_vol,
                                                            'phys_vol_diff/mm^3': patient_metric.get_gt_vol(
                                                                                cls_label) * voxel_vol -
                                                                                  patient_metric.get_pred_vol(
                                                                                cls_label) * voxel_vol,
                                                            self.score_type: patient_metric.get_fscore(
                                                                cls_label=cls_label, beta=self.fscore_beta)},
                                                           ignore_index=True)
                    gt_tot_phys_vol[index][cls_label - 1] += patient_metric.get_gt_vol(cls_label) * voxel_vol
                    pred_tot_phys_vol[index][cls_label - 1] += patient_metric.get_pred_vol(cls_label) * voxel_vol

        for index, thresh in enumerate(self.conf_thresh):
            for cls_label in range(len(self.cls_name)):
                if self.cls_name[cls_label] == '__background__':
                    continue

                if cls_metric_list[index].tp[cls_label - 1] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric_list[index].fp[cls_label - 1] / cls_metric_list[index].tp[cls_label - 1]

                self.result_df = self.result_df.append({'PatientID': 'total',
                                                        'class': self.cls_name[cls_label],
                                                        'threshold': thresh,
                                                        'tp_count': cls_metric_list[index].tp[cls_label - 1],
                                                        'tn_count': cls_metric_list[index].tn[cls_label - 1],
                                                        'fp_count': cls_metric_list[index].fp[cls_label - 1],
                                                        'fn_count': cls_metric_list[index].fn[cls_label - 1],
                                                        'accuracy': cls_metric_list[index].get_acc(cls_label),
                                                        'recall': cls_metric_list[index].get_rec(cls_label),
                                                        'precision': cls_metric_list[index].get_prec(cls_label),
                                                        'fp/tp': fp_tp,
                                                        'dice': cls_metric_list[index].get_dice(cls_label),
                                                        'gt_vol': cls_metric_list[index].get_gt_vol(cls_label),
                                                        'pred_vol': cls_metric_list[index].get_pred_vol(cls_label),
                                                        'gt_phys_vol/mm^3': gt_tot_phys_vol[index][cls_label - 1],
                                                        'pred_phys_vol/mm^3': pred_tot_phys_vol[index][cls_label - 1],
                                                        'phys_vol_diff/mm^3': gt_tot_phys_vol[index][cls_label - 1] -
                                                                              pred_tot_phys_vol[index][cls_label - 1],
                                                        self.score_type: cls_metric_list[index].get_fscore(
                                                            cls_label=cls_label, beta=self.fscore_beta),
                                                        'Patient_Number': self.patient_num},
                                                       ignore_index=True)
                self.result_df = self.result_df.append({'PatientID': 'average',
                                                        'class': self.cls_name[cls_label],
                                                        'threshold': thresh,
                                                        'tp_count': cls_metric_list[index].tp[cls_label - 1]/self.patient_num,
                                                        'tn_count': cls_metric_list[index].tn[cls_label - 1]/self.patient_num,
                                                        'fp_count': cls_metric_list[index].fp[cls_label - 1]/self.patient_num,
                                                        'fn_count': cls_metric_list[index].fn[cls_label - 1]/self.patient_num,
                                                        'accuracy': cls_metric_list[index].get_acc(cls_label),
                                                        'recall': cls_metric_list[index].get_rec(cls_label),
                                                        'precision': cls_metric_list[index].get_prec(cls_label),
                                                        'fp/tp': fp_tp,
                                                        'dice': cls_metric_list[index].get_dice(cls_label),
                                                        'gt_vol': cls_metric_list[index].get_gt_vol(cls_label)/self.patient_num,
                                                        'pred_vol': cls_metric_list[index].get_pred_vol(cls_label)/self.patient_num,
                                                        'gt_phys_vol/mm^3': gt_tot_phys_vol[index][cls_label - 1]/self.patient_num,
                                                        'pred_phys_vol/mm^3': pred_tot_phys_vol[index][cls_label - 1]/self.patient_num,
                                                        'phys_vol_diff/mm^3': (gt_tot_phys_vol[index][cls_label - 1] -
                                                                              pred_tot_phys_vol[index][cls_label - 1])/self.patient_num,
                                                        self.score_type: cls_metric_list[index].get_fscore(
                                                            cls_label=cls_label, beta=self.fscore_beta),
                                                        'Patient_Number': self.patient_num},
                                                       ignore_index=True)

                # find the optimal threshold
                if self.cls_name[cls_label] not in self.opt_thresh:

                    self.opt_thresh[self.cls_name[cls_label]] = self.result_df.iloc[-1]

                    self.opt_thresh[self.cls_name[cls_label]].loc['threshold'] = thresh

                else:
                    # we choose the optimal threshold corresponding to the one that gives the highest model score
                    if self.result_df.iloc[-1][self.score_type] > self.opt_thresh[self.cls_name[cls_label]][
                        self.score_type]:
                        self.opt_thresh[self.cls_name[cls_label]] = self.result_df.iloc[-1]
                        self.opt_thresh[self.cls_name[cls_label]].loc['threshold'] = thresh

        self.result_df = self.result_df.sort_values(['threshold', 'PatientID', 'class'])

        save_xlsx_json(self.result_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'multi-class_evaluation', 'optimal_threshold')

    # 二分类分割模型评分，用阈值筛选正类别,不管gt有多少类别，我们只关心检出(正样本全部归为一类,pos_cls_fusion=True)
    def binary_class_evaluation(self):

        for thresh in self.conf_thresh:
            # reset data iterator, otherwise we cannot perform new iteration
            self.data_iter.reset()
            print ('threshold = %s' % thresh)
            # predict_data.shape() = [figure number, class number, 512, 512], gt_nrrd.shape() = [512, 512, figure number],
            # the pictures are arranged in the same order

            # initialize ClassificationMetric class for total stat (all patients) and update with ground truth/predict labels
            cls_metric = ClassificationMetric(cls_num=1, if_binary=True, pos_cls_fusion=True)
            gt_tot_phys_vol = [0.]
            pred_tot_phys_vol = [0.]
            self.patient_num = 0.

            for data in self.data_iter:
                if data is None:
                    break
                if data[self.predict_key] is None:
                    continue
                self.patient_num += 1.
                predict_data = self.predictor(data[self.predict_key])
                gt_nrrd = data[self.gt_key]
                PatientID = data[self.patient_key]
                print ('processing PatientID: %s' % PatientID)

                predict_label_list = []
                gt_label_list = []

                binary_pmap = np.array(predict_data > self.thresh, dtype=np.int)

                if self.if_post_process:
                    binary_pmap = self.post_processor(data[self.predict_key], binary_pmap)

                # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
                predict_data_cpy = predict_data.copy()
                # transpose predict_data_cpy to shape [512, 512, figure number, class number]
                predict_data_cpy = predict_data_cpy.transpose((2, 3, 0, 1))
                assert predict_data_cpy.shape[-1] == len(self.cls_name), 'the num of classes: %s in predict labels ' \
                                                                         'should equal that defined in brain/classname_labelname_mapping.xls: %s' % (
                                                                             predict_data_cpy.shape[-1],
                                                                             len(self.cls_name))
                predict_data_slice_cpy = predict_data[:, 0, :, :].copy()
                predict_data_slice_cpy = predict_data_slice_cpy.transpose((1, 2, 0))
                # check if the predict and ground truth labels have the same shape
                if not predict_data_slice_cpy.shape == gt_nrrd[0].shape:
                    raise Exception("predict and ground truth labels must have the same shape")

                for fig_num in range(predict_data_cpy.shape[-2]):
                    gt_label = gt_nrrd[0][:, :, fig_num]
                    gt_label_list.append(gt_label)
                    predict_score = predict_data_cpy[:, :, fig_num, :].copy()
                    for cls in range(predict_data_cpy.shape[-1]):
                        if self.cls_name[cls] == '__background__':
                            predict_score[:, :, cls] = 0.
                            continue
                        predict_score[predict_data_cpy[:, :, fig_num, cls] <= thresh] = 0.
                    # return the predict label as the index of the maximum value across all classes, the __background__
                    # class should be the first one, so if none of the positive class score is greater than zero, the
                    # pixel is classified as the background
                    predict_label = np.argmax(predict_score, axis=2)

                    if self.if_post_process:
                        # transpose data[self.predict_key] shape = (figure number, 3(RGB channel), 512, 512) to
                        # shape = (512, 512, figure number, 3(RGB channel))
                        img_data = ((data[self.predict_key].copy()).transpose(2, 3, 0, 1))[:, :, fig_num, :]
                        predict_label = self.post_processor(img_data, predict_label)

                    predict_label_list.append(predict_label)

                # list-to-array-to-list conversion for calculating the overall stat (each patient)
                predict_list = [np.asarray(predict_label_list)]
                gt_list = [np.asarray(gt_label_list)]

                # calculate physical volume using space directions info stored in .nrrd
                space_matrix = np.zeros((3, 3))
                space_matrix[0] = np.asarray(gt_nrrd[1]['space directions'][0]).astype('float32')
                space_matrix[1] = np.asarray(gt_nrrd[1]['space directions'][1]).astype('float32')
                space_matrix[2] = np.asarray(gt_nrrd[1]['space directions'][2]).astype('float32')
                # calculate voxel volume as the determinant of spacing matrix
                voxel_vol = np.linalg.det(space_matrix)

                # initialize ClassificationMetric class for each patient and update with ground truth/predict labels
                patient_metric = ClassificationMetric(cls_num=1, if_binary=True,
                                                      pos_cls_fusion=True)

                cls_metric.update(gt_list, predict_list, cls_label=1)

                patient_metric.update(gt_list, predict_list, cls_label=1)

                if patient_metric.tp[0] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = patient_metric.fp[0] / patient_metric.tp[0]

                self.result_df = self.result_df.append({'PatientID': PatientID,
                                                        'class': 'positive sample',
                                                        'threshold': thresh,
                                                        'tp_count': patient_metric.tp[0],
                                                        'tn_count': patient_metric.tn[0],
                                                        'fp_count': patient_metric.fp[0],
                                                        'fn_count': patient_metric.fn[0],
                                                        'accuracy': patient_metric.get_acc(cls_label=1),
                                                        'recall': patient_metric.get_rec(cls_label=1),
                                                        'precision': patient_metric.get_prec(cls_label=1),
                                                        'fp/tp': fp_tp,
                                                        'dice': patient_metric.get_dice(cls_label=1),
                                                        'gt_vol': patient_metric.get_gt_vol(cls_label=1),
                                                        'pred_vol': patient_metric.get_pred_vol(cls_label=1),
                                                        'gt_phys_vol/mm^3': patient_metric.get_gt_vol(
                                                            cls_label=1) * voxel_vol,
                                                        'pred_phys_vol/mm^3': patient_metric.get_pred_vol(
                                                            cls_label=1) * voxel_vol,
                                                        'phys_vol_diff/mm^3': patient_metric.get_gt_vol(
                                                            cls_label=1) * voxel_vol - patient_metric.get_pred_vol(
                                                            cls_label=1) * voxel_vol,
                                                        self.score_type: patient_metric.get_fscore(
                                                            cls_label=1, beta=self.fscore_beta)},
                                                       ignore_index=True)
                gt_tot_phys_vol[0] += patient_metric.get_gt_vol(cls_label=1) * voxel_vol
                pred_tot_phys_vol[0] += patient_metric.get_pred_vol(cls_label=1) * voxel_vol


            if cls_metric.tp[0] == 0:
                fp_tp = np.nan
            else:
                fp_tp = cls_metric.fp[0] / cls_metric.tp[0]

            self.result_df = self.result_df.append({'PatientID': 'total',
                                                    'class': 'positive sample',
                                                    'threshold': thresh,
                                                    'tp_count': cls_metric.tp[0],
                                                    'tn_count': cls_metric.tn[0],
                                                    'fp_count': cls_metric.fp[0],
                                                    'fn_count': cls_metric.fn[0],
                                                    'accuracy': cls_metric.get_acc(cls_label=1),
                                                    'recall': cls_metric.get_rec(cls_label=1),
                                                    'precision': cls_metric.get_prec(cls_label=1),
                                                    'fp/tp': fp_tp,
                                                    'dice': cls_metric.get_dice(cls_label=1),
                                                    'gt_vol': cls_metric.get_gt_vol(cls_label=1),
                                                    'pred_vol': cls_metric.get_pred_vol(cls_label=1),
                                                    'gt_phys_vol/mm^3': gt_tot_phys_vol[0],
                                                    'pred_phys_vol/mm^3': pred_tot_phys_vol[0],
                                                    'phys_vol_diff/mm^3': gt_tot_phys_vol[0] - pred_tot_phys_vol[0],
                                                    self.score_type: cls_metric.get_fscore(
                                                        cls_label=1, beta=self.fscore_beta),
                                                    'Patient_Number': self.patient_num},
                                                   ignore_index=True)
            self.result_df = self.result_df.append({'PatientID': 'average',
                                                    'class': 'positive sample',
                                                    'threshold': thresh,
                                                    'tp_count': cls_metric.tp[0]/self.patient_num,
                                                    'tn_count': cls_metric.tn[0]/self.patient_num,
                                                    'fp_count': cls_metric.fp[0]/self.patient_num,
                                                    'fn_count': cls_metric.fn[0]/self.patient_num,
                                                    'accuracy': cls_metric.get_acc(cls_label=1),
                                                    'recall': cls_metric.get_rec(cls_label=1),
                                                    'precision': cls_metric.get_prec(cls_label=1),
                                                    'fp/tp': fp_tp,
                                                    'dice': cls_metric.get_dice(cls_label=1),
                                                    'gt_vol': cls_metric.get_gt_vol(cls_label=1)/self.patient_num,
                                                    'pred_vol': cls_metric.get_pred_vol(cls_label=1)/self.patient_num,
                                                    'gt_phys_vol/mm^3': gt_tot_phys_vol[0]/self.patient_num,
                                                    'pred_phys_vol/mm^3': pred_tot_phys_vol[0]/self.patient_num,
                                                    'phys_vol_diff/mm^3': (gt_tot_phys_vol[0] - pred_tot_phys_vol[0])/self.patient_num,
                                                    self.score_type: cls_metric.get_fscore(
                                                        cls_label=1, beta=self.fscore_beta),
                                                    'Patient_Number': self.patient_num},
                                                   ignore_index=True)

            # find the optimal threshold
            if 'positive sample' not in self.opt_thresh:

                self.opt_thresh['positive sample'] = self.result_df.iloc[-1]

                self.opt_thresh['positive sample'].loc['threshold'] = thresh

            else:
                # we choose the optimal threshold corresponding to the one that gives the highest model score
                if self.result_df.iloc[-1][self.score_type] > self.opt_thresh['positive sample'][
                    self.score_type]:
                    self.opt_thresh['positive sample'] = self.result_df.iloc[-1]
                    self.opt_thresh['positive sample'].loc['threshold'] = thresh

        self.result_df = self.result_df.sort_values(['threshold', 'PatientID', 'class'])

        save_xlsx_json(self.result_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'binary-class_evaluation', 'optimal_threshold')

    # 二分类分割模型评分的轻量级版本，每张图只做一遍预测，用阈值筛选正类别,不管gt有多少类别，我们只关心检出(正样本全部归为一类,pos_cls_fusion=True)
    def binary_class_evaluation_light(self):
        cls_metric_list = []
        gt_tot_phys_vol = []
        pred_tot_phys_vol = []
        self.data_iter.reset()
        for thresh in self.conf_thresh:
            # reset data iterator, otherwise we cannot perform new iteration
            self.data_iter.reset()
            print ('threshold = %s' % thresh)
            # predict_data.shape() = [figure number, class number, 512, 512], gt_nrrd.shape() = [512, 512, figure number],
            # the pictures are arranged in the same order

            # initialize ClassificationMetric class for total stat (all patients) and update with ground truth/predict labels
            cls_metric_list.append(ClassificationMetric(cls_num=1, if_binary=True, pos_cls_fusion=True))
            gt_tot_phys_vol.append([0.])
            pred_tot_phys_vol.append([0.])

        self.patient_num = 0.

        for data in self.data_iter:
            if data is None:
                break
            if data[self.predict_key] is None:
                continue
            self.patient_num += 1.
            predict_data = self.predictor(data[self.predict_key])
            gt_nrrd = data[self.gt_key]
            PatientID = data[self.patient_key]
            print ('processing PatientID: %s' % PatientID)
            # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
            predict_data_cpy = predict_data.copy()
            # transpose predict_data_cpy to shape [512, 512, figure number, class number]
            predict_data_cpy = predict_data_cpy.transpose((2, 3, 0, 1))
            assert predict_data_cpy.shape[-1] == len(self.cls_name), 'the num of classes: %s in predict labels ' \
                                                                     'should equal that defined in brain/classname_labelname_mapping.xls: %s' % (
                                                                         predict_data_cpy.shape[-1],
                                                                         len(self.cls_name))
            predict_data_slice_cpy = predict_data[:, 0, :, :].copy()
            predict_data_slice_cpy = predict_data_slice_cpy.transpose((1, 2, 0))
            # check if the predict and ground truth labels have the same shape
            if not predict_data_slice_cpy.shape == gt_nrrd[0].shape:
                raise Exception("predict and ground truth labels must have the same shape")


            for index, thresh in enumerate(self.conf_thresh):
                predict_label_list = []
                gt_label_list = []
                for fig_num in range(predict_data_cpy.shape[-2]):
                    gt_label = gt_nrrd[0][:, :, fig_num]
                    gt_label_list.append(gt_label)
                    predict_score = predict_data_cpy[:, :, fig_num, :].copy()
                    for cls in range(predict_data_cpy.shape[-1]):
                        if self.cls_name[cls] == '__background__':
                            predict_score[:, :, cls] = 0.
                            continue
                        predict_score[predict_data_cpy[:, :, fig_num, cls] <= thresh] = 0.
                    # return the predict label as the index of the maximum value across all classes, the __background__
                    # class should be the first one, so if none of the positive class score is greater than zero, the
                    # pixel is classified as the background
                    predict_label = np.argmax(predict_score, axis=2)

                    if self.if_post_process:
                        # transpose data[self.predict_key] shape = (figure number, 3(RGB channel), 512, 512) to
                        # shape = (512, 512, figure number, 3(RGB channel))
                        img_data = ((data[self.predict_key].copy()).transpose(2, 3, 0, 1))[:, :, fig_num, :]
                        predict_label = self.post_processor(img_data, predict_label)

                    predict_label_list.append(predict_label)

                # list-to-array-to-list conversion for calculating the overall stat (each patient)
                predict_list = [np.asarray(predict_label_list)]
                gt_list = [np.asarray(gt_label_list)]

                # calculate physical volume using space directions info stored in .nrrd
                space_matrix = np.zeros((3, 3))
                space_matrix[0] = np.asarray(gt_nrrd[1]['space directions'][0]).astype('float32')
                space_matrix[1] = np.asarray(gt_nrrd[1]['space directions'][1]).astype('float32')
                space_matrix[2] = np.asarray(gt_nrrd[1]['space directions'][2]).astype('float32')
                # calculate voxel volume as the determinant of spacing matrix
                voxel_vol = np.linalg.det(space_matrix)

                # initialize ClassificationMetric class for each patient and update with ground truth/predict labels
                patient_metric = ClassificationMetric(cls_num=1, if_binary=True,
                                                      pos_cls_fusion=True)

                cls_metric_list[index].update(gt_list, predict_list, cls_label=1)

                patient_metric.update(gt_list, predict_list, cls_label=1)

                if patient_metric.tp[0] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = patient_metric.fp[0] / patient_metric.tp[0]

                self.result_df = self.result_df.append({'PatientID': PatientID,
                                                        'class': 'positive sample',
                                                        'threshold': thresh,
                                                        'tp_count': patient_metric.tp[0],
                                                        'tn_count': patient_metric.tn[0],
                                                        'fp_count': patient_metric.fp[0],
                                                        'fn_count': patient_metric.fn[0],
                                                        'accuracy': patient_metric.get_acc(cls_label=1),
                                                        'recall': patient_metric.get_rec(cls_label=1),
                                                        'precision': patient_metric.get_prec(cls_label=1),
                                                        'fp/tp': fp_tp,
                                                        'dice': patient_metric.get_dice(cls_label=1),
                                                        'gt_vol': patient_metric.get_gt_vol(cls_label=1),
                                                        'pred_vol': patient_metric.get_pred_vol(cls_label=1),
                                                        'gt_phys_vol/mm^3': patient_metric.get_gt_vol(
                                                            cls_label=1) * voxel_vol,
                                                        'pred_phys_vol/mm^3': patient_metric.get_pred_vol(
                                                            cls_label=1) * voxel_vol,
                                                        'phys_vol_diff/mm^3': patient_metric.get_gt_vol(
                                                            cls_label=1) * voxel_vol - patient_metric.get_pred_vol(
                                                            cls_label=1) * voxel_vol,
                                                        self.score_type: patient_metric.get_fscore(
                                                            cls_label=1, beta=self.fscore_beta)},
                                                       ignore_index=True)
                gt_tot_phys_vol[index][0] += patient_metric.get_gt_vol(cls_label=1) * voxel_vol
                pred_tot_phys_vol[index][0] += patient_metric.get_pred_vol(cls_label=1) * voxel_vol

        for index, thresh in enumerate(self.conf_thresh):
            if cls_metric_list[index].tp[0] == 0:
                fp_tp = np.nan
            else:
                fp_tp = cls_metric_list[index].fp[0] / cls_metric_list[index].tp[0]

            self.result_df = self.result_df.append({'PatientID': 'total',
                                                    'class': 'positive sample',
                                                    'threshold': thresh,
                                                    'tp_count': cls_metric_list[index].tp[0],
                                                    'tn_count': cls_metric_list[index].tn[0],
                                                    'fp_count': cls_metric_list[index].fp[0],
                                                    'fn_count': cls_metric_list[index].fn[0],
                                                    'accuracy': cls_metric_list[index].get_acc(cls_label=1),
                                                    'recall': cls_metric_list[index].get_rec(cls_label=1),
                                                    'precision': cls_metric_list[index].get_prec(cls_label=1),
                                                    'fp/tp': fp_tp,
                                                    'dice': cls_metric_list[index].get_dice(cls_label=1),
                                                    'gt_vol': cls_metric_list[index].get_gt_vol(cls_label=1),
                                                    'pred_vol': cls_metric_list[index].get_pred_vol(cls_label=1),
                                                    'gt_phys_vol/mm^3': gt_tot_phys_vol[index][0],
                                                    'pred_phys_vol/mm^3': pred_tot_phys_vol[index][0],
                                                    'phys_vol_diff/mm^3': gt_tot_phys_vol[index][0] - pred_tot_phys_vol[index][0],
                                                    self.score_type: cls_metric_list[index].get_fscore(
                                                        cls_label=1, beta=self.fscore_beta),
                                                    'Patient_Number': self.patient_num},
                                                   ignore_index=True)
            self.result_df = self.result_df.append({'PatientID': 'average',
                                                    'class': 'positive sample',
                                                    'threshold': thresh,
                                                    'tp_count': cls_metric_list[index].tp[0]/self.patient_num,
                                                    'tn_count': cls_metric_list[index].tn[0]/self.patient_num,
                                                    'fp_count': cls_metric_list[index].fp[0]/self.patient_num,
                                                    'fn_count': cls_metric_list[index].fn[0]/self.patient_num,
                                                    'accuracy': cls_metric_list[index].get_acc(cls_label=1),
                                                    'recall': cls_metric_list[index].get_rec(cls_label=1),
                                                    'precision': cls_metric_list[index].get_prec(cls_label=1),
                                                    'fp/tp': fp_tp,
                                                    'dice': cls_metric_list[index].get_dice(cls_label=1),
                                                    'gt_vol': cls_metric_list[index].get_gt_vol(cls_label=1)/self.patient_num,
                                                    'pred_vol': cls_metric_list[index].get_pred_vol(cls_label=1)/self.patient_num,
                                                    'gt_phys_vol/mm^3': gt_tot_phys_vol[index][0]/self.patient_num,
                                                    'pred_phys_vol/mm^3': pred_tot_phys_vol[index][0]/self.patient_num,
                                                    'phys_vol_diff/mm^3': (gt_tot_phys_vol[index][0] -
                                                                          pred_tot_phys_vol[index][0])/self.patient_num,
                                                    self.score_type: cls_metric_list[index].get_fscore(
                                                        cls_label=1, beta=self.fscore_beta),
                                                    'Patient_Number': self.patient_num},
                                                   ignore_index=True)

            # find the optimal threshold
            if 'positive sample' not in self.opt_thresh:

                self.opt_thresh['positive sample'] = self.result_df.iloc[-1]

                self.opt_thresh['positive sample'].loc['threshold'] = thresh

            else:
                # we choose the optimal threshold corresponding to the one that gives the highest model score
                if self.result_df.iloc[-1][self.score_type] > self.opt_thresh['positive sample'][
                    self.score_type]:
                    self.opt_thresh['positive sample'] = self.result_df.iloc[-1]
                    self.opt_thresh['positive sample'].loc['threshold'] = thresh

        self.result_df = self.result_df.sort_values(['threshold', 'PatientID', 'class'])

        save_xlsx_json(self.result_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'binary-class_evaluation', 'optimal_threshold')

class BrainSemanticSegEvaluatorOnline(object):
    '''
    This class takes lists of predicted data, ground truth data, raw image data and patient IDs to evaluate model performance.
    The current version includes functionality of:
        1. binary_class_contour_plot_single_thresh: take a single threshold to draw contour plot for binary-class segmentation,
            both predicted result and ground truth are plotted in the same image (shape = (512, 1024, 3)), with predicted result on the left
        2. binary_class_contour_plot_multi_thresh: take multiple thresholds to draw contour plot for binary-class segmentation,
            both predicted result and ground truth are plotted in the same image (shape = (512, 1024, 3)), with predicted result on the left
        3. multi_class_evaluation: take each class and regard all other classes as background, evaluate model performance,
            compatible with only one positive class
        4. binary_class_evaluation: take all positive classes and make them the same class (positive sample), evaluate
            model performance.
    Parameters
    ----------
    predict_data_list: list of predicted data, each element has shape = (figure num(batch size), 2 (cls num), 512, 512)
    gt_nrrd_list: list of ground truth data, each element has shape = (512, 512, figure num(batch size))
    img_nrrd_list: list of raw image data, each element = list: [ground truth label/mask, shape = (512, 512, figure num(batch size)), info of ct scan]
    patient_list: list of patient ID
    '''

    def __init__(self, cls_label_xls_path, predict_data_list, gt_nrrd_list, img_nrrd_list, patient_list,
                 img_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_contour'),
                 score_type='fscore', result_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_result'),
                 xlsx_name='BrainSemanticSegEvaluation.xlsx', json_name='BrainSemanticSegEvaluation',
                 conf_thresh=np.linspace(0.1, 0.9, num=9).tolist(),
                 fscore_beta=1., thresh=0.5
                 ):
        config = BrainConfig(cls_label_xls_path=cls_label_xls_path)
        self.predict_data_list = predict_data_list
        self.gt_nrrd_list = gt_nrrd_list
        self.img_nrrd_list = img_nrrd_list
        self.patient_list = patient_list
        self.img_save_dir = img_save_dir
        # config.CLASSES 包含background class,是模型预测的所有类别
        self.cls_name = config.CLASSES
        self.cls_dict = config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.result_df = pd.DataFrame(
            columns=['PatientID', 'class', 'threshold', 'tp_count', 'tn_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision', 'fp/tp', 'dice', 'gt_vol', 'pred_vol', 'gt_phys_vol/mm^3',
                     'pred_phys_vol/mm^3', 'phys_vol_diff/mm^3', self.score_type])
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.conf_thresh = conf_thresh
        self.cls_weights = config.CLASS_WEIGHTS
        self.fscore_beta = fscore_beta
        self.thresh = thresh

    def binary_class_contour_plot_single_thresh(self):
        predict_data_list = self.predict_data_list
        gt_nrrd_list = self.gt_nrrd_list
        img_nrrd_list = self.img_nrrd_list

        if not os.path.exists(self.img_save_dir):
            os.mkdir(self.img_save_dir)

        for predict_data, gt_nrrd, img_nrrd, PatientID in zip(predict_data_list, gt_nrrd_list, img_nrrd_list,
                                                              self.patient_list):

            print ('processing PatientID: %s' % PatientID)
            assert predict_data.shape[
                       1] == 2, 'the number of classes %s in predict labels should be 2 for binary classification' % (
                predict_data.shape[1])
            # transpose the predict_data to be shape = [512, 512, figure number]
            # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
            predict_data_cpy = predict_data[:, 1, :, :].copy()
            predict_data_cpy = predict_data_cpy.transpose((1, 2, 0))
            # dim of the raw image
            dim = [img_nrrd.shape[0], img_nrrd.shape[1]]

            # check if the predict and ground truth labels have the same shape
            if not predict_data_cpy.shape == gt_nrrd[0].shape:
                raise Exception("predict and ground truth labels must have the same shape")

            for fig_num in range(predict_data_cpy.shape[-1]):
                gt_img = img_nrrd[:, :, fig_num]
                # copy original img three times for RGB channels
                gt_img = np.repeat(gt_img[:, :, np.newaxis], axis=2, repeats=3)
                gt_img = window_convert(gt_img, 40, 80)
                gt_map = gt_nrrd[0][:, :, fig_num]
                predict_map = predict_data_cpy[:, :, fig_num]
                img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                lab = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                binary_pmap = np.array(predict_map > self.thresh, dtype=np.int)

                lab, _ = contour_and_draw(lab, gt_map)
                img, _ = contour_and_draw(img, binary_pmap)

                prob = predict_map[:, :, np.newaxis] * 255
                prob = np.array(np.repeat(prob, axis=2, repeats=3), dtype=np.uint8)

                to_show = np.zeros(shape=(dim[0], dim[1]*2, 3), dtype=np.uint8)
                to_show[:, :dim[1], :] = img
                to_show[:, dim[1]:, :] = lab

                cv2.imwrite(os.path.join(self.img_save_dir, PatientID + '-' + str(fig_num).zfill(2) + '.jpg'), to_show)

    def binary_class_contour_plot_multi_thresh(self):
        predict_data_list = self.predict_data_list
        gt_nrrd_list = self.gt_nrrd_list
        img_nrrd_list = self.img_nrrd_list

        if not os.path.exists(self.img_save_dir):
            os.mkdir(self.img_save_dir)

        for predict_data, gt_nrrd, img_nrrd, PatientID in zip(predict_data_list, gt_nrrd_list, img_nrrd_list,
                                                              self.patient_list):
            print ('processing PatientID: %s' % PatientID)
            assert predict_data.shape[
                       1] == 2, 'the number of classes %s in predict labels should be 2 for binary classification' % (
                predict_data.shape[1])
            # transpose the predict_data to be shape = [512, 512, figure number]
            # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
            predict_data_cpy = predict_data[:, 1, :, :].copy()
            predict_data_cpy = predict_data_cpy.transpose((1, 2, 0))
            # dim of the raw image
            dim = [img_nrrd.shape[0], img_nrrd.shape[1]]

            # check if the predict and ground truth labels have the same shape
            if not predict_data_cpy.shape == gt_nrrd[0].shape:
                raise Exception("predict and ground truth labels must have the same shape")

            for fig_num in range(predict_data_cpy.shape[-1]):
                gt_img = img_nrrd[:, :, fig_num]
                # copy original img three times for RGB channels
                gt_img = np.repeat(gt_img[:, :, np.newaxis], axis=2, repeats=3)
                gt_img = window_convert(gt_img, 40, 80)
                gt_map = gt_nrrd[0][:, :, fig_num]
                predict_map = predict_data_cpy[:, :, fig_num]
                img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                lab = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                lab, _ = contour_and_draw(lab, gt_map)
                # superimpose contour images for multiple thresholds
                for index, thresh in enumerate(self.conf_thresh):
                    binary_pmap = np.array(predict_map > thresh, dtype=np.int)

                    img, _ = contour_and_draw_rainbow(img, binary_pmap, color_range=len(self.conf_thresh),
                                                      color_num=index)

                    prob = predict_map[:, :, np.newaxis] * 255
                    prob = np.array(np.repeat(prob, axis=2, repeats=3), dtype=np.uint8)

                    to_show = np.zeros(shape=(dim[0], dim[1] * 2, 3), dtype=np.uint8)
                    to_show[:, :dim[1], :] = img
                    to_show[:, dim[1]:, :] = lab

                cv2.imwrite(os.path.join(self.img_save_dir, PatientID + '-' + str(fig_num).zfill(2) + '.jpg'),
                            to_show)

    # 多分类分割模型通用评分,用阈值筛选正类别，并在其中选取最大值作为one-hot label
    def multi_class_evaluation(self):
        predict_data_list = self.predict_data_list
        gt_nrrd_list = self.gt_nrrd_list

        for thresh in self.conf_thresh:
            print ('threshold = %s' % thresh)
            # predict_data.shape() = [figure number, class number, 512, 512], gt_nrrd.shape() = [512, 512, figure number],
            # the pictures are arranged in the same order

            # list to keep track of the total physical volume for each segmentation class
            # initialize ClassificationMetric class for total stat (all patients) and update with ground truth/predict labels
            cls_metric = ClassificationMetric(cls_num=len(self.cls_name) - 1, if_binary=True, pos_cls_fusion=False)
            gt_tot_phys_vol = [0. for _ in range(len(self.cls_name) - 1)]
            pred_tot_phys_vol = [0. for _ in range(len(self.cls_name) - 1)]

            for predict_data, gt_nrrd, PatientID in zip(predict_data_list, gt_nrrd_list, self.patient_list):
                print ('processing PatientID: %s' % PatientID)
                # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
                predict_data_cpy = predict_data.copy()
                # transpose predict_data_cpy to shape [512, 512, figure number, class number]
                predict_data_cpy = predict_data_cpy.transpose((2, 3, 0, 1))
                assert predict_data_cpy.shape[-1] == len(self.cls_name), 'the num of classes: %s in predict labels ' \
                                                                         'should equal that defined in brain/classname_labelname_mapping.xls: %s' % (
                                                                             predict_data_cpy.shape[-1],
                                                                             len(self.cls_name))
                predict_data_slice_cpy = predict_data[:, 0, :, :].copy()
                predict_data_slice_cpy = predict_data_slice_cpy.transpose((1, 2, 0))
                # check if the predict and ground truth labels have the same shape
                if not predict_data_slice_cpy.shape == gt_nrrd[0].shape:
                    raise Exception("predict and ground truth labels must have the same shape")

                predict_label_list = []
                gt_label_list = []
                for fig_num in range(predict_data_cpy.shape[-2]):
                    gt_label = gt_nrrd[0][:, :, fig_num]
                    gt_label_list.append(gt_label)
                    predict_score = predict_data_cpy[:, :, fig_num, :].copy()
                    for cls in range(predict_data_cpy.shape[-1]):
                        if self.cls_name[cls] == '__background__':
                            predict_score[:, :, cls] = 0.
                            continue
                        predict_score[predict_data_cpy[:, :, fig_num, cls] <= thresh] = 0.
                    # return the predict label as the index of the maximum value across all classes, the __background__
                    # class should be the first one, so if none of the positive class score is greater than zero, the
                    # pixel is classified as the background
                    predict_label = np.argmax(predict_score, axis=2)
                    predict_label_list.append(predict_label)

                # list-to-array-to-list conversion for calculating the overall stat (each patient)
                predict_list = [np.asarray(predict_label_list)]
                gt_list = [np.asarray(gt_label_list)]

                # calculate physical volume using space directions info stored in .nrrd
                space_matrix = np.zeros((3, 3))
                space_matrix[0] = np.asarray(gt_nrrd[1]['space directions'][0]).astype('float32')
                space_matrix[1] = np.asarray(gt_nrrd[1]['space directions'][1]).astype('float32')
                space_matrix[2] = np.asarray(gt_nrrd[1]['space directions'][2]).astype('float32')
                # calculate voxel volume as the determinant of spacing matrix
                voxel_vol = np.linalg.det(space_matrix)

                # initialize ClassificationMetric class for each patient and update with ground truth/predict labels
                patient_metric = ClassificationMetric(cls_num=len(self.cls_name) - 1, if_binary=True,
                                                      pos_cls_fusion=False)
                for cls_label in range(len(self.cls_name)):
                    if self.cls_name[cls_label] == '__background__':
                        continue

                    cls_metric.update(gt_list, predict_list, cls_label=cls_label)

                    patient_metric.update(gt_list, predict_list, cls_label=cls_label)

                    if patient_metric.tp[cls_label - 1] == 0:
                        fp_tp = np.nan
                    else:
                        fp_tp = patient_metric.fp[cls_label - 1] / patient_metric.tp[cls_label - 1]

                    self.result_df = self.result_df.append({'PatientID': PatientID,
                                                            'class': self.cls_name[cls_label],
                                                            'threshold': thresh,
                                                            'tp_count': patient_metric.tp[cls_label - 1],
                                                            'tn_count': patient_metric.tn[cls_label - 1],
                                                            'fp_count': patient_metric.fp[cls_label - 1],
                                                            'fn_count': patient_metric.fn[cls_label - 1],
                                                            'accuracy': patient_metric.get_acc(cls_label),
                                                            'recall': patient_metric.get_rec(cls_label),
                                                            'precision': patient_metric.get_prec(cls_label),
                                                            'fp/tp': fp_tp,
                                                            'dice': patient_metric.get_dice(cls_label),
                                                            'gt_vol': patient_metric.get_gt_vol(cls_label),
                                                            'pred_vol': patient_metric.get_pred_vol(cls_label),
                                                            'gt_phys_vol/mm^3': patient_metric.get_gt_vol(
                                                                cls_label) * voxel_vol,
                                                            'pred_phys_vol/mm^3': patient_metric.get_pred_vol(
                                                                cls_label) * voxel_vol,
                                                            'phys_vol_diff/mm^3': patient_metric.get_gt_vol(
                                                                cls_label) * voxel_vol - patient_metric.get_pred_vol(
                                                                cls_label) * voxel_vol,
                                                            self.score_type: patient_metric.get_fscore(
                                                                cls_label=cls_label, beta=self.fscore_beta)},
                                                           ignore_index=True)
                    gt_tot_phys_vol[cls_label - 1] += patient_metric.get_gt_vol(cls_label) * voxel_vol
                    pred_tot_phys_vol[cls_label - 1] += patient_metric.get_pred_vol(cls_label) * voxel_vol

            for cls_label in range(len(self.cls_name)):
                if self.cls_name[cls_label] == '__background__':
                    continue

                if cls_metric.tp[cls_label - 1] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = cls_metric.fp[cls_label - 1] / cls_metric.tp[cls_label - 1]

                self.result_df = self.result_df.append({'PatientID': 'total',
                                                        'class': self.cls_name[cls_label],
                                                        'threshold': thresh,
                                                        'tp_count': cls_metric.tp[cls_label - 1],
                                                        'tn_count': cls_metric.tn[cls_label - 1],
                                                        'fp_count': cls_metric.fp[cls_label - 1],
                                                        'fn_count': cls_metric.fn[cls_label - 1],
                                                        'accuracy': cls_metric.get_acc(cls_label),
                                                        'recall': cls_metric.get_rec(cls_label),
                                                        'precision': cls_metric.get_prec(cls_label),
                                                        'fp/tp': fp_tp,
                                                        'dice': cls_metric.get_dice(cls_label),
                                                        'gt_vol': cls_metric.get_gt_vol(cls_label),
                                                        'pred_vol': cls_metric.get_pred_vol(cls_label),
                                                        'gt_phys_vol/mm^3': gt_tot_phys_vol[cls_label - 1],
                                                        'pred_phys_vol/mm^3': pred_tot_phys_vol[cls_label - 1],
                                                        'phys_vol_diff/mm^3': gt_tot_phys_vol[cls_label - 1] -  pred_tot_phys_vol[cls_label - 1],
                                                        self.score_type: cls_metric.get_fscore(
                                                            cls_label=cls_label, beta=self.fscore_beta)},
                                                       ignore_index=True)

                # find the optimal threshold
                if self.cls_name[cls_label] not in self.opt_thresh:

                    self.opt_thresh[self.cls_name[cls_label]] = self.result_df.iloc[-1]

                    self.opt_thresh[self.cls_name[cls_label]].loc['threshold'] = thresh

                else:
                    # we choose the optimal threshold corresponding to the one that gives the highest model score
                    if self.result_df.iloc[-1][self.score_type] > self.opt_thresh[self.cls_name[cls_label]][
                        self.score_type]:
                        self.opt_thresh[self.cls_name[cls_label]] = self.result_df.iloc[-1]
                        self.opt_thresh[self.cls_name[cls_label]].loc['threshold'] = thresh

        self.result_df = self.result_df.sort_values(['threshold', 'PatientID', 'class'])

        save_xlsx_json(self.result_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'multi-class_evaluation', 'optimal_threshold')


    # 二分类分割模型评分，用阈值筛选正类别
    def binary_class_evaluation(self):
        predict_data_list = self.predict_data_list
        gt_nrrd_list = self.gt_nrrd_list

        for thresh in self.conf_thresh:
            print ('threshold = %s' % thresh)
            # predict_data.shape() = [figure number, class number, 512, 512], gt_nrrd.shape() = [512, 512, figure number],
            # the pictures are arranged in the same order
            gt_tot_phys_vol = [0.]
            pred_tot_phys_vol = [0.]
            # initialize ClassificationMetric class for total stat (all patients) and update with ground truth/predict labels
            cls_metric = ClassificationMetric(cls_num=1, if_binary=True, pos_cls_fusion=True)

            for predict_data, gt_nrrd, PatientID in zip(predict_data_list, gt_nrrd_list, self.patient_list):
                print ('processing PatientID: %s' % PatientID)
                # one has to make a copy of part of predict_data, otherwise it will implicitly convert float to int
                predict_data_cpy = predict_data.copy()
                # transpose predict_data_cpy to shape [512, 512, figure number, class number]
                predict_data_cpy = predict_data_cpy.transpose((2, 3, 0, 1))
                assert predict_data_cpy.shape[-1] == len(self.cls_name), 'the num of classes: %s in predict labels ' \
                                                                         'should equal that defined in brain/classname_labelname_mapping.xls: %s' % (
                                                                             predict_data_cpy.shape[-1],
                                                                             len(self.cls_name))
                predict_data_slice_cpy = predict_data[:, 0, :, :].copy()
                predict_data_slice_cpy = predict_data_slice_cpy.transpose((1, 2, 0))
                # check if the predict and ground truth labels have the same shape
                if not predict_data_slice_cpy.shape == gt_nrrd[0].shape:
                    raise Exception("predict and ground truth labels must have the same shape")

                predict_label_list = []
                gt_label_list = []
                for fig_num in range(predict_data_cpy.shape[-2]):
                    gt_label = gt_nrrd[0][:, :, fig_num]
                    gt_label_list.append(gt_label)
                    predict_score = predict_data_cpy[:, :, fig_num, :].copy()
                    for cls in range(predict_data_cpy.shape[-1]):
                        if self.cls_name[cls] == '__background__':
                            predict_score[:, :, cls] = 0.
                            continue
                        predict_score[predict_data_cpy[:, :, fig_num, cls] <= thresh] = 0.
                    # return the predict label as the index of the maximum value across all classes, the __background__
                    # class should be the first one, so if none of the positive class score is greater than zero, the
                    # pixel is classified as the background
                    predict_label = np.argmax(predict_score, axis=2)
                    predict_label_list.append(predict_label)

                # list-to-array-to-list conversion for calculating the overall stat (each patient)
                predict_list = [np.asarray(predict_label_list)]
                gt_list = [np.asarray(gt_label_list)]

                # calculate physical volume using space directions info stored in .nrrd
                space_matrix = np.zeros((3, 3))
                space_matrix[0] = np.asarray(gt_nrrd[1]['space directions'][0]).astype('float32')
                space_matrix[1] = np.asarray(gt_nrrd[1]['space directions'][1]).astype('float32')
                space_matrix[2] = np.asarray(gt_nrrd[1]['space directions'][2]).astype('float32')
                # calculate voxel volume as the determinant of spacing matrix
                voxel_vol = np.linalg.det(space_matrix)

                # initialize ClassificationMetric class for each patient and update with ground truth/predict labels
                patient_metric = ClassificationMetric(cls_num=1, if_binary=True,
                                                      pos_cls_fusion=True)

                cls_metric.update(gt_list, predict_list, cls_label=1)

                patient_metric.update(gt_list, predict_list, cls_label=1)

                if patient_metric.tp[0] == 0:
                    fp_tp = np.nan
                else:
                    fp_tp = patient_metric.fp[0] / patient_metric.tp[0]

                self.result_df = self.result_df.append({'PatientID': PatientID,
                                                        'class': 'positive sample',
                                                        'threshold': thresh,
                                                        'tp_count': patient_metric.tp[0],
                                                        'tn_count': patient_metric.tn[0],
                                                        'fp_count': patient_metric.fp[0],
                                                        'fn_count': patient_metric.fn[0],
                                                        'accuracy': patient_metric.get_acc(cls_label=1),
                                                        'recall': patient_metric.get_rec(cls_label=1),
                                                        'precision': patient_metric.get_prec(cls_label=1),
                                                        'fp/tp': fp_tp,
                                                        'dice': patient_metric.get_dice(cls_label=1),
                                                        'gt_vol': patient_metric.get_gt_vol(cls_label=1),
                                                        'pred_vol': patient_metric.get_pred_vol(cls_label=1),
                                                        'gt_phys_vol/mm^3': patient_metric.get_gt_vol(
                                                            cls_label=1) * voxel_vol,
                                                        'pred_phys_vol/mm^3': patient_metric.get_pred_vol(
                                                            cls_label=1) * voxel_vol,
                                                        'phys_vol_diff/mm^3': patient_metric.get_gt_vol(
                                                            cls_label=1) * voxel_vol - patient_metric.get_pred_vol(
                                                            cls_label=1) * voxel_vol,
                                                        self.score_type: patient_metric.get_fscore(
                                                            cls_label=1, beta=self.fscore_beta)},
                                                       ignore_index=True)
                gt_tot_phys_vol[0] += patient_metric.get_gt_vol(cls_label=1) * voxel_vol
                pred_tot_phys_vol[0] += patient_metric.get_pred_vol(cls_label=1) * voxel_vol

            if cls_metric.tp[0] == 0:
                fp_tp = np.nan
            else:
                fp_tp = cls_metric.fp[0] / cls_metric.tp[0]

            self.result_df = self.result_df.append({'PatientID': 'total',
                                                    'class': 'positive sample',
                                                    'threshold': thresh,
                                                    'tp_count': cls_metric.tp[0],
                                                    'tn_count': cls_metric.tn[0],
                                                    'fp_count': cls_metric.fp[0],
                                                    'fn_count': cls_metric.fn[0],
                                                    'accuracy': cls_metric.get_acc(cls_label=1),
                                                    'recall': cls_metric.get_rec(cls_label=1),
                                                    'precision': cls_metric.get_prec(cls_label=1),
                                                    'fp/tp': fp_tp,
                                                    'dice': cls_metric.get_dice(cls_label=1),
                                                    'gt_vol': cls_metric.get_gt_vol(cls_label=1),
                                                    'pred_vol': cls_metric.get_pred_vol(cls_label=1),
                                                    'gt_phys_vol/mm^3': gt_tot_phys_vol[0],
                                                    'pred_phys_vol/mm^3': pred_tot_phys_vol[0],
                                                    'phys_vol_diff/mm^3': gt_tot_phys_vol[0] - pred_tot_phys_vol[0],
                                                    self.score_type: cls_metric.get_fscore(
                                                        cls_label=1, beta=self.fscore_beta)},
                                                   ignore_index=True)

            # find the optimal threshold
            if 'positive sample' not in self.opt_thresh:

                self.opt_thresh['positive sample'] = self.result_df.iloc[-1]

                self.opt_thresh['positive sample'].loc['threshold'] = thresh

            else:
                # we choose the optimal threshold corresponding to the one that gives the highest model score
                if self.result_df.iloc[-1][self.score_type] > self.opt_thresh['positive sample'][
                    self.score_type]:
                    self.opt_thresh['positive sample'] = self.result_df.iloc[-1]
                    self.opt_thresh['positive sample'].loc['threshold'] = thresh

        self.result_df = self.result_df.sort_values(['threshold', 'PatientID', 'class'])

        save_xlsx_json(self.result_df, self.opt_thresh, self.result_save_dir, self.xlsx_name, self.json_name,
                       'binary-class_evaluation', 'optimal_threshold')


class BrainSemanticSegEvaluatorOffline(BrainSemanticSegEvaluatorOnline):
    '''
    This class is decoupled from model inference stage. It reads predicted data, ground truth data, raw image data and patient IDs
    from designated directory to evaluate model performance. All functionality is inherited from BrainSemanticSegEvaluatorOnline.
    Parameters
    ----------
    data_dir: directory of predicted data
    data_type: type of predicted data, currently supports only npy
    gt_dir: directory of ground truth data, nrrd by default
    img_dir: directory of raw image data, nrrd by default
    '''

    def __init__(self, cls_label_xls_path, data_dir, data_type, gt_dir, img_dir, img_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_contour'),
                 score_type='fscore', result_save_dir=os.path.join(os.getcwd(), 'BrainSemanticSegEvaluation_result'),
                 xlsx_name ='BrainSemanticSegEvaluation.xlsx', json_name='BrainSemanticSegEvaluation',
                 conf_thresh=np.linspace(0.1, 0.9, num=9).tolist(), fscore_beta=1., thresh=0.5
                 ):
        config = BrainConfig(cls_label_xls_path=cls_label_xls_path)
        assert os.path.isdir(data_dir), 'must initialize it with a valid directory of segmentation data'
        self.data_dir = data_dir
        self.data_type = data_type
        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.img_save_dir = img_save_dir
        # config.CLASSES 包含background class,是模型预测的所有类别
        self.cls_name = config.CLASSES
        self.cls_dict = config.CLASS_DICT
        self.score_type = score_type
        self.opt_thresh = {}

        self.result_df = pd.DataFrame(
            columns=['PatientID', 'class', 'threshold', 'tp_count', 'tn_count', 'fp_count', 'fn_count',
                     'accuracy', 'recall', 'precision',
                     'fp/tp', 'gt_vol', 'pred_vol', 'gt_phys_vol/mm^3', 'pred_phys_vol/mm^3', self.score_type])
        self.result_save_dir = result_save_dir
        self.xlsx_name = xlsx_name
        self.json_name = json_name
        self.conf_thresh = conf_thresh
        self.cls_weights = config.CLASS_WEIGHTS
        self.fscore_beta = fscore_beta
        self.thresh = thresh
        self.patient_list = []

        predict_data_list, gt_nrrd_list, img_nrrd_list = self.load_data()
        super(BrainSemanticSegEvaluatorOffline, self).__init__(cls_label_xls_path=cls_label_xls_path,
                                                               predict_data_list=predict_data_list,
                                                               gt_nrrd_list=gt_nrrd_list,
                                                               img_nrrd_list=img_nrrd_list,
                                                               patient_list=self.patient_list,
                                                               conf_thresh=self.conf_thresh,
                                                               fscore_beta=self.fscore_beta,
                                                               thresh=self.thresh)
    def load_data(self):
        '''
        data-storing convention:
        predict data:
        data_dir/
        ├── patient_id_1/
        |   ├─────────── patient_id_1_predict.npy
        |   ├─────────── patient_id_1_predict.npy
        ├── patient_id_2/
        ├── patient_id_3/

        ground truth label:
        gt_dir/
        ├── patient_id_1/
        |   ├─────────── patient_id_1_gt.nrrd
        |   ├─────────── patient_id_1_gt.nrrd
        ├── patient_id_2/
        ├── patient_id_3/

        image data:
        data_dir/
        ├── patient_id_1/
        |   ├─────────── patient_id_1_img.npy
        |   ├─────────── patient_id_1_img.npy
        ├── patient_id_2/
        ├── patient_id_3/

        :return:
        predict_npy_list: list of predict class score, default type: float32, default shape: [figure number, class number, 512, 512]
        (background class number = 0), softmax value in range [0, 1]
        gt_nrrd_list: list of ground truth nrrd mask, default type: int16, default shape: [512, 512, figure number]
        img_nrrd_list: list of nrrd image, default type: int16, default shape: [512, 512, figure number]
        '''
        predict_data_list = []
        gt_nrrd_list = []
        img_nrrd_list = []
        for PatientID in os.listdir(self.data_dir):
            if self.data_type == 'npy':
                predict_npy_path = os.path.join(self.data_dir, PatientID, PatientID + '_predict.npy')
                try:
                    predict_data = np.load(predict_npy_path)
                except:
                    print ("broken directory structure, maybe no predict npy file found: %s" % predict_npy_path)
                    raise NameError
            else:
                # to be implemented: HDF5 format data
                raise NotImplementedError
            predict_data_list.append(predict_data)

            gt_nrrd_path = os.path.join(self.gt_dir, PatientID, PatientID + '_gt.nrrd')

            try:
                gt_nrrd = nrrd.read(gt_nrrd_path)
            except:
                print ("broken directory structure, maybe no ground truth nrrd file found: %s" % gt_nrrd_path)
                raise NameError

            gt_nrrd_list.append(gt_nrrd)

            # check if the predict and ground truth labels have the same number of pictures
            if not predict_data.shape[0] == gt_nrrd[0].shape[-1]:
                raise Exception("predict and ground truth labels do not contain the same number of pictures")

            img_nrrd_path = os.path.join(self.img_dir, PatientID, PatientID + '_img.nrrd')

            try:
                img_nrrd = nrrd.read(img_nrrd_path)[0]
            except:
                print ("broken directory structure, maybe no image nrrd file found: %s" % img_nrrd_path)
                raise NameError

            img_nrrd_list.append(img_nrrd)
            self.patient_list.append(PatientID)
        return predict_data_list, gt_nrrd_list, img_nrrd_list


