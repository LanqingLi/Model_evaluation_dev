import model_eval.lung.evaluator as evaluator
import argparse
from lung.config import config

def parse_args():
    parser = argparse.ArgumentParser(description='Infervision auto test')
    parser.add_argument('--data_dir',
                        help='predict result stored dir, .npy by default',
                        default='./json_for_auto_test',
                        type=str)
    parser.add_argument('--data_type',
                        help='type of data that store prediction result',
                        default='json',
                        type=str
                        )
    parser.add_argument('--xlsx_save_dir',
                        help='dir for saving xlsx',
                        default='./excel_result',
                        type=str)
    parser.add_argument('--image_dir',
                        help='directory of ct we need to predict',
                        default='./dcm',
                        type=str)
    parser.add_argument('--image_save_dir',
                        help='dir for saving FP FN TP pictures',
                        default='./auto_test/NMS',
                        type=str)
    parser.add_argument('--dicom',
                        help='process dicom scan', action='store_true')
    parser.add_argument('--norm',
                        help='if normalize image pixel values to (0, 255)', action='store_true')
    parser.add_argument('--windowshift',
                        help='if apply intensity window to images', action='store_true')
    parser.add_argument('--save_img',
                        help='if store FP FN TP pictures', action='store_true')
    parser.add_argument('--xlsx_name',
                        help='name of xlsx',
                        default='result.xlsx',
                        type=str)
    parser.add_argument('--gt_anno_dir',
                        help='ground truth anno stored dir',
                        default='./anno',
                        type=str)
    parser.add_argument('--multi_class',
                        help='multi-class evaluation', action='store_true')
    parser.add_argument('--score_type',
                        help='type of model score',
                        default='F_score',
                        type=str)
    parser.add_argument('--clustering_test',
                        help='evaluate in terms of clustering metric', action='store_true')
    parser.add_argument('--nodule_cls',
                        help='evaluate with specified nodule class', action='store_true')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    find_nodules_eval = evaluator.FindNodulesEvaluator(gt_anno_dir=args.gt_anno_dir, if_nodule_cls=args.nodule_cls,
                                                       same_box_threshold_gt=config.FIND_NODULES.SAME_BOX_THRESHOLD_PRED,
                                                       score_threshold_gt=config.FIND_NODULES.SCORE_THRESHOLD_PRED,
                                                       z_threshold_gt=config.CLASS_Z_THRESHOLD_PRED)

    if not args.clustering_test:
        # test nodule count, no nodule cls, all set to 'nodule' by default:
        find_nodules_eval.evaluation_without_nodule_cls()

    else:
    # test with ground truth nodule num, such that we can quantify the result in terms of clustering metrics (see common/custom_metric.ClusteringMetric)
        find_nodules_eval.evaluation_with_nodule_num()

