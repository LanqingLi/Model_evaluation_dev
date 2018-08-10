import model_eval.lung.evaluator as evaluator
import argparse
import numpy as np

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
    parser.add_argument('--result_save_dir',
                        help='dir for saving xlsx',
                        default='./LungNoduleEvaluation_result',
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
                        help='name of generated .xlsx',
                        default='LungNoduleEvaluation.xlsx',
                        type=str)
    parser.add_argument('--json_name',
                        help='name of generated json file, no postfix',
                        default='LungNoduleEvaluation',
                        type=str)
    parser.add_argument('--gt_anno_dir',
                        help='ground truth anno stored dir',
                        default='./anno',
                        type=str)
    parser.add_argument('--multi_class',
                        help='multi-class evaluation', action='store_true')
    parser.add_argument('--nodule_threshold',
                        help='filter nodule instead of boxes with threshold', action='store_true')
    parser.add_argument('--nodule_json',
                        help='whether to generate _nodule.json which contains matched nodules information', action='store_true')
    parser.add_argument('--score_type',
                        help='type of model score',
                        default='F_score',
                        type=str)
    parser.add_argument('--clustering_test',
                        help='evaluate in terms of clustering metric', action='store_true')
    parser.add_argument('--nodule_cls',
                        help='evaluate with specified nodule class', action='store_true')
    parser.add_argument('--thickness_thresh',
                        help='threshold for filtering nodules with thickness greater or equal to certain integer',
                        default= 0,
                        type=int)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()

    model_eval = evaluator.LungNoduleEvaluatorOffline(data_dir=args.data_dir,
                                            data_type=args.data_type,
                                            anno_dir=args.gt_anno_dir,
                                            score_type=args.score_type,
                                            result_save_dir=args.result_save_dir,
                                            xlsx_name=args.xlsx_name,
                                            json_name=args.json_name,
                                            if_nodule_threshold=args.nodule_threshold,
                                            if_nodule_json=args.nodule_json,
                                            thickness_thresh=args.thickness_thresh,
                                            conf_thresh=np.linspace(0.1, 0.9, num=9).tolist())

    if model_eval.if_nodule_json:
        model_eval.generate_df_nodules_to_json()
        if model_eval.thickness_thresh > 0:
            model_eval.nodule_thickness_filter()
        exit()

    if args.multi_class:
        if not model_eval.if_nodule_threshold:
            model_eval.multi_class_evaluation()
        else:
            model_eval.multi_class_evaluation_nodule_threshold()
        print model_eval.opt_thresh


    else:
        if not model_eval.if_nodule_threshold:
            model_eval.binary_class_evaluation()
        else:
            model_eval.binary_class_evaluation_nodule_threshold()