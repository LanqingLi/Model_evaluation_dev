import brain.evaluator as evaluator
import argparse
from brain.config import config

def parse_args():
    parser = argparse.ArgumentParser(description='Infervision auto test')
    parser.add_argument('--data_dir',
                        help='predict result stored dir, .npy by default',
                        default='./predict',
                        type=str)
    parser.add_argument('--data_type',
                        help='type of data that store prediction result',
                        default='npy',
                        type=str
                        )
    parser.add_argument('--xlsx_save_dir',
                        help='dir for saving xlsx',
                        default='./excel_result',
                        type=str)
    parser.add_argument('--xlsx_name',
                        help='name of xlsx',
                        default='result.xlsx',
                        type=str)
    parser.add_argument('--img_dir',
                        help='directory for ground truth images',
                        default='/img',
                        type=str)
    parser.add_argument('--img_save_dir',
                        help='dir for saving contour images',
                        default='./img_compare',
                        type=str)
    parser.add_argument('--draw',
                        help='whether to draw contour images for gt and predict comparison',
                        action='store_true')
    parser.add_argument('--multi_thresh',
                        help='whether to superimpose contour images for multiple thresholds',
                        action='store_true')
    parser.add_argument('--gt_dir',
                        help='ground truth label stored dir',
                        default='./gt',
                        type=str)
    parser.add_argument('--multi_class',
                        help='multi-class evaluation', action='store_true')
    parser.add_argument('--score_type',
                        help='type of model score',
                        default='F_score',
                        type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    semantic_seg_eval = evaluator.BrainSemanticSegEvaluatorOffline(gt_dir=args.gt_dir,
                                                                   data_type=args.data_type,
                                                                   data_dir=args.data_dir,
                                                                   img_dir=args.img_dir)

    if args.draw:
        if args.multi_thresh:
            semantic_seg_eval.binary_class_contour_plot_multi_thresh()
        else:
            semantic_seg_eval.binary_class_contour_plot_single_thresh(config.THRESH)

    elif args.multi_class:
        semantic_seg_eval.multi_class_evaluation()

    else:
        semantic_seg_eval.binary_class_evaluation()


