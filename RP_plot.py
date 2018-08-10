from model_eval.tools.plot import RP_plot_xlsx, RP_plot_json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Infervision auto test')
    parser.add_argument('--json',
                        help='whether to generate RP plot from .json',
                        action='store_true')
    parser.add_argument('--AUC',
                        help='whether to calculate AUC of the curve',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    xlsx_save_dir = '/mnt/data2/model_evaluation_dev/LungNoduleEvaluation_result/multi_class_evaluation_demo'
    xlsx_name = 'LungNoduleEvaluation.xlsx'
    json_save_dir = '/mnt/data2/model_evaluation_dev/LungNoduleEvaluation_result/multi_class_evaluation_demo'
    json_name = 'LungNoduleEvaluation_multi-class_evaluation.json'
    sheet_name = 'multi-class_evaluation'
    xmin = 0.
    xmax = 1.
    args = parse_args()
    if args.json:
        RP_plot_json(json_save_dir=json_save_dir, json_name=json_name, sheet_name=sheet_name, xmin=xmin, xmax=xmax)
    else:
        RP_plot_xlsx(xlsx_save_dir=xlsx_save_dir, xlsx_name=xlsx_name, sheet_name=sheet_name, xmin=xmin, xmax=xmax)