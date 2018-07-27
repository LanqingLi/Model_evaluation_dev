# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import bisect

# 读入evaluator生成好的.xlsx并画出各个种类结节的PR曲线，横轴为recall, 纵轴为precision
def RP_plot(xlsx_save_dir, xlsx_name, sheet_name, if_AUC=True, xmin = 0., xmax = 1.):
    evaluation_df = pd.read_excel(os.path.join(xlsx_save_dir, xlsx_name), sheet_name=sheet_name)
    class_name = []
    for cls in evaluation_df['nodule_class'].tolist():
        if cls not in class_name:
            class_name.append(cls)
    # we use the Tableau Colors from the 'T10' categorical palette, for up to ten classes
    for index, cls in enumerate(class_name):
        evaluation_df = evaluation_df.sort_values(['nodule_class', 'recall'])
        plt.plot(evaluation_df['recall'][evaluation_df['nodule_class'] == cls],
                     evaluation_df['precision'][evaluation_df['nodule_class'] == cls], linestyle = '-',
                     label = '%s' %(cls), color = 'C%s' %index )
        recall = evaluation_df['recall'][evaluation_df['nodule_class'] == cls].tolist()
        precision = evaluation_df['precision'][evaluation_df['nodule_class'] == cls].tolist()
        print len(recall), len(precision)
        # recall.reverse()
        # precision.reverse()
        x = []
        y = []
        for i in range(len(precision)):
            for j in range(len(precision) - i - 1):
                precision[i] = max(precision[i], precision[i+j+1])
            # if i == 0:
            #     x.append([0, recall[i]])
            #     y.append([precision[i], precision[i]])
            # else:
            #     x.append([recall[i-1], recall[i]])
            #     y.append([precision[i], precision[i]])
            #     if i < len(precision) - 1:
            #         x.append([recall[i], recall[i]])
            #         y.append([precision[i], precision[i+1]])

            if i == 0:
                x.append(0)
                x.append(recall[i])
                y.append(precision[i])
                y.append(precision[i])
            else:
                x.append(recall[i-1])
                x.append(recall[i])
                y.append(precision[i])
                y.append(precision[i])
        print cls
        if if_AUC:
            area = cal_AUC(x, y, xmin, xmax)
        plt.plot(x, y, linestyle = '--', label = '%s, AUC(%s)' %(cls, area), color = 'C%s' %index)
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')

    plt.ylim([0.0, 1.0])
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.savefig('RP_%s, AUC xrange = [%s, %s].png' %(sheet_name, xmin, xmax), dpi = 900)
    plt.show()
    plt.draw()

def cal_AUC(xlist, ylist, xmin, xmax):
    """
    compute the area under curve (AUC) for a curve whose x labels are sorted in increasing order


    :param xlist:
    :param ylist:
    :param xmin:
    :param xmax:
    :return:
    """
    # xlist should be a sorted list in increasing order
    assert sorted(xlist) == xlist, 'xlist must be sorted in increasing order'
    assert xmax >= xmin, 'xmax must be greater than or equal to xmin'
    le = bisect.bisect_right(xlist, xmin) - 1
    re = bisect.bisect_left(xlist, xmax) - 1
    print le, re
    area = 0.
    if le == re:
        return ylist[le] * (xmax - xmin)
    for i in range(le, re+1):
        if i == le:
            area += ylist[le] * (xlist[le] - xmin)
        elif i == re:
            area += ylist[re] * (xmax - xlist[re-1])
        else:
            area += ylist[i] * (xlist[i] - xlist[i-1])
    return area


if __name__ == '__main__':
    xlsx_save_dir = '/mnt/data2/model_evaluation/excel_result'
    xlsx_name = 'LungModelEvaluation_multi_class_avg_310_find_nodules.xlsx'
    sheet_name = 'multi-class evaluation'
    RP_plot(xlsx_save_dir, xlsx_name, sheet_name, xmin=0., xmax=1)