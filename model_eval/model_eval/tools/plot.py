# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import bisect

# 读入evaluator生成好的.xlsx并画出各个种类结节的PR曲线，横轴为recall, 纵轴为precision
def RP_plot_xlsx(xlsx_save_dir, xlsx_name, sheet_name, if_AUC=True, xmin = 0., xmax = 1., cls_key = 'class'):
    evaluation_df = pd.read_excel(os.path.join(xlsx_save_dir, xlsx_name), sheet_name=sheet_name)
    class_name = []
    for cls in evaluation_df[cls_key].tolist():
        if cls not in class_name:
            class_name.append(cls)
    # we use the Tableau Colors from the 'T10' categorical palette, for up to ten classes
    for index, cls in enumerate(class_name):
        evaluation_df = evaluation_df.sort_values([cls_key, 'threshold'], ascending=False)
        plt.plot(evaluation_df['recall'][evaluation_df[cls_key] == cls],
                     evaluation_df['precision'][evaluation_df[cls_key] == cls], linestyle = '-',
                     label = '%s' %(cls), color = 'C%s' %index )
        recall = evaluation_df['recall'][evaluation_df[cls_key] == cls].tolist()
        precision = evaluation_df['precision'][evaluation_df[cls_key] == cls].tolist()
        print len(recall), len(precision)
        x = []
        y = []
        for i in range(len(precision)):
            for j in range(len(precision) - i - 1):
                precision[i] = max(precision[i], precision[i+j+1])
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
                if i == len(precision) - 1:
                    x.append(1)
                    y.append(precision[i])
        # if the recall values are all None except the first item 0 and the last item 1, quit drawing for the corresponding class
        if x.count(None) == len(x) - 2:
            continue
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

# 读入evaluator生成好的.json并画出各个种类结节的PR曲线，横轴为recall, 纵轴为precision
def RP_plot_json(json_save_dir, json_name, sheet_name, if_AUC=True, xmin = 0., xmax = 1., cls_key = 'class'):
    evaluation_df = pd.read_json(os.path.join(json_save_dir, json_name)).T
    class_name = []
    for cls in evaluation_df[cls_key].tolist():
        if cls not in class_name:
            class_name.append(cls)
    # we use the Tableau Colors from the 'T10' categorical palette, for up to ten classes
    for index, cls in enumerate(class_name):
        evaluation_df = evaluation_df.sort_values([cls_key, 'threshold'], ascending=False)
        print evaluation_df
        plt.plot(evaluation_df['recall'][evaluation_df[cls_key] == cls],
                     evaluation_df['precision'][evaluation_df[cls_key] == cls], linestyle = '-',
                     label = '%s' %(cls), color = 'C%s' %index )
        recall = evaluation_df['recall'][evaluation_df[cls_key] == cls].tolist()
        precision = evaluation_df['precision'][evaluation_df[cls_key] == cls].tolist()
        print len(recall), len(precision)
        x = []
        y = []
        for i in range(len(precision)):
            for j in range(len(precision) - i - 1):
                precision[i] = max(precision[i], precision[i+j+1])

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
                if i == len(precision) - 1:
                    x.append(1)
                    y.append(precision[i])
        # if the recall values are all None except the first item 0 and the last item 1, quit drawing for the corresponding class
        if x.count(None) == len(x) - 2:
            continue
        if if_AUC:
            print x
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

# 读入evaluator生成好的.xlsx并画出各个种类结节的PR曲线，横轴为recall, 纵轴为precision
def RP_plot_xlsx_tot(xlsx_save_dir, xlsx_name, sheet_name, if_AUC=True, xmin = 0., xmax = 1., cls_key = 'class'):
    evaluation_df = pd.read_excel(os.path.join(xlsx_save_dir, xlsx_name), sheet_name=sheet_name)
    class_name = []
    for cls in evaluation_df[cls_key].tolist():
        if cls not in class_name:
            class_name.append(cls)
    # we use the Tableau Colors from the 'T10' categorical palette, for up to ten classes
    for index, cls in enumerate(class_name):
        evaluation_df = evaluation_df.sort_values([cls_key, 'threshold'], ascending=False)
        evaluation_df = evaluation_df[(evaluation_df[cls_key] == cls) * (evaluation_df['PatientID'] == 'total')]
        plt.plot(evaluation_df['recall'],
                     evaluation_df['precision'], linestyle = '-',
                     label = '%s' %(cls), color = 'C%s' %index )
        recall = evaluation_df['recall'].tolist()
        precision = evaluation_df['precision'].tolist()
        print len(recall), len(precision)
        x = []
        y = []
        for i in range(len(precision)):
            for j in range(len(precision) - i - 1):
                precision[i] = max(precision[i], precision[i+j+1])
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
                if i == len(precision) - 1:
                    x.append(1)
                    y.append(precision[i])
        # if the recall values are all None except the first item 0 and the last item 1, quit drawing for the corresponding class
        if x.count(None) == len(x) - 2:
            continue
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

# 读入evaluator生成好的.json并画出各个种类结节的PR曲线，横轴为recall, 纵轴为precision
def RP_plot_json_tot(json_save_dir, json_name, sheet_name, if_AUC=True, xmin = 0., xmax = 1., cls_key = 'class'):
    evaluation_df = pd.read_json(os.path.join(json_save_dir, json_name)).T
    class_name = []
    for cls in evaluation_df[cls_key].tolist():
        if cls not in class_name:
            class_name.append(cls)
    # we use the Tableau Colors from the 'T10' categorical palette, for up to ten classes
    for index, cls in enumerate(class_name):
        evaluation_df = evaluation_df.sort_values([cls_key, 'threshold'], ascending=False)
        evaluation_df = evaluation_df[(evaluation_df[cls_key] == cls) * (evaluation_df['PatientID'] == 'total')]
        plt.plot(evaluation_df['recall'],
                     evaluation_df['precision'], linestyle = '-',
                     label = '%s' %(cls), color = 'C%s' %index )
        recall = evaluation_df['recall'].tolist()
        precision = evaluation_df['precision'].tolist()
        print len(recall), len(precision)
        x = []
        y = []
        for i in range(len(precision)):
            for j in range(len(precision) - i - 1):
                precision[i] = max(precision[i], precision[i+j+1])

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
                if i == len(precision) - 1:
                    x.append(1)
                    y.append(precision[i])
        # if the recall values are all None except the first item 0 and the last item 1, quit drawing for the corresponding class
        if x.count(None) == len(x) - 2:
            continue
        if if_AUC:
            print x
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

# 读入evaluator生成好的.xlsx并画出各个种类结节的ROC曲线，横轴为false positive rate, 纵轴为true positive rate (recall)
def ROC_plot_xlsx(xlsx_save_dir, xlsx_name, sheet_name, if_AUC=True, xmin=0., xmax=1., cls_key='class'):
    evaluation_df = pd.read_excel(os.path.join(xlsx_save_dir, xlsx_name), sheet_name=sheet_name)
    class_name = []
    for cls in evaluation_df[cls_key].tolist():
        if cls not in class_name:
            class_name.append(cls)
    # we use the Tableau Colors from the 'T10' categorical palette, for up to ten classes
    for index, cls in enumerate(class_name):
        evaluation_df = evaluation_df.sort_values([cls_key, 'threshold'], ascending=False)
        label = (evaluation_df[cls_key] == cls) * (evaluation_df['PatientID'] == 'total')
        x_data = evaluation_df['fp_count'][label] / (evaluation_df['fp_count'][label] + evaluation_df['tn_count'][label])
        y_data = evaluation_df['recall'][label]
        plt.plot(x_data, y_data, linestyle='-', label='%s' % (cls), color='C%s' % index)
        fp_rate = x_data.tolist()
        tp_rate = y_data.tolist()
        tp_rate.reverse()
        print len(fp_rate), len(tp_rate)
        x = []
        y = []
        for i in range(len(tp_rate)):
            for j in range(len(tp_rate) - i - 1):
                tp_rate[i] = max(tp_rate[i], tp_rate[i + j + 1])
            if i == 0:
                x.append(0)
                x.append(fp_rate[i])
                y.append(tp_rate[i])
                y.append(tp_rate[i])
            else:
                x.append(fp_rate[i - 1])
                x.append(fp_rate[i])
                y.append(tp_rate[i])
                y.append(tp_rate[i])
                if i == len(tp_rate) - 1:
                    x.append(1)
                    y.append(tp_rate[i])
        # if the recall values are all None except the first item 0, quit drawing for the corresponding class
        if x.count(None) == len(x) - 1:
            continue
        if if_AUC:
            print x
            y.reverse()
            area = cal_AUC(x, y, xmin, xmax)
        plt.plot(x, y, linestyle='--', label='%s, AUC(%s)' % (cls, area), color='C%s' % index)

    plt.plot(np.linspace(0., 1., num=100), np.linspace(0., 1., num=100), linestyle='--', color=(0,1,0))
    plt.xlim([0.0, 1.0])
    plt.xlabel('FP Rate')

    plt.ylim([0.0, 1.0])
    plt.ylabel('TP Rate')
    plt.legend()
    plt.grid()
    plt.savefig('ROC_%s, AUC xrange = [%s, %s].png' % (sheet_name, xmin, xmax), dpi=900)
    plt.show()
    plt.draw()


def ROC_plot_json(json_save_dir, json_name, sheet_name, if_AUC=True, xmin = 0., xmax = 1., cls_key = 'class'):
    evaluation_df = pd.read_json(os.path.join(json_save_dir, json_name)).T
    class_name = []
    for cls in evaluation_df[cls_key].tolist():
        if cls not in class_name:
            class_name.append(cls)
    # we use the Tableau Colors from the 'T10' categorical palette, for up to ten classes
    for index, cls in enumerate(class_name):
        evaluation_df = evaluation_df.sort_values([cls_key, 'threshold'], ascending=False)
        label = (evaluation_df[cls_key] == cls) * (evaluation_df['PatientID'] == 'total')
        x_data = evaluation_df['fp_count'][label] / (
        evaluation_df['fp_count'][label] + evaluation_df['tn_count'][label])
        y_data = evaluation_df['recall'][label]
        plt.plot(x_data, y_data, linestyle='-', label='%s' % (cls), color='C%s' % index)
        fp_rate = x_data.tolist()
        tp_rate = y_data.tolist()
        tp_rate.reverse()
        print len(fp_rate), len(tp_rate)
        x = []
        y = []
        for i in range(len(tp_rate)):
            for j in range(len(tp_rate) - i - 1):
                tp_rate[i] = max(tp_rate[i], tp_rate[i + j + 1])
            if i == 0:
                x.append(0)
                x.append(fp_rate[i])
                y.append(tp_rate[i])
                y.append(tp_rate[i])
            else:
                x.append(fp_rate[i - 1])
                x.append(fp_rate[i])
                y.append(tp_rate[i])
                y.append(tp_rate[i])
                if i == len(tp_rate) - 1:
                    x.append(1)
                    y.append(tp_rate[i])
        # if the recall values are all None except the first item 0, quit drawing for the corresponding class
        if x.count(None) == len(x) - 1:
            continue
        if if_AUC:
            print x
            y.reverse()
            area = cal_AUC(x, y, xmin, xmax)
        plt.plot(x, y, linestyle='--', label='%s, AUC(%s)' % (cls, area), color='C%s' % index)

    plt.plot(np.linspace(0., 1., num=100), np.linspace(0., 1., num=100), linestyle='--', color=(0, 1, 0))
    plt.xlim([0.0, 1.0])
    plt.xlabel('FP Rate')

    plt.ylim([0.0, 1.0])
    plt.ylabel('TP Rate')
    plt.legend()
    plt.grid()
    plt.savefig('ROC_%s, AUC xrange = [%s, %s].png' % (sheet_name, xmin, xmax), dpi=900)
    plt.show()
    plt.draw()

def cal_AUC(xlist, ylist, xmin, xmax):
    """
    compute the area under curve (AUC) for a curve whose x labels are sorted in increasing order


    :param xlist: list of values on x axis
    :param ylist: list of values on y axis
    :param xmin: lower bound of x axis
    :param xmax: upper bound of x axis
    :return: area under the curve in range (xmin, xmax)
    """
    # xlist should be a sorted list in increasing order
    assert sorted(xlist) == xlist, 'xlist must be sorted in increasing order'
    assert xmax >= xmin, 'xmax must be greater than or equal to xmin'
    # search and insert xmin and xmax to the step function curve
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


