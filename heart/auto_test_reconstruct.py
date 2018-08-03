from common.threeD_test_tools import *
from plaqify_boxes import Plaque, plaqify
import os
import pandas as pd
from private_config import *


def issame_pos(pt, gt):
    if pt.pos == gt.pos:
        return True
    return False


def issame_type(pt, gt):
    if PLAQUE_COMPARISON_MODE == 1:
        if pt.plaque_type() == gt.plaque_type():
            return True
        if pt.plaque_type() != 'ncP' and gt.plaque_type() != 'ncP':
            return True
        return False
    elif PLAQUE_COMPARISON_MODE == 2:
        diff = abs(pt.calcium_percentage() - gt.calcium_percentage())
        if diff < MAX_CAL_PERCENTAGE_DIFF:
            return True


def heart_comparison_metric(pt, gt):
    if issame_pos(pt, gt) and issame_type(pt, gt):
        return pt.vol_miov(gt)
    return(0.0)


def print_conclusion(output_path, conc):
    dfn = pd.DataFrame(conc, columns=
        ['PatientID', 'tp_count', 'fp_count', 'fn_count'])
    dfn = dfn.sort_values(by=['PatientID'])
    df1 = dfn.agg(['sum','mean'])
    df = pd.concat([dfn, df1])
    df['Sensitivity'] = df.apply(lambda row: float(row.tp_count / max(row.tp_count + row.fn_count, 1)), axis=1)
    df['fp/tp'] = df.apply(lambda row: float(row.fp_count / max(row.tp_count, 1)), axis=1)
    recall_plq = df.at['sum', 'Sensitivity']
    fp_plq = df.at['sum', 'fp/tp']
    print 'Plaque: recall = ', recall_plq, ' fp/tp = ', fp_plq
    writer2 = pd.ExcelWriter(output_path)
    df.to_excel(writer2, 'sheet1')
    writer2.save()


def print_conclusion2(output_path, conc):
    dfn = pd.DataFrame(conc, columns=
        ['PatientID', 'tp_count', 'fp_count', 'tn_count', 'fn_count'])
    dfn = dfn.sort_values(by=['PatientID'])
    df1 = dfn.agg(['sum','mean'])
    df = pd.concat([dfn, df1])
    df['Sensitivity'] = df.apply(lambda row : float(row.tp_count/ max(row.tp_count + row.fn_count, 1)), axis = 1)
    df['specificity'] = df.apply(lambda  row : float(row.tn_count/ max(row.tn_count + row.fp_count, 1)), axis = 1)
    df['fp/tp'] = df.apply(lambda row: float(row.fp_count / max(row.tp_count, 1)), axis=1)
    recall_plq = df.at['sum', 'Sensitivity']
    fp_plq = df.at['sum', 'fp/tp']
    print 'Segment: recall = ', recall_plq, ' fp/tp = ', fp_plq
    writer2 = pd.ExcelWriter(output_path)
    df.to_excel(writer2, 'sheet1')
    writer2.save()


def print_result(list1, list2, xlsx_name, writeout_path):
    '''
    #save plaque information to dataframes
    :param list1:
    :param list2:
    :return:
    '''
    output = []
    if os.path.exists(writeout_path):
        shutil.rmtree(writeout_path)
    os.mkdir(writeout_path)
    for ix, plaque in enumerate(list1):
        output.append(['Ground Truth', ix, plaque.pos,
                       plaque.plaque_type(), plaque.boxes[0][5], plaque.boxes[-1][5], plaque.noofboxes(), plaque.res])
    for ix, plaque in enumerate(list2):
        output.append(['Predicted', ix, plaque.pos,
                       plaque.plaque_type(), plaque.boxes[0][5], plaque.boxes[-1][5], plaque.noofboxes(), plaque.res])
    dataframe = pd.DataFrame(output, columns=
    ['Check', 'PlaqueId', 'Position', 'Type', 'StartingSlice', 'EndingSlice', 'Noofboxes', 'Result'])
    output_path = os.path.join(writeout_path, xlsx_name)
    writer = pd.ExcelWriter(output_path)
    dataframe.to_excel(writer, 'sheet1')
    writer.save()
    txt_name = xlsx_name.split('.')[-2] + '.txt'
    txt_path = os.path.join(writeout_path, txt_name)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    txt_file = open(txt_path, 'a')
    txt_file.write('Ground Truth\n')
    for plaque in list1:
        txt_file.write(str(plaque.pos) + ' ' + str(plaque.plaque_type()) + ' ' + str(plaque.max_matching) + '\n')
        for box in plaque.boxes:
            txt_file.write(str(getboxcenter(box)) + ' ' + str(box[5]) + '\n')
        txt_file.write('\n')
    txt_file.write('Predicted Results\n')
    for plaque in list2:
        txt_file.write(str(plaque.pos) + ' ' + str(plaque.plaque_type()) + ' ' + str(plaque.max_matching) + '\n')
        for box in plaque.boxes:
            txt_file.write(str(getboxcenter(box)) + ' ' + str(box[5]) + '\n')
        txt_file.write('\n')
    txt_file.close()



class Auto_test:

    def __init__(self, anno_path_pt, anno_path_gt, output_path, model_name):
        self.patient_list = []
        self.conclusion = []
        self.conclusion2 = []
        self.output = os.path.join(output_path, model_name)
        if os.path.exists(self.output):
            shutil.rmtree(self.output)
        os.mkdir(self.output)
        pname_set = set(os.listdir(anno_path_pt)).union(set(os.listdir(anno_path_gt)))
        print 'Total testing patient number = %d' % len(pname_set)
        for patient_name in pname_set:
            patient = Patient(patient_name)
            patient.add_objectlist(plaqify(os.path.join(anno_path_gt, patient_name), 'gt'), 'gt')
            patient.add_objectlist(plaqify(os.path.join(anno_path_pt, patient_name), 'pt'), 'pt')
            # patient.add_objectlist(boxesinto_plaques(os.path.join(anno_path_pt, patient_name), 'pt'), 'pt')
            stats = patient.match(heart_comparison_metric, VOL_THRESHOLD)
            self.patient_list.append(patient)
            self.conclusion.append([patient_name, stats['tp'], stats['fp'], stats['fn']])

    def print_plaque_xls(self):
        xlsx_path = os.path.join(self.output, 'conclusion.xlsx')
        print_conclusion(xlsx_path, self.conclusion)

    def print_plaque_boxes(self):
        for patient in self.patient_list:
            xlsx_name = patient.name + '_conclusion.xlsx'
            patient_path = os.path.join(self.output, patient.name)
            print_result(patient.gt_list, patient.pt_list, xlsx_name, patient_path)

    def print_segment_xls(self):
        xlsx_path = os.path.join(self.output, 'conclusion_segment.xlsx')
        print_conclusion2(xlsx_path, self.conclusion2)

auto = Auto_test(pt_path, gt_path, output_path, 'SSD_0.99_result')
auto.print_plaque_xls()
auto.print_plaque_boxes()
