


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
    tp = df.at['sum', 'tp_count']
    fp = df.at['sum', 'fp_count']
    fn = df.at['sum', 'fn_count']
    print 'Plaque: recall = ', recall_plq, ' fp/tp = ', fp_plq
    writer2 = pd.ExcelWriter(output_path)
    df.to_excel(writer2, 'sheet1')
    writer2.save()
    return(tp, fp, fn)


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
    :param list1: gt list
    :param list2: pt list
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


def print_specific(output_path, stats_dict):
    '''
    :param output_path: xlsx path
    :param stats_dict: {segment + type : {'tp' : tp_count, ...}
    :return: print xlsx file about specific results.
    '''
    spec_conc = []
    all_class = list(set_class2)
    all_class.sort()
    all_class.extend(['cP', 'ncP', 'mP'])
    for item in all_class:
        spec_conc.append([item, stats_dict[item]['tp'], stats_dict[item]['fn'], stats_dict[item]['fp'],
                          float(stats_dict[item]['tp']) / max(stats_dict[item]['tp'] + stats_dict[item]['fn'], 1),
                          float(stats_dict[item]['fp']) / max(stats_dict[item]['tp'], 1)])
    df = pd.DataFrame(spec_conc, columns=
                      ['Seg/Type', 'tp_count', 'fn_count', 'fp_count', 'recall', 'fp/tp'])
    writer = pd.ExcelWriter(output_path)
    df.to_excel(writer, 'sheet1')
    writer.save()


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
        self.tp, self.fp, self.fn = print_conclusion(xlsx_path, self.conclusion)

    def print_plaque_boxes(self):
        for patient in self.patient_list:
            xlsx_name = patient.name + '_conclusion.xlsx'
            patient_path = os.path.join(self.output, patient.name)
            print_result(patient.gt_list, patient.pt_list, xlsx_name, patient_path)

    def print_segment_xls(self):
        for patient in self.patient_list:
            set_fp = set()
            set_p = set()
            set_fn = set()
            for plaque in patient.gt_list:
                set_p.add(plaque.pos)
                if plaque.max_matching < VOL_THRESHOLD:
                    set_fn.add(plaque.pos)
            for plaque in patient.pt_list:
                if plaque.max_matching < VOL_THRESHOLD:
                    set_fp.add(plaque.pos)
            set_tp = set_p.difference(set_fn)
            set_fp = set_fp.difference(set_tp)
            # tn number is based on current segment classification, i.e. there are 45 segments.
            self.conclusion2.append(
                [patient.name, len(set_tp), len(set_fp), 45 - len(set_tp) - len(set_fp), len(set_fn)])
        xlsx_path = os.path.join(self.output, 'seg_conclusion.xlsx')
        print_conclusion2(xlsx_path, self.conclusion2)

    def print_specific_stats(self):
        specifics = {'cP': {'tp': 0, 'fp': 0, 'fn': 0}, 'ncP': {'tp': 0, 'fp': 0, 'fn': 0},
                     'mP': {'tp': 0, 'fp': 0, 'fn': 0}}
        for segment in set_class2:
            specifics[segment] = {'tp': 0, 'fp': 0, 'fn': 0}
        for patient in self.patient_list:
            for plaque in patient.gt_list:
                pos = plaque.pos
                ptype = plaque.plaque_type()
                label = plaque.res
                specifics[pos][label] = specifics[pos][label] + 1
                specifics[ptype][label] = specifics[ptype][label] + 1
            for plaque in patient.pt_list:
                if plaque.res == 'fp':
                    pos = plaque.pos
                    ptype = plaque.plaque_type()
                    label = plaque.res
                    specifics[pos][label] = specifics[pos].get(label, 0) + 1
                    specifics[ptype][label] = specifics[ptype].get(label, 0) + 1
        xlsx_path = os.path.join(self.output, 'specific_conclusion.xlsx')
        print_specific(xlsx_path, specifics)

    def print_all_conclusions(self):
        self.print_plaque_xls()
        self.print_specific_stats()
        self.print_segment_xls()
        self.print_plaque_boxes()


if __name__ == '__main__':
    auto = Auto_test(pt_path, gt_path, output_path, model_name)
    auto.print_all_conclusions()
