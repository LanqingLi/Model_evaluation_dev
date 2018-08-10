# Hyperparameters
from common.xml_tools import get_label_classes_from_xls

# Plaqify_boxes.py
# A box would only be considered as Plaque A only if its iou with plaque A's last box > MIN_ADJBOXES_IOU
MIN_ADJBOXES_IOU = 0.3
# i_w, a_w and t_w are weights of each variable in the calculation of similarity index
I_W = 2.0
A_W = 3.0
T_W = 0.5
POS_BONUS = 2.0
# If two boxes in the same layer has miov greater than MIOV_MAX, only one will be considered to be part of a plaque.
MIOV_MAX = 0.6
# constant used in plqify_boxes.py: is_reliable_plaque
MIN_ACCURACY = 0.9
# paths of input and output files
pt_path = '/home/tx-deepocean/jy/model_evaluation/heart/sample/SSD_pt_0.99'
gt_path = '/home/tx-deepocean/jy/model_evaluation/heart/sample/anno'
output_path = '/home/tx-deepocean/jy/model_evaluation/heart/sample/result'
model_name = pt_path.rsplit('/')[-1]

# make it a global constant so modules in this program can be exported
# class_list = ['PLV_mP', 'Diag_ncP', 'PLV_cP', 'LAD_M_ncP', 'OM_ncP', 'PLV_ncP', 'PDA_ncP', 'RCA_P_mP', 'LAD_D_cP',
#               'RCA_D_mP', 'LAD_M_cP', 'LCX_P_mP', 'LCX_M_cP', 'LCX_D_ncP', 'OM_mP', 'LCX_P_cP', 'LCX_M_ncP',
#               'LAD_P_ncP', 'PDA_mP', 'LAD_P_cP', 'RCA_M_ncP', 'LCX_P_ncP', 'RCA_M_cP', 'RCA_D_ncP', 'RCA_P_ncP',
#               'IM_cP', 'IM_ncP', 'LCX_M_mP', 'LAD_P_mP', 'LM_mP', 'Diag_mP', 'LAD_D_ncP', 'OM_cP', 'LCX_D_mP',
#               'PDA_cP', 'LCX_D_cP', 'LAD_M_mP', 'RCA_D_cP', 'Diag_cP', 'RCA_P_cP', 'LM_cP', 'LM_ncP', 'IM_mP',
#               'RCA_M_mP', 'LAD_D_mP']
#
# class_dict = {'PLV_mP': 'PLV_mP', 'Diag_ncP': 'Diag_ncP', 'abnormal': 'abnormal', 'PLV_cP': 'PLV_cP',
#               'LAD_M_ncP': 'LAD_M_ncP', 'OM_ncP': 'OM_ncP', 'PLV_ncP': 'PLV_ncP', 'PDA_ncP': 'PDA_ncP',
#               'RCA_P_mP': 'RCA_P_mP', 'LAD_D_cP': 'LAD_D_cP', 'RCA_D_mP': 'RCA_D_mP', 'LAD_M_cP': 'LAD_M_cP',
#               'LCX_P_mP': 'LCX_P_mP', 'LCX_M_cP': 'LCX_M_cP', 'LCX_D_ncP': 'LCX_D_ncP', 'OM_mP': 'OM_mP',
#               'LCX_P_cP': 'LCX_P_cP', 'LCX_M_ncP': 'LCX_M_ncP', 'myocardial_bridge': 'myocardial_bridge',
#               'LAD_P_ncP': 'LAD_P_ncP', 'PDA_mP': 'PDA_mP', 'LAD_P_cP': 'LAD_P_cP',
#               'motion_artifacts': 'motion_artifacts', 'RCA_M_ncP': 'RCA_M_ncP', 'LCX_P_ncP': 'LCX_P_ncP',
#               'RCA_M_cP': 'RCA_M_cP', 'RCA_D_ncP': 'RCA_D_ncP', 'RCA_P_ncP': 'RCA_P_ncP', 'IM_cP': 'IM_cP',
#               'IM_ncP': 'IM_ncP', 'LCX_M_mP': 'LCX_M_mP', 'LAD_P_mP': 'LAD_P_mP', 'LM_mP': 'LM_mP',
#               'Diag_mP': 'Diag_mP', 'LAD_D_ncP': 'LAD_D_ncP', 'OM_cP': 'OM_cP', 'LCX_D_mP': 'LCX_D_mP',
#               'stenting': 'stenting', 'PDA_cP': 'PDA_cP', 'LCX_D_cP': 'LCX_D_cP', 'LAD_M_mP': 'LAD_M_mP',
#               'RCA_D_cP': 'RCA_D_cP', 'Diag_cP': 'Diag_cP', 'RCA_P_cP': 'RCA_P_cP', 'LM_cP': 'LM_cP',
#               'LM_ncP': 'LM_ncP', 'IM_mP': 'IM_mP', 'disorder': 'disorder', 'RCA_M_mP': 'RCA_M_mP',
#               'LAD_D_mP': 'LAD_D_mP'}

cls_xmlpath = '/home/tx-deepocean/jy/model_evaluation/heart/classname_labelname_mapping.xls'
class_list, label_classes, class_dict, conf_thresh = get_label_classes_from_xls(cls_xmlpath)
class_list.remove('motion_artifacts')
class_list.remove('abnormal')
class_list.remove('disorder')
class_list.remove('myocardial_bridge')
class_list.remove('stenting')
class_list.remove('__background__')
set_class2 = set([])
for item in class_list:
    set_class2.add(item.rsplit('_', 1)[0])

# Auto test params
# Max distance between layers that can be considered as one plaque
LAYER_TOLERANCE = 3
# If the intersection index > VOL_THRESHOLD and other requirements are met,
# those two plaques will be considered as the same one
VOL_THRESHOLD = 0.3
# If distance of box center is greater than MAX_BOX_DISTANCE(unit: pixel), they won't be considered as the same plaque
MAX_BOX_DISTANCE = 30
# Judge if predicted plaque is the same as ground truth plaque
#       Mode 1: Categorical judgement, i.e. judge 100% ncP as one class and others as the other class. Two plaques
#               are of the same type if they belong to the same class.
#       Mode 2: Continuous judgement, i.e. Calculate calcium percentage of each plaque, with mP considered as 0.5 cP and
#               0.5 ncP. If difference in calcium percentage is greater than MAX_CAL_PERCENTAGE_DIFF, they will be
#               considered as different.
# Mode 2 is only available for plaque auto_test, not applicable for segment auto_test yet.
PLAQUE_COMPARISON_MODE = 1
MAX_CAL_PERCENTAGE_DIFF = 100