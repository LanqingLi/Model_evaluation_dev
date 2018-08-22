# -- coding: utf-8 --
import numpy as np
import os, json
import pandas as pd

def save_xlsx_json(result_df, opt_thresh, result_save_dir, xlsx_name, json_name, sheet_name_1, sheet_name_2):
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    print ("saving %s" % os.path.join(result_save_dir, xlsx_name))

    # 　如果已存在相同名字的.xlsx文件，默认删除该文件并重新生成同名的新文件
    if os.path.isfile(os.path.join(result_save_dir, xlsx_name)):
        os.remove(os.path.join(result_save_dir, xlsx_name))

    writer = pd.ExcelWriter(os.path.join(result_save_dir, xlsx_name))
    result_df_cpy = result_df.copy()
    result_df_cpy.to_excel(writer, sheet_name_1, index=False)
    opt_thresh_df = pd.DataFrame.from_dict(opt_thresh, orient='index')
    opt_thresh_df = opt_thresh_df.reset_index(drop=True)
    opt_thresh_df.to_excel(writer, 'optimal_threshold')
    writer.save()

    print ("saving %s" % os.path.join(result_save_dir, json_name))
    # 　如果已存在相同名字的.json文件，默认删除该文件并重新生成同名的新文件
    if os.path.isfile(
            os.path.join(result_save_dir, json_name + '_' + sheet_name_1 + '.json')):
        os.remove(
            os.path.join(result_save_dir, json_name + '_' + sheet_name_1 + '.json'))
    if os.path.isfile(os.path.join(result_save_dir, json_name + '_' + sheet_name_2 + '.json')):
        os.remove(os.path.join(result_save_dir, json_name + '_' + sheet_name_2 + '.json'))

    json_result_df = result_df_cpy.T.to_json()
    with open(os.path.join(result_save_dir, json_name + '_' + sheet_name_1 + '.json'),
              "w") as fp:
        json_result_df = json.loads(json_result_df, "utf-8")
        json.dump(json_result_df, fp)

    json_opt_thresh = opt_thresh_df.T.to_json()
    with open(os.path.join(result_save_dir, json_name + '_' + sheet_name_2 + '.json'),
              "w") as fp:
        js_opt_thresh = json.loads(json_opt_thresh, "utf-8")
        json.dump(js_opt_thresh, fp)
