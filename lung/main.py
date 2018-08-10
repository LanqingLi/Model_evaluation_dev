from model_eval.lung.evaluator import predict_json_to_xml


if __name__ == '__main__':
    data_dir = '/mnt/data2/ct_2d_fpn_multi_channel_detection/json_for_auto_test_9items'
    save_dir = '/mnt/data2/ct_2d_fpn_multi_channel_detection/json_for_auto_test_9items_anno'
    predict_json_to_xml(data_dir, save_dir)