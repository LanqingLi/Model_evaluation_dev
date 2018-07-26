# 基于Faster-RCNN ./auto_test的模型评分系统说明

## 文件说明
model_evaluation.py: 

目的：统计多阈值下模型分类检出的效果，并进行综合评分及筛选最优阈值，画出ROC

具体操作：读取Faster-RCNN的Predict类(./predict.py)的Predict.get_boxes()输出的anchor boxes（其数据结构为长度为CT图像层面数的一个四层列表，第二层为以框的类别排列的列表，第三层为以每个框排列的列表，第四层为每个框的列表，包含如下信息: [xmin, ymin, xmax, ymax, nodule_class, prob]，目前可读取的数据文件格式为_predict.json/_predict.npy), 经过conf_thresh的阈值筛选后，调用get_df_nodules.py(./post_process/get_df_nodules.py)进行预测框与ground truth结节的匹配。匹配后输出预测结节与ground truth结节的pandas DataFrame, 经过compute_fn_fp_tp()统计出对应threshold和nodule_class的recall, fp/tp, precision等信息。

数据生成：_predict.json/_predict.npy需要运行./predict_demo.py生成，具体代码是在clean_box_new()筛掉肺外的框后调用./common/utils.py中的generate_return_boxes_npy()/generate_df_nodules_2_json()生成并存储

## 函数说明
Model_evaluation.multi_class_evaluation: 多分类模型评分

Model_evaluation.binary_class_evaluation: 二分类(检出)模型评分

Model_evaluation.model_score: 对于每个类别，计算模型检出评分，兼容多种评分函数，例如F_score

Model_evaluation.ROC_plot: 读入生成好的model_evaluation.xlsx并画出各个种类结节的ROC曲线，横轴为precision, 纵轴为recall

Model_evaluation.get_mAP: 计算多分类average precision

## 代码运行指令(默认在ct_2d_fpn_multi_channel_detection目录下)

先按照上层README.md预测代码运行predict_demo.py及auto_test，如果图像数据不是dicom格式或需要窗宽窗位归一化等操作，需修改config.FOR_DICOM等相关flag,并关闭--dicom

python predict_demo.py --image_dir patient_series_folder --prefix model/resnet_e2e --epoch 15 --dicom --multi --auto

运行时会自动生成_predict.json/_predict.npy的预测框以及ground truth结节和匹配之后的结节_nodule.json。为方便后面进行阈值筛选，需自定义每个类别的最小阈值min_conf_thresh。

 # 多分类:

读取.json

python -m auto_test.model_evaluation --multi_class --gt_anno_dir ../Infervision/test_gt/test310/test310 --data_type json --data_dir ./json_for_auto_test/

读取.npy

python -m auto_test.model_evaluation --multi_class --gt_anno_dir ../Infervision/test_gt/test310/test310 --data_type npy

 # 二分类：

相比多分类去掉 --multi_class即可

 # 筛选SliceRange大于一定阈值的结节：

在main()函数中设定好thickness_thresh(整数，目前综合效果设为１最好)，屏蔽掉multi/binary_class_evaluation函数并重新跑一遍model_evaluation.py:

python -m auto_test.model_evaluation --image_dir ../Infervision/test_gt/test310/test310/dcm/ --dicom --data_dir ./json_for_auto_test/ --data_type json

会重新自动运行一遍auto_test,生成后缀为('_nodule%s.json'%thickness_thresh)的json文件及相应的MNS、result.xlsx

 # Model_evaluation、generate_df_nodules_2_json、筛选SliceRange、auto_test画图一体化运行：

python -m auto_test.model_evaluation --multi_class --image_dir ../Infervision/test_gt/test310/test310 --gt_anno_dir ../Infervision/test_gt/test310/test310 --data_type json --data_dir ./json_for_auto_test/　
 

