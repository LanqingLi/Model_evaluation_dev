# 模型组评分系统肺部CT模块说明

版本号：0.0.2 加入了f_score定义beta参数的接口以及肺结节多分类测试的nodule_count统计

该模块主要功能是对肺部CT项目的模型输出进行评估，用以筛选最优模型。目前版本的主要功能是针对肺结节检出、分类的模型（Faster RCNN/SSD）,将模型输出的2D
层面的框匹配成3D的结节，并统计其分类检出的tp、fp、fscore等指标，画出ROC曲线。

## 输入输出测试集等相关格式
详见https://git.infervision.com/T1745中CT厚层/薄层检测相关文档

-输入：

 -ground truth annotation (存放在gt_anno_dir中的人工标记xml文件, 且每个.xml文件中的<filename>属性必须与其文件名完全一致)：
 
    ```
    gt_anno_dir/
    ├── patient_id_1/
    |   ├─────────── patient_id_1_xxx.xml
    |   ├─────────── patient_id_1_xxx.xml
    ├── patient_id_2/
    ├── patient_id_3/
    ```  
 -predict json (存放在data_dir中的模型输出的后缀为_predict.json/_predict.npy文件，以json为例)：
    
    ```
    data_dir/
    ├── patient_id_1/
    |   ├─────────── patient_id_1_predict.json
    ├── patient_id_2/
    |   ├─────────── patient_id_2_predict.json
    ├── patient_id_3/
    |   ├─────────── patient_id_3_predict.json
    ```  
 json格式：至少需要包含以下关键词：｛'Mask', 'instanceNumber'(结节层面数，从１算起),'nodule_class', 'prob'（softmax置信度概率），
 'sliceId'(结节层面数，从０算起), 'xmax', 'xmin', 'ymax', 'ymin'} 

## 文件说明
evaluator.py:　模型评估的主要功能实现 

－目的：统计多阈值下模型分类检出的效果，并进行综合评分及筛选最优阈值，画出ROC；对结节匹配算法find_nodules(已封装在公司内部的pypi源的python package
objmatch.objmatch.find_nodules中)

－具体操作：读取检出模型（Faster-RCNN/SSD）输出的anchor boxes（默认读取格式为.json,可支持的数据文件格式为_predict.json/_predict.npy),以及
人工标记的ground truth label的.xml文件，经过conf_thresh的阈值筛选后，调用get_df_nodules.py(./post_process/get_df_nodules.py)进行
预测框与ground truth结节的匹配。匹配后输出预测结节与ground truth结节的pandas DataFrame, 经过df_to_cls_label和ClassificationMetric
统计出对应threshold和nodule_class的recall, fp/tp, precision等信息,对于多分类可以自动生成多类别加权的结果。

－数据生成：
    -对于Faster RCNN, _predict.json/_predict.npy需要运行./predict_demo.py生成，具体代码是在clean_box_new()筛掉肺外的框后调用
./common/utils.py中的generate_return_boxes_npy()/generate_df_nodules_2_json()生成并存储
    -对于SSD,

config.py: 统一存放评估系统后处理相关参数默认值的配置文件,默认值都是经测试选取的最优参数，但不适用于所有测试集，用户可以根据需求重新定义

-分类相关参数：默认值全部定义在lung.classname_labelname_mapping.xls中，依次包括结节细分类别、结节粗分类别、结节类别置信度概率阈值、结节类别权重、
结节匹配向上搜索层面数（预测框）、结节匹配向上搜索层面数（ground truth框）

-结节匹配算法相关参数：存放在config.FIND_NODULES中:

    SAME_BOX_THRESHOLD_PRED:对于模型预测出的结果，在同一层面判断是否合并两个框的阈值,小于该阈值则视为一个等价类
    SAME_BOX_THRESHOLD_GT:对于人工标记的ground truth，在同一层面判断是否合并两个框的阈值,小于该阈值则视为一个等价类

    SAME_BOX_THRESHOLD_PRED_OLD：同上，用于旧版find_nodules的阈值
    SAME_BOX_THRESHOLD_GT_OLD：同上，用于旧版find_nodules的阈值

    SCORE_THRESHOLD_PRED: 对于模型预测出的不同层面两个框的匹配，将其视为二分图中一条边的阈值,小于该阈值则加入二分图
    SCORE_THRESHOLD_GT = 对于人工标记的不同层面两个框的匹配，将其视为二分图中一条边的阈值,小于该阈值则加入二分图
    
-评估/测试相关参数：存放在config.TEST中：

    NMS：RCNN相同类别间non-maximum supression的IOU阈值
    IOU_THRESHOLD:post_process.nodule_compare中对比两个框是否为一个结节（不论类别）的IOU阈值
    CONF_THRESHOLD:evaluator中筛选检出框/结节的softmax置信度概率阈值
    
## 函数说明
LungNoduleEvaluatorOffline类（肺结节模型评估）：

-multi_class_evaluation: 多分类模型评分,先用阈值筛选框之后再进行结节匹配

-multi_class_evaluation_nodule_threshold: 多分类模型评分，先进行结节匹配再用阈值筛选结节，结节置信度概率为最高层面的概率值

-binary_class_evaluation: 二分类(检出)模型评分，先用阈值筛选框之后再进行结节匹配

-binary_class_evaluation_nodule_threshold: 二分类(检出)模型评分，先进行结节匹配再用阈值筛选结节，结节置信度概率为最高层面的概率值

FindNodulesEvaluator类（结节匹配算法测试）：

-evaluation_with_nodule_num: 结节匹配可以看做是一个聚类问题，为了评估聚类效果，需要给每个二维框加上一个结节编号的标记.该函数读入带有结节编号的
ground truth label，与算法匹配好的聚类结果通过调用ClusteringMetric输出比对结果。

-evaluation_without_nodule_cls：该函数针对不带有类别信息的数据（例如cfda的金标准数据）进行测试

-load_data_xml_with_nodule_num:　evaluation_with_nodule_num对应的.xml标签读取函数,结节编号信息默认存在<annotation><object><lobe_pos>中

-load_data_xml_without_nodule_cls: evaluation_without_nodule_cls对应的.xml标签读取函数，所有物体类别默认为'nodule'

其它公有函数：

-predict_json_to_xml: 将模型输出的_predict.json文件按层面转换为xml标签,存储文件名格式为'{}_{}.xml'.format(PatientID, 三位层面数)

-json_df_2_df: 两种pandas.DataFrame格式的转换

-slice_num_to_three_digit_str: 将层面数转换成三位数字，如果原层面数不到三位则在前面填零

## 代码运行指令(默认在model_evaluation目录下)

### 肺结节模型评估（生成结果默认存放在./LungNoduleEvaluation_result中）：

主程序：lung.lung_nodule_test.py
如果希望自定义config中的配置参数，可以自行调用evaluator.LungNoduleEvaluatorOffline并传入自定义参数，使用方式可以参考lung_nodule_test中
的main函数

以下功能均会默认在当前目录下的LungNoduleEvaluation_result文件夹中生成结果,用户可以通过改变LungNoduleEvaluatorOffline的result_save_dir
属性来改变存放地点

```
1. 多分类:

先用阈值筛框再进行结节匹配
-调用multi_class_evaluation：
 python -m lung.lung_nodule_test --multi_class --gt_anno_dir [annotation_directory] --data_type json --data_dir [json_directory]
 
先进行结节匹配再用阈值筛选
-调用multi_class_evaluation_nodule_threshold
 在上述命令行上加入 --nodule_threshold
 
2. 二分类：

先用阈值筛框再进行结节匹配
-调用binary_class_evaluation:
 python -m lung.lung_nodule_test --gt_anno_dir [annotation_directory] --data_type json --data_dir [json_directory]

先进行结节匹配再用阈值筛选
-调用binary_class_evaluation_nodule_threshold
 在上述命令行上加入 --nodule_threshold
 
3. 由ground truth的.xml标记以及模型预测出来的_predict.json生成_gt.json和_nodule.json(调用generate_df_nodules_to_json):
python -m lung.lung_nodule_test  --gt_anno_dir [annotation_directory] --data_type json --data_dir [json_directory] --nodule_json

4. 读入匹配好的结节信息(_nodule.json)筛选一定层厚(SliceRange)以上的结节(调用nodule_thickness_filter):
python -m lung.lung_nodule_test  --gt_anno_dir [annotation_directory] --data_type json --data_dir [json_directory] --nodule_json --thickness_thresh [positive integer]

```

以上操作如果读入的是.npy而非.json文件，则将--data_type json 改为 --data_type npy即可

### 结节匹配算法测试（生成结果默认存放在./FindNodulesEvaluation_result中）：

主程序：lung.find_nodules_test.py
如果希望自定义config中的配置参数，可以自行调用evaluator.FindNodulesEvaluator并传入自定义参数，使用方式可以参考find_nodules_test中
的main函数

以下功能均会默认在当前目录下的FindNodulesEvaluation_result文件夹中生成结果,用户可以通过改变FindNodulesEvaluator的result_save_dir
属性来改变存放地点

```
1. 对有结节编号的检出框的聚类测试：

-调用evaluation_with_nodule_num：

  --调用load_data_xml_with_nodule_num，ground truth含有种类信息：
  python -m lung.find_nodules_test --clustering_test --gt_anno_dir [annotation_directory] --nodule_cls

　--调用load_data_xml_with_nodule_num_without_nodule_cls, ground truth没有种类信息(cfda测试集)：
  python -m lung.find_nodules_test --clustering_test --gt_anno_dir [annotation_directory]  

2. 对没有结节编号及种类的检出框的数量测试：

－调用evaluation_without_nodule_cls:
python -m lung.find_nodules_test --gt_anno_dir [annotation_directory] 

```

### 将模型预测输出(_predict.json)转成xml(生成结果存放在save_dir定义的路径下):

python -m lung.main 

## 注意事项

- find_nodules接收的df_boxes,作为一个pandas.DataFrame,!!!index必须从0开始依次加一递增排列!!!（可以通过reset_index(drop=True)来实现），
否则可能会出现union_find_set调用find_parent无限循环的bug。

- FindNodulesEvaluator类中以load_data_xml为前缀的函数中，ground_truth_boxes_dict必须初始化为collections.OrderedDict(),否则在遍历时
可能会出现病人错位的问题

- FindNodulesEvaluator.evaluation_with_nodule_num.load_data_xml_with_nodule_num不能直接读入predict_json_to_xml生成的xml，因为里面
检出框的类别是预测出来的结节类别(config.CLASSES)，而不是原始的标记类别(config.NODULE_CLASSES),不能当做正常的人工ground truth label读入，否则会漏
GGN等类别。如果要强行读入，则必须强制config.NODULE_CLASSSES和config.CLASSES完全一致。

- 对于ground truth labels以及predict labels,objmatch.objmatch.find_nodules算法的默认最优阈值不一样。尤其对于调用FindNodulesEvaluator,
默认参数均为ground truth的最优参数，如果对于predict labels测试，需要自行定义 same_box_threshold_gt=config.FIND_NODULES.SAME_BOX_THRESHOLD_PRED,
score_threshold_gt=config.FIND_NODULES.SCORE_THRESHOLD_PRED, z_threshold_gt=config.CLASS_Z_THRESHOLD_PRED。并且要注意的是，
predict_labels测试，最终经过find_nodules输出的label中，没有匹配上的框均为-1,ClusteringMetric会出现问题，所以不建议这样做。




