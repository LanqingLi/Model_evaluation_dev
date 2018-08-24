# 模型评分系统脑部CT模块说明

版本号：0.1.4

该模块主要功能是对脑部CT项目的模型输出进行评估，用以筛选最优模型。目前版本的主要功能是针对脑卒中出血的语义分割模型，统计其分类的tp、fp、fscore、出血体积等指标，画出分割的contour、RP、ROC曲线。
最新版0.1.3实现了evaluator的offline(以文件形式读入模型预测结果，用于inference和evaluation的解耦运行)和online(直接将模型预测输出作为输入，实现inference和evaluation的一体化运行)功能
0.1.4实现了evaluator的onlineiter(读入数据迭代器，可以直接与模型预测结合进行inference、evaluation一体化的运行，同时因为每次只读入一个病人，
避免了数据过大内存不够用的问题)功能
	
## 环境安装

请首先配置公司pypi源: 详见https://git.infervision.com/w/%E7%A0%94%E5%8F%91/%E5%86%85%E9%83%A8python%E4%BB%93%E5%BA%93%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E/

### 安装objmatch

先删除本地的`objmatch`

 -sudo pip uninstall objmatch

安装'objmatch'

 -sudo pip install objmatch
 
### 安装model_eval

先删除本地的model_eval

 -sudo pip uninstall model_eval

安装'model_eval'

 -sudo pip install model_eval
 
## 输入输出测试集等相关格式
详见https://git.infervision.com/T1745 中出血性卒中相关文档

##　文件说明
evaluator.py: 模型评估的主要功能实现 

- 目的：统计多阈值下模型分割的效果，并进行综合评分及筛选最优阈值，画出分割的contour、RP、ROC曲线

- 具体操作：读取分割模型输出的各像素点的mask(默认读取格式为.npy,可支持的数据文件格式为_predict.npy),以及
人工标记的ground truth label的.nrrd文件，经过conf_thresh的阈值筛选后, 经过ClassificationMetric
统计出对应threshold和class的recall, fp/tp, precision等信息,并进行可视化

- 数据生成：
  调用预测函数predict得到预测结果（prob_map）, 直接np.save(save_path, prob_map)得到相应的patient_id_x_predict.npy

config.py: 统一存放评估系统后处理相关参数默认值的配置文件,用户可以根据需求重新定义

- 分类相关参数：默认值全部定义在brain.classname_labelname_mapping.xls中，依次包括分割细分类别（人工标记类别）、分割粗分类别（模型识别类别）、
分割类别置信度概率阈值、分割类别权重

## 代码运行指令(默认在model_evaluation目录下)

### 语义分割功能

offline模式：

1. 多分类(目前仅在二分类数据上有测试)：

-调用multi_class_evaluation:
 python -m brain.semantic_seg_test --gt_dir [ground truth label directory] --data_type npy --data_dir [predict label directory] \
 --img_dir [image directory] --multi_class
 
2. 二分类：

-调用binary_class_evaluation:
 python -m brain.semantic_seg_test --gt_dir [ground truth label directory] --data_type npy --data_dir [predict label directory] \
 --img_dir [image directory]
 
3. 画分割区域轮廓图：

- 二分类：
 - 多阈值对比图　(调用binary_class_contour_plot_multi_thresh, 阈值在config.TEST.CONF_THRESHOLD定义)
 　python -m brain.semantic_seg_test --gt_dir [ground truth label directory] --data_type npy --data_dir [predict label directory] \
 　--img_dir [image directory] --draw --multi_thresh
 - 单阈值对比图 (调用binary_class_contour_plot_single_thresh，阈值在config.THRESH定义)
 　python -m brain.semantic_seg_test --gt_dir [ground truth label directory] --data_type npy --data_dir [predict label directory] \
 　--img_dir [image directory] --draw

online模式：

- 在inference时让模型输出predict_data_list, gt_nrrd_list, img_nrrd_list, patient_list, data_type,　各变量定义如下：

 - predict_data_list: list of predict class score, default type: float32, default shape: [figure number, class number, 512, 512] 
 (background class number = 0), softmax value in range (0, 1)
 - gt_nrrd_list: list of ground truth nrrd mask, default type: int16, default shape: [512, 512, figure number]
 - img_nrrd_list: list of nrrd image data, default type: int16, each element = list: [ground truth label/mask, shape = (512, 512, figure num(batch size)), info of ct scan]
 - patient_list: list of patient id, e.g.: [patient_id_1, patient_id_2, ...]

!!!注意：此处patient_list的顺序必须与其余几个list顺序一致!!!

之后初始化BrainSemanticSegEvaluatorOnline类：

```
brain_evaluator = evaluator.BrainSemanticSegEvaluatorOnline(predict_data_list=predict_data_list,
                                                            gt_nrrd_list=gt_nrrd_list,
                                                            img_nrrd_list=img_nrrd_list,
                                                            patient_list=patient_list)
```
                                       
之后调用brain_evaluator如下函数即可：

1. 多分类(目前仅在二分类数据上有测试)：

-调用multi_class_evaluation: brain_evaluator.multi_class_evaluation()

2. 二分类：

-调用binary_class_evaluation: brain_evaluator.binary_class_evaluation()
 
3. 画分割区域轮廓图：

- 二分类：
 - 多阈值对比图　(调用binary_class_contour_plot_multi_thresh, 阈值在config.TEST.CONF_THRESHOLD定义):
   brain_evaluator.binary_class_contour_plot_multi_thresh()
 - 单阈值对比图 (调用binary_class_contour_plot_single_thresh，阈值在config.THRESH定义)
 　brain_evaluator.binary_class_contour_plot_single_thresh()
                                          
onlineIter模式：

此模式接收的是一个带有predict_key(模型预测结果关键词), gt_key(ground truth标记关键词), img_key(原图数据关键词)和patient_key(病人号关键词)
的数据迭代器(data_iter),这样子在评估模型预测结果时会按照病人号逐个读取处理而不是一次性读入所有病人的数据，避免数据量大时内存不够的问题。

具体操作，在生成data_iter后初始化BrainSemanticSegEvaluatorOnlineIter类：

```
brain_evaluator = evaluator.BrainSemanticSegEvaluatorOnlineIter(data_iter=data_iter,
                                                                predictor=predictor.predict,
                                                                predict_key=predict_key,
                                                                gt_key=gt_key,
                                                                img_key=img_key,
                                                                patient_key=patient_key)
```
                                          
之后调用brain_evaluator如下函数即可：

1. 多分类(目前仅在二分类数据上有测试)：

-调用multi_class_evaluation: brain_evaluator.multi_class_evaluation()

2. 二分类：

-调用binary_class_evaluation: brain_evaluator.binary_class_evaluation()
 
3. 画分割区域轮廓图：

- 二分类：
 - 多阈值对比图　(调用binary_class_contour_plot_multi_thresh, 阈值在config.TEST.CONF_THRESHOLD定义):
   brain_evaluator.binary_class_contour_plot_multi_thresh()
 - 单阈值对比图 (调用binary_class_contour_plot_single_thresh，阈值在config.THRESH定义)
 　brain_evaluator.binary_class_contour_plot_single_thresh()
 
###　实例分割功能

目前模型组业务中尚未涉及实例分割，今后会根据需求添加

## 函数说明

BrainSemanticSegEvaluatorOffline类（语义分割模型评估）：

 - binary_class_contour_plot_single_thresh: 画二分类分割模型单阈值的轮廓线(contour),与ground truth画成一张对照图，轮廓线统一用cv2中
 (0, 255, 0)的亮绿色
 
 - binary_class_contour_plot_multi_thresh: 画二分类分割模型多阈值的轮廓线(contour),与ground truth画成一张对照图，轮廓线使用cv2中从(255, 0, 0)
 到(0, 0, 255)的渐变彩虹色（类似热力图）
 
 - multi_class_evaluation: 分割模型评分,将每个类别单独拿出来，其余作为背景负样本，用阈值筛选类别，并在其中选取最大值作为one-hot label
 
 - binary_class_evaluation: 分割模型评分，将所有正类别统一成一个类别，用阈值筛选
 
BrainInstanceSegEvaluatorOffline（实例分割模型评估）：

 - 待补充
 
## 注意事项

 - 在模型输出的一致性测试过程中，发现与伟导代码给出的结果tp一致，但fp、tn都要偏大。后来证实是因为脑部项目的源代码中，对于ground truth label
 以及predict label无重合的图(tp = 0)直接忽略，而目前的评分系统将所有图都统计在内，所以导致了fp偏大（gt没有出血而模型预测有出血，或者两者都有但没有重合部分）、
 fn偏大（模型预测没有出血但gt有出血，或者两者都有但没有重合）。经和伟导讨论，决定保持现在这种统计方式。
 
 - 在测试过程中，发现较老opencv-python的cv2.findContours函数的输出参数为两个，而基于较新版本(3.3.0.10)开发的cv2.findContours函数
 返回参数为三个，故要求opencv-python版本>=3.3.0.10
 

