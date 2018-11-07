# 模型评分系统心脏CT模块说明

版本号：0.2.3

该模块主要功能是对心脏CT项目的模型输出进行评估，用以筛选最优模型。目前版本的主要功能是针对心脏/肺部CT中的心脏钙化积分的语义分割模型，统计其分类的tp、fp、fscore、钙化体积钙化积分等指标，画出分割的contour、RP、ROC曲线，暂不涉及检测功能。基本代码继承了脑部评估程序，从0.1.3实现了evaluator的offline(以文件形式读入模型预测结果，用于inference和evaluation的解耦运行)和online(直接将模型预测输出作为输入，实现inference和evaluation的一体化运行)功能
0.1.4实现了evaluator的onlineiter(读入数据迭代器，可以直接与模型预测结合进行inference、evaluation一体化的运行，同时因为每次只读入一个病人，避免了数据过大内存不够用的问题)功能
	
## 环境安装

请首先配置公司pypi源: 详见 https://git.infervision.com/w/%E7%A0%94%E5%8F%91/%E5%86%85%E9%83%A8python%E4%BB%93%E5%BA%93%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E/

### 安装objmatch

对于以下操作，如想安装在虚拟环境中请删掉'sudo'指令

先删除本地的`objmatch`

 - sudo pip uninstall objmatch

安装'objmatch'

 - sudo pip install objmatch
 
### 安装model_eval

先删除本地的model_eval

 - sudo pip uninstall model_eval

安装'model_eval'

 - sudo pip install model_eval
 
## 输入输出测试集等相关格式
待补充，当前阶段测试集在NAS:/volume2/Doctor_Daily/Heart/CardiacCalcified/testData上，详细的数据集信息及权限请联系李蓝青(llanqing@infervision.com)

##　文件说明
evaluator.py: 模型评估的主要功能实现 

- 目的：统计多阈值下模型分割的效果，并进行综合评分及筛选最优阈值，画出分割的contour、RP、ROC曲线

- 具体操作：
 - OnlineIter模式：传入模型预测的预测器(predictor)及数据迭代器(data_iter),后续操作与Offline模式相同，相比Online不需要一次性将所有数据
 放入内存中，大大地减少了空间占用
 - Online模式：传入以病人代号排列的模型预测结果、人工标记与原图的list，后续操作与Offline模式相同
 - Offline模式：读取分割模型输出的各像素点的mask(默认读取格式为.npy,可支持的数据文件格式为_predict.npy),以及
人工标记的ground truth label和原图的.nrrd文件，经过conf_thresh的阈值筛选后, 经过ClassificationMetric
统计出对应threshold和class的recall, fp/tp, precision等信息,并进行可视化

- 数据生成：
  调用预测函数predict得到预测结果（prob_map）, 默认原始图像和分割标签(gt及预测结果)均为.nrrd格式，命名为img.nrrd以及label.nrrd

config.py: 定义了评估系统相关参数的配置参数类,默认值都是经测试选取的最优参数，但不适用于所有测试集，用户可以根据需求重新定义

- 分类相关参数：默认值全部定义在brain.classname_labelname_mapping.xls中，依次包括分割细分类别（人工标记类别）、分割粗分类别（模型识别类别）、
分割类别置信度概率阈值、分割类别权重

## 代码运行指令(默认在model_evaluation目录下)

### 语义分割功能

offline模式（继承于online模式，可以读入数据后初始化online模式）：

online模式

当前版本尚不支持

onlineIter模式：

此模式接收的是一个带有predict_key(模型预测结果关键词), gt_key(ground truth标记关键词), img_key(原图数据关键词)和patient_key(病人号关键词)的数据迭代器(data_iter),这样子在评估模型预测结果时会按照病人号逐个读取处理而不是一次性读入所有病人的数据，避免数据量大时内存不够的问题。

- 分割模型的迭代器迭代时会输出字典，其对应关键词的数据为：predict_key='data':经预处理的待预测数据;gt_key='softmax_label':ground truth的分割标记; patient_key='pid':病人号;img_key='raw':未经任何预处理的原始图像数据';voxel_vol_key='voxel_vol':扫描序列的体素体积(nrrd 'space directions'的行列式)

 - 'data': pre-processed images as input to Predictor, default type: float32, default shape:  [batch size, seqlen, 512, 512] (no rgb channel due to greyscale image), in which image shape '512' can be modified through config.img_shape, sequence length (2.5D model only) can be modified through config.seqlen
 - 'softmax_label': ground truth segmentation label, default type: int16, default shape: [batch size, class number, 512, 512] 
 (background class number = 0, positive class number = 1 if FileIter.if_binary = True)
 - 'raw': raw image data, default type: int16, default shape: [batch size, rgb channel = 3, 512, 512] (RGB channel for plotting contours)

之后初始化BrainSemanticSegEvaluatorOnline类：

```
heart_eval = HeartSemanticSegEvaluatorOnlineIter(cls_label_xls_path=cls_label_xls_path,
						　data_iter=eval_data,
                                                 predictor=predictor.predict,
                                                 predict_key='data',
                                                 gt_key='softmax_label',
                                                 img_key='raw',
                                                 patient_key='pid',
						　voxel_vol_key='voxel_vol',
                                                 conf_thresh=np.linspace(0.1, 0.9, num=3).tolist(),
                                                 if_save_mask=False,
                                                 post_processor=get_calcium_mask,
                                                 if_post_process=True
                                                 )
```

其中if_save_mask=True表示将模型预测的mask存成.npy格式，默认在'HeartSemanticSegEvaluation_mask'文件夹下;
if_post_process=True表示将模型预测结果经过后处理后(例如用130阈值筛选钙化区域)再统计各种指标; post_processor
则定义了if_post_preocess=True时所需的后处理函数

                                       
之后调用heart_eval如下函数即可：


1. 多分类(目前仅在二分类数据上有测试)：

- 调用multi_class_evaluation: heart_eval.multi_class_evaluation(), 用阈值筛选正类别，并在其中选取最大值作为one-hot label

- 调用multi_class_evaluation_light: heart_eval.multi_class_evaluation_light(), 轻量级版本，每张图只做一遍预测

2. 二分类：

- 调用binary_class_evaluation: heart_eval.binary_class_evaluation(), 不管gt有多少类别，我们只关心检出(正样本全部归为一类,pos_cls_fusion=True)

- 调用binary_class_evaluation_light: heart_eval.binary_class_evaluation_light(), 轻量级版本，每张图只做一遍预测
                 
3. 画分割区域轮廓图：

- 二分类：

 - 多阈值对比图　(调用binary_class_contour_plot_multi_thresh, 阈值在config.TEST.CONF_THRESHOLD定义):
   heart_eval.binary_class_contour_plot_multi_thresh()
 
 - 单阈值对比图 (调用binary_class_contour_plot_single_thresh，阈值在config.THRESH定义)
 　 heart_eval.binary_class_contour_plot_single_thresh()
                         

###　实例分割功能

目前模型组业务中尚未涉及实例分割，今后会根据需求添加

## 函数说明

HeartSemanticSegEvaluatorOnlineIter类（语义分割模型评估）：

 - binary_class_contour_plot_single_thresh: 画二分类分割模型单阈值的轮廓线(contour),与ground truth画成一张对照图，轮廓线统一用cv2中
 (0, 255, 0)的亮绿色
 
 - binary_class_contour_plot_multi_thresh: 画二分类分割模型多阈值的轮廓线(contour),与ground truth画成一张对照图，轮廓线使用cv2中从(255, 0, 0)
 到(0, 0, 255)的渐变彩虹色（类似热力图）
 
 - multi_class_evaluation: 分割模型评分,将每个类别单独拿出来，其余作为背景负样本，用阈值筛选类别，并在其中选取最大值作为one-hot label
 
 - binary_class_evaluation: 分割模型评分，将所有正类别统一成一个类别，用阈值筛选
 
BrainInstanceSegEvaluatorOffline（实例分割模型评估）：

 - 待补充
 
## 注意事项
脑部项目测试：
 - 在模型输出的一致性测试过程中，发现与伟导代码给出的结果tp一致，但fp、tn都要偏大。后来证实是因为脑部项目的源代码中，对于ground truth label
 以及predict label无重合的图(tp = 0)直接忽略，而目前的评分系统将所有图都统计在内，所以导致了fp偏大（gt没有出血而模型预测有出血，或者两者都有但没有重合部分）、
 fn偏大（模型预测没有出血但gt有出血，或者两者都有但没有重合）。经和伟导讨论，决定保持现在这种统计方式。
 
 - 在测试过程中，发现较老opencv-python的cv2.findContours函数的输出参数为两个，而基于较新版本(3.3.0.10)开发的cv2.findContours函数
 返回参数为三个，故要求opencv-python版本>=3.3.0.10

心脏项目测试：

 - 画contour时默认有三幅视图，从左到右依次为：模型直接输出的mask;经过130HU值筛选后的mask;ground truth label.当模型训练得很好时，前两幅图在肉眼上几乎无差别(仅个别像素有差别)
 

