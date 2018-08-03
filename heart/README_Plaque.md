# 说明
## 文件说明
auto_test.py：用于比对ground truth和predict的结果，并生成相应对比结果和统计结果的。  
## 使用说明
单独使用的时候，请带上参数，示例如下：  
```
python new_auto_test.py --xlsx_save_dir ./compare_result2 --conclusion_xlsx_path ./conlusion_result2.xlsx 
--anno_root_path_pt cardiac_plaque_detection/pred_xml --anno_root_path_gt /valid/anno
```
## 可能需要安装的库
pandas,建议使用0.21.1及以上版本

### 参数说明
- --anno_root_path_pt：模型生成的预测结果存放位置，其中文件的存放格式如下：

```
predict_anno
|
├──BJFW11700348S5
|  |
|  ├─BJFW11700348S5_059.xml
|  |
|  └─BJFW11700348S5_060.xml
|  
└──BJFW11700348S6
   |
   ├─BJFW11700348S6_073.xml
   |
   └─BJFW11700348S6_074.xml
```

- --anno_root_path_gt：groundtruth anno 存放的位置，其中文件的存放格式如下：

```
groundtruth_anno
|
├──BJFW11700348S5
|  |
|  ├─BJFW11700348S5_059.xml
|  |
|  └─BJFW11700348S5_060.xml
|  
└──BJFW11700348S6
   |
   ├─BJFW11700348S6_073.xml
   |
   └─BJFW11700348S6_074.xml


### 方法论说明

这一版auto_test主要改动在于从以box为基准改成以斑块为基准，并且同时统计了plaque-based的结果和segment-based的结果。会以两个xlsx的文件形式中呈现。

同时还会输出一个compare result文件夹，每个病人有三个文件。里面xlsx包括了plaque基本信息，patient_name.txt包含了每个plaque的具体信息，包括了它的每个box的中心坐标和最大匹配度。patient_name_segment.txt包含了每个病人分段结果的具体信息。

本版auto-test在判别是否是同时一个斑块时忽略mP与cP的区别而判为一种，不过如果想采用其他的判别标准可以轻松改动。我把plaque建成了一个类，其中plaque.ptype是一个dictionary，里面统计了标记为mP,cP,ncP的box个数。

还有需要注意的是此程序中计算intersection volume是并不是严格的几何体积，算判别系数的公式也不是iou，而是(intersection/plaque_vol1 + intersection/plaque_vol2)/2，具体信息请见程序里注释。

如有问题请联系wjinyi@infervision.com

### Change log

Fix some bugs, sort patient by their IDs
Print a new file 'specific_stats.xlsx' that prints recall and fp/tp according to segment or type
Add in a new metric for judging if predicted plaque and ground truth plaque are the same type. The new metric calculates a calcium percentage for each plaque.\
Separate hyperparameters from file to auto_test_config.py
