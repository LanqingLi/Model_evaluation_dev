# 说明

## 文件说明
auto_test.py：用于比对ground truth和predict的结果，并生成相应对比结果和统计结果的。 
直接使用的话，在private config里面改pt_path和gt_path，之后直接运行auto_test.py即可
若是在模型中调用直接调用auto_test.py里面的Auto_test类，初始化之后直接使用类下的print_all_conclusions即可

## 可能需要安装的库
依赖common文件夹里面两个文件
pandas,建议使用0.21.1及以上版本

### 参数说明
- pt_path：模型生成的预测结果存放位置，其中文件的存放格式如下：

```
%model_name
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

- --gt_path：groundtruth anno 存放的位置，其中文件的存放格式如下：

```
anno
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


