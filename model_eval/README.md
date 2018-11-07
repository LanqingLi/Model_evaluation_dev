# 模型组评分系统说明

目前最新的0.2.1 - 0.2.2版本涵盖了肺部、脑部、心脏钙化积分三个模块，并修改了common.custom_metric中的ClassificationMetric,将多分类指标
按类别存成list列表，并兼容了泛化后的objmatch 0.0.4版本

0.2.3版本涵盖了多模型ensemble预测的功能，通过在evaluator中添加if_ensemble开关实现，对应objmatch　0.0.5版本(由于采用类似模型投票机制，需要修改最底层object匹配算法，故在find_objects.py中添加了find_objects_ensemble函数在if_ensemble=True时替代原有的find_objects函数)

运行肺、心脏、脑部、胸腔等各项目的具体流程请见各项目子文件夹的README.md

## python库的版本依赖

objmatch: networkx <= 2.0

model_eval: pandas >= 0.22.0, pandas != 0.23.0, opencv-python >= 3.3.0.10, matplotlib<=2.1.1，objmatch >= 0.0.5

## 超参数调整

### F-Score
对于通过f-score选择最优模型的功能模块(lung, brain等)，在该模块目录下的配置文件config.py中可以设置config.FSCORE_BETA。该参数
为f-score中recall和precision的相对权重，越大则recall相对precision的权重越大，默认值为1.,具体定义见https://en.wikipedia.org/wiki/F1_score

### confidence threshold
对于具有多置信度概率筛选功能的模块(lung, brain等)，在该模块目录下的配置文件config.py中可以设置config.TEST.CONF_THRESHOLD。评估程序
会根据其中的一系列阈值统计模型输出结果。

## 评估指标可视化

### RP曲线

在RP_plot中定义读入文件的路径、文件名，支持.xlsx与.json
```
由模型评估生成的.json文件画图：
python RP_plot.py --json

由模型评估生成的.xlsx文件画图：
python RP_plot.py

```
tip: 对于脑部分割任务，在RP_plot.py中设置cls_key='PatientID'可以生成像https://git.infervision.com/w/%E5%87%BA%E8%A1%80%E6%80%A7%E5%8D%92%E4%B8%AD/ 上以病人为单位统计的RP曲线

### ROC 曲线
在ROC_plot中定义读入文件的路径、文件名，支持.xlsx与.json
```
由模型评估生成的.json文件画图：
python ROC_plot.py --json

由模型评估生成的.xlsx文件画图：
python ROC_plot.py
```

##　注意事项

- 在测试过程中，发现较新的matplotlib(2.2.0)在画图时不允许出现x坐标相同的两个不同点，而目前开发的版本的tools.plot中为了画阶跃的ROC、RP曲线是会出现这种情况的，故要求matplotlib版本<=2.1.1

