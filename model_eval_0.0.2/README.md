# 模型组评分系统说明

版本号：0.0.2

该模块主要功能是对AI模型输出进行评估，用以筛选最优模型。目前版本的主要功能是针对肺结节检出、分类的模型（Faster RCNN/SSD）,将模型输出的2D
层面的框匹配成3D的结节，并统计其分类检出的tp、fp、fscore等指标，画出ROC曲线。详见lung.README.md

## 输入输出测试集等相关格式
详见https://git.infervision.com/T1745中CT厚层/薄层检测相关文档

## 评估指标可视化

### RP曲线

在RP_plot中定义读入文件的路径、文件名，支持.xlsx与.json
```
由模型评估生成的.json文件画图：
python RP_plot.py --json

由模型评估生成的.xlsx文件画图：
python RP_plot.py

```

### ROC 曲线
在ROC_plot中定义读入文件的路径、文件名，支持.xlsx与.json
```
由模型评估生成的.json文件画图：
python ROC_plot.py --json

由模型评估生成的.xlsx文件画图：
python ROC_plot.py
```