# 模型组评分系统说明
	
运行肺、心脏、脑部、胸腔等各项目的具体流程请见各项目子文件夹的README.md
	
## python库的版本依赖
	
objmatch: networkx <= 2.0
	
model_eval: pandas >= 0.22.0, pandas != 0.23.0, 推荐0.22.0(基于此版本开发)
	
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