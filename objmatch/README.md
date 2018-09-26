# 版本0.0.2 说明
该objmatch package封装了针对检测模型输出进行后处理的匹配算法。该版本目前只包含2D到3D的匹配功能(2D-3DMatch)。

## 2D-3DMatch 说明
该package当前版本包含以CT肺结节检测的find_nodules算法为模板的find_objects,其被get_df_nodules中的get_nodule_stat调用，将detection模型输出的按二维层面排列的检出框
匹配成三维的结节，最终用于模型的评估与测试。该功能可以推广到将任何基于二维层面的检测结果重新匹配/聚类成三维结构的问题，例如心脏CT的斑块检测等。

### find_nodules/find_objects说明

#### 需要安装的库
numpy
networkx: 目前只支持2.0或以下版本

#### 使用方法
find_nodules/find_objects为v0.0.2 objmatch的核心算法，定义在objmatch.find_nodules.py/objmatch.find_objects中。find_objects
为find_nodules的泛化版本，将所有包含'nodule'的关键词替换为'object',方便未来的其他需求的调用。具体使用时要求的输入输出如下：

 """将boundingbox转换为结节。
    :param bboxInfo: Bounding Box表格（pandas.DataFrame类型）,index必须为自然顺序。
                     每一行代表一个预测出来的bounding box，包含下面的列：
                     * 'instanceNumber'列：表示当前bounding box所在的层面的InstanceNumber，编号从1开始。
                     * 'xmin'列：表示当前bounding box的左上角x坐标（高度[H]方向）。
                     * 'ymin'列：表示当前bounding box的左上角y坐标（左右[W]方向）。
                     * 'xmax'列：表示当前bounding box的右下角x坐标（高度[H]方向）。
                     * 'ymax'列：表示当前bounding box的右下角y坐标（左右[W]方向）。
                     * 'nodule_class'列：表示当前bounding box的预测类别（如'nodule', 'mass'等）,在find_objects中为'class'。
                     * 'prob'里：表示当前bounding box的预测概率。
    :param Z_THRESHOLD： 每个新层面向前做贪心匹配时往前找的最大层面数，类别为一个字典，格式例如{'class_1': 1, 'class_2': 2, ...}
    :param SAME_BOX_THRESHOLD: 判断同一层面两个框是否为等价类的中心点偏移阈值, e.g. : np.array([int x, int y])
    :param SCORE_THRESHOLD: 判断不同层面两个框是否匹配的sim_metric_3d阈值
    :param nodule_cls_weights:　检出框的类别的权重,与Z_THRESHOLD类似，类别为一个字典，格式例如{'class_1': 1, 'class_2': 2, ...}
    :param same_cls_boost: 当不同层面两个框匹配时，如果类别相同的奖励系数
    :return noduleInfo: 在bnd上附加一列'nodule'，取值为-1, 1..n。
                        -1代表当前Bounding box不属于任何一个nodule;
                        1..n代表当前bounding box所属的结节编号。
                        我们不允许一个结节在同一个层面内存在多个重合的bounding box(一个等价类中没有匹配上的框设为-1)。
    :return nodules: 结节信息的列表，每个元素为一个字典,储存一个匹配好的三维结节的信息,e.g.: 
                     df_nodules = pd.DataFrame({'Bndbox List': [], 'Nodule Id': [], 'Pid': prefix, 'Type': [],
                               'SliceRange': [], 'prob': []})
                     其中'Nodule Id'对于find_objects应替换为'Object Id'
 """

#### 设计思路
该算法是基于韦人医生初版的find_nodules改进后的版本。原先的设计中，定义了等价类的树状数据结构（嵌入并查集unionFindSet中实现）。即在同一层面内，
dice（similarity metric）大于一定阈值的两个框会归为同一颗等价类的树, 最终进行层面间结节匹配时只保留整个等价类的根节点(随机选取)，
并且评判标准中没有考虑框的类别以及概率。本算法设计的初衷是为了解决原算法的如下问题：

１．原算法在等价类的框中随机保留一个，由于没有考虑检出框的概率、种类，这样保留下来的框可能不是最优的，那么可能会影响最终匹配出的结节的种类和置信度。
一个显著的例子是其会增大先用阈值筛框后匹配结节与先匹配结节再筛阈值的两种方法结果的差异性。如果先匹配结节，那么原算法可能在某一层面的等价类中随机选取了
概率、种类权值较低的框，那么最终匹配成结节后可能使得结节的置信度概率较低，在较低阈值时就被筛掉。而如果该层面等价类中实际存在概率、种类权值高的框，那么
如果先筛阈值则会保留下来这个框，在匹配成结节后结节的置信度概率就会更高，带来与前种方法统计上的差异。

２．对于一个真实的结节，其不同层面的截面差异可以很大，模型预测出来的框在不同层面可能大小也有较大差异。原算法在不同层面间框的匹配时采取的是类似于IOU的
评价方式，但这对于前后大小差异较大的框，即使其中心点对得很准，也会很容易匹配不上，认为是多个结节，从而带来较高的假阳。同时对于同一层面内有一定重合的两个
框，单纯用IOU判断是否合并有一定局限性，所以我们设计了一种结合中心点偏移和IOU来判断的方式。

####算法概述
基于旧版find_nodules的改进：

1.重新定义sim_metric为两个框中心点x、y位移除以两个框x、y线度的最大值的一半，再除以IOU,输出为np.array([ , ])，默认阈值为np.array([1.6, 1.6])。
除以IOU的原因是想结合IOU和中心点偏移进行框的合并，例如两个一大一小的框,如果中心点有一定偏移，但在1以内，我们认为小框可能框住的是别的东西，也不倾向于
将两个框合并。

2.在每个2d层面，建立等价类树的方式不变，但先不做等价类间合并。对于三维结节匹配，仍采用贪心算法，但二分图的边权定义为：
w = same_cls_boost * nodule_weight * box_weight * misalign_suppress 其中： same_cls_boost：２(两个框为相同类)，１（两个框不同类） 
nodule_weight：已匹配层面的结节权重（结节类别权重×置信度概率） box_weight: 待匹配层面的检出框权重 （检出框类别权重×置信度概率） 
misalign_suppress: exp(-sim_metric)，基于先验假设：结节具有在三维层面中心点不移动 的特征，对于中心点相对平移较多的框进行惩罚

3.根据上述w计算最大权值匹配，最后检查是否存在不同结节指向同一等价类的不同框，如果存在，去掉权值相对较小的那个。

4.对于认为标记的ground truth和模型预测出的框，往往最优化的find_nodules参数不一样，同时对于不同的标记方式、模型也会有明显差异。具体取值见测试结果。

####测试结果
见https://git.infervision.com/T1610

####注意事项

- find_nodules接收的df_boxes,作为一个pandas.DataFrame,!!!index必须从0开始依次加一递增排列!!!（可以通过reset_index(drop=True)来实现），
否则可能会出现union_find_set调用find_parent无限循环的bug。

- post_process中比较gt与pred的匹配结果时默认SliceRange会按instanceNumber按正序排列，但有时数据会出现instanceNumber反序的情况，需要规范数据
同时在经过post_process之前对此进行排查。当然，一旦出现反序大概率会使得一个结节所有层面为空(原来range(smin, smax)变成了range(smax, smin)=0,
但注意代码会有一到两层的层面延伸操作)，则会在差值补全层面时报错

### common_metrics说明

#### AnchorMetric

包含了对比两个锚框(anchor)相似度的相关函数，例如iou(intersection over union), center_deviation(中心点偏移)等，兼容各种维度，维度信息
在类初始化时需要定义好。

### post_process说明

包含了对比将锚框匹配后的两个物体(例如三维结节)是否一致的相关函数，目前主要用于对比模型预测出物体与ground truth是否一致(fp or tp)的问题。