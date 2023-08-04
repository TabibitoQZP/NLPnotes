# DETR

目标检测问题转化为集合预测问题, 将目标检测问题转化为了一个端到端的框架. 以往的端到端模型会有一些non-maximum supression (nms) 问题, 会导致模型建立非常复杂. 本文就避免了后处理的复杂操作, 且给出了一个简单优雅的代码, 附在论文的最后一页 (50行不到) .

## 目标检测简介

目标检测问题可以概括为, 给定一张图, 要输出一个物体及其对应的标注框, 常用的包括YOLO. 以往的目标检测算法都会生成大量冗余框, 所以最后会涉及到nms操作, 把多余的框抑制掉. 因此, 此前的算法都需要一定的后处理才能达到检测效果, 且不是端到端的.


## loss

DETR不管给定了什么样的图, 其最后的输出框的个数是固定的, 假设为$N$个, 为了能够检测所有物体, 这个$N$一般会设置得比较大. 但ground truth的个数不一定是$N$, 且位置和$N$个预测也不知道是什么关系, 那么如何做loss计算呢, 这里要使用二分图匹配.

所谓二分图, 就是图$G(V,E)$. 其点集能分割成$\{V_1,V_2\}$, 所有边都满足一个点在$V_1$, 一个点在$V_2$. 而二分图匹配则是找到二分图$G$的一个子图$M$, 其边都不共顶点.

上述表述还是太图论化了, 实际上, 对于具体问题, 还有具体的匹配方案. 现实中, 假设有3个工人, 有2个活要他们做, 那么每个工人对于每一项活都有个价位, 如$i$号工人对于$j$号活的开价是$C_{ij}$, 那么会有个cost矩阵, 比如长这样

$$
\left(\begin{array}{cc} 
1 & 3 \\
2 & 2 \\
3 & 1 \\
\end{array}\right)
$$

对于工头来说, 自然是要给出一个方案, 使得每一个活有一个工人做, 且总的花费最低. 这类问题最后都能转化为一个矩阵表示, 且再不济也能用暴力求解. 事实上, 由于问题常见且需求清晰, 这个问题是有相应的库来解决的, 如下

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

costMat = np.array([[1, 3],
                    [2, 2],
                    [3, 1]])

row_ind, col_ind = linear_sum_assignment(costMat)
print(row_ind, col_ind)  # 行索引, 列索引, 对应匹配
"""
[0 2] [0 1]
"""
```

就很棒.

针对于本问题, 假设实际的物体有$M$个, 且$M< N$, 则会有一个$N$行, $M$列的cost matrix. 针对每一个标注框及其对应的物体类型, 会有一个loss, 这就是矩阵内元素的值. 用最小二分图匹配可以获得最小的匹配, 对应的和就是这次预测的loss.

针对于每一个元素loss的计算, 会有一个分类的loss, 会有一个出框的loss. 需要补充说明的是, 这里不是简单找每个ground truth对应的loss最小的预测, 虽然我没有看出这种方式会对最后结果造成明显差异.

## 模型结构

![structure](https://media.geeksforgeeks.org/wp-content/uploads/20200601074340/detr-arch.jpg)

图首先会通过一个CNN抽特征, 然后抽完过个线性层, 并用位置编码相加. 后接入一个transformer, 这里的transformer是包括encoder和decoder的完整transformer, 在论文附录里的python代码也能看出来, 直接用的是pytorch提供的transformer, 这个transformer就是attention is all you need的实现. 这里decoder的输入是query, 假设特征维度为为$D$, 则$Q$矩阵为$N\times D$. 即有$N$条query, 对应要给出$N$个框. 每个query会输出要么是物体及其对应的bounding box, 要么就输出no object.

这里的no object实际上是一个特殊的object, 如果判定为这个物体类型, 则这个box是无效的. 因此倘若u训练集里有$C$种物体类型, 那么训练时需要给$C+1$种物体类型.

## 模型代码

论文里就有... 也是离谱, 实话实说看代码可能更清晰, 加了点注释, 顺便这代码写的真挺好的.

```python
import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers):
        super().__init__()
        # 使用restnet50作为特征提取器
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])
        
        # 使用2d卷积核, 注意用法, 其接受的输入是B, C, H, W, 可能很多人印象里是B, H, W, C
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads,
                                          num_encoder_layers, num_decoder_layers)
        # 可以看到, 分类比原本的多一个
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # 是的, 对于box来说, 就是拟合x1, y1, x2, y2四个数
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # 初始化3个可以调整的参数(原来是这样搞...), 后两个是位置编码
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        # 这里相当于位置编码的前半部分来自row, 后半部分来自col
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()


detr = DETR(num_classes=91, hidden_dim=256, nheads=8,
            num_encoder_layers=6, num_decoder_layers=6)
detr.eval()
inputs = torch.randn(1, 3, 800, 1200) # 从这里也可以看到channel是第二维
logits, bboxes = detr(inputs)
```