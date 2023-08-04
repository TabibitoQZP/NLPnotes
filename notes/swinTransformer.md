# SWIN transformer

## 结构

论文里有结构, 如下

![structure](https://amaarora.github.io/images/swin-transformer.png)

里面涉及到3个层

- patch partition: 将图片打成一块块的patch, 这里写的挺清楚的, 原始图片尺寸是$(H,W,3)$, 图片按$4\times 4$的块打成patch, 则图片变成了$(H/4,W/4,3\times 4\times 4)$的序列, 最后channel组成为$(R_0,G_0,B_0,...,R_{15},G_{15},B_{15})$

- linear embedding: 将原始的像素RGB序列信息经过线性映射到$C$尺寸上, 即经过一个$Linear(48,C)$的线性层

- swin transformer block

- patch merging

其中后两个稍微复杂一点, 下面以stage 1和stage 2作详细说明.

## swin transformer block

上面已经介绍到了, stage 1中经embedding后, 其维度为$(H/4,W/4,C)$. 这时候经过一个swin tranformer block, 由于其也是一个tranformer, 因此维度不会变化, 仍然保持为$(H/4,W/4,C)$.

下面主要是讲其结构, 其结构右图也表示了, 有2种

- LN, W-MSA, RES, LN, MLP, RES

- LN, SW-MSA, RES, LN, MLP, RES

这里的LN就是单纯的LayerNorm层, 可见, 其最重要的曾仍然是W-MSA层, 所谓的SW-MSA只是多了个swin操作, 这个后面再说.

### W-MSA

这里的MSA, 就是所谓的Multihead Self-Attention. 倘若是原始的ViT, 则给定$(H,W,C)$的输入序列, 需要对$HW$个patch相互做自注意力, 对于小尺寸的patch而言是灾难, 会造成极高的复杂度. 作者在这里做了简化, 就是我不管有多少个patch, 我都以$M\times M$的窗口去分割它们, 这样它们只会在自己的窗口内做自注意力. 这样随着图片尺寸增长, 这样的自注意力计算复杂度增长是线性的, 也就比ViT简化了非常多.

各个小窗口计算完后, 直接把其对应的结果放在对应位置, 然后拼凑起来即可, 这样就获得了最后的计算结果, 该结果的维度大小是不变的.

### SW-MSA

这里的区别在于所谓的swin操作, 用个具体例子来说如下, 假设$M=2$. 然后对个$4\times4$个patch做分割, 那么可以分割成如下几块

$$
\left(\begin{array}{cccc} 
0 & 0 & 1 & 1 \\
0 & 0 & 1 & 1 \\
2 & 2 & 3 & 3 \\
2 & 2 & 3 & 3
\end{array}\right)
$$

其中序号相同的patch则相互做自注意力. 此时, 有个问题就是, 这几个patch之间, 序号不同的没有做过自注意力, 相互之间也就不能感受到信息. 因此, 需要把窗口向左下角都移动$W/2$

$$
\left(\begin{array}{cccc} 
0 & 1 & 1 & 2 \\
3 & 4 & 4 & 5 \\
3 & 4 & 4 & 5 \\
6 & 7 & 7 & 8
\end{array}\right)
$$

这样, 原先相互之间没有做过attention的也可以做了. 实际上在这里可以引入一些tricky的方式优化计算, 因为我们这个例子里面, W-MSA只算了4次MSA, 但SW-MSA却算了9次MSA. 因此, 需要通过一些方式简化计算次数, 作者用了平移的方式, 把上图变成下面

$$
\left(\begin{array}{cccc} 
4 & 4 & 5 & 3 \\
4 & 4 & 5 & 3 \\
7 & 7 & 8 & 6 \\
1 & 1 & 2 & 0
\end{array}\right)
$$

此时还是分成左上右上左下右下来算, 只不过需要给出一些mask方式, 让不是同一个序号的相互之间不做attention. 这可以说是实际编写代码过程中的一个trick, 用于加快运算. 但作者确实花了不少篇幅讲这个.

可以看到, 论文配图是two successive swin transformer block. 两个block一个是W-MSA, 另一个是SW-MSA. 这样可以保证做attention能够相互注意, 且仔细看a图中的block都是偶数个的, 因为需要把两个这样的串联.

## patch merging

这个的目的是把$(H,W,C)$变成$(H/2,W/2,2C)$. 假设有个$(4,6,C)$的张量, 其索引为

$$
\left(\begin{array}{cccccc} 
0 & 1 & 2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9 & 10 & 11 \\
12 & 13 & 14 & 15 & 16 & 17\\
18 & 19 & 20 & 21 & 22 & 23
\end{array}\right)
$$

按照$2\times 2$的窗口做分割, 则$0,1,6,7$一块, $2,3,8,9$一块, 以此类推. 将这样的小块对应同一位置的通道提取出来, 构成4个$(H/2,W/2)$的小块, 如第一个小块为

$$
\left(\begin{array}{ccc} 
0 & 2 & 4 \\
12 & 14 & 16\\
\end{array}\right)
$$

这样的小块一共有4个, 其实拼起来会构成一个$(H/2,W/2,4C)$的张量. 此时需要做一次LayerNorm, 但维度还是不变, 最后, 在最后一维做线性映射, 即$Linear(4C,2C)$. 就完成了一整个patch merging的过程.

这个过程会将通道数增加, 但会缩小张量的尺寸, 非常类似于CNN.

## 总结

在swin transformer中, 个人觉得最本质的还是提出了类似于CNN那种层级式的特征提取网络. 原始的CNN为了保证网络既能看到局部信息又能看到全局信息, 通过小卷积核提取小的局部信息, 同时通过持化层让信息聚合, 这样下一层的卷积核能看到更大范围的图像. 整个CNN都是在这样的减少尺寸, 增加通道的思路下不断连接.

为了达成类似的目的, 作者修改MSA为W-MSA以及SW-MSA, 用于简化小patch下的计算复杂度. 使得解析更高尺寸的图片. 由于transformer不会改变输入输出的尺寸, 因此设计了patch-merging的方式来降低尺寸, 增加通道.

最后是模型输出, 可以看到, 在图a中直接把结果输出了, 但其最后的结果怎么用还是不知道的. 原因是这个输出将会接入到别的结构, 这些结构依具体的模型而定. 可见, swin本质是提出了一个general-purpose backbone. 这个backbone的权重是不确定的, 但针对不同任务都可以采用这个backbone做训练.

这一点和NLP还是有区别的. 在NLP里, 使用预训练的BERT作为backbone, 其权重也是确定的, 这就使得小样本训练, 微调成为可能. 在最初的SWIN提出时, 我们只能把它作为backbone, 但我们没有预训练模型, 其权重是不确定的, 仍然需要较大的数据集去训练.