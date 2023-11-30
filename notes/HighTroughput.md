# High Throughput Scenario

## FlexGen

这个paper是面试前看的, 但没记住名字导致面试的时候有点不明所以. 在这里, 作者用一图表达了其所做的贡献

![](/home/qzp/Documents/tmp/20230923meeting/notes/img/FlexGen-Poster.png)

个人觉得其贡献集中在computer order上. 本文是对单GPU加载模型的部分, 这样显存有较大的富余, 可以计算大batch, 计算完毕后将KV cache放回显存的下一级. 同时加载另一个batch. 换言之, 从示意图可以看出, 别人是反复加载模型块, 本文是反复加载KV cache, 直到算掉所有batch, 然后在加载下一个模型块.

还有一个个人认为还行的贡献是Quantization, 这个上述公式就概括了.

最后是模型的代码, 可以看出代码对OPT类做了重写, 定义类时把相关policy的操作加进去了, 甚至对torch部分接口做了重写. 实话实说比纯NLP来的硬核, 因为这边写代码, 最复杂的时候, 也就是把人家的模型扒下来, 然后重写, 然后改相关结构.

## DeepSpeed

看一整个DeepSpeed的Github仓库, 可以看到其是针对四块做了优化

- training

- inference

- compressing

- science

整个东西做的比较工程, 从介绍可以看出, 针对inference其融合了多种技术, 因此整个论文写的比较像一个技术文档. 在其中其实是有相当多的概念以前没见过, 如DGX-2, 这是一个集成了16个H100的系统, 论文里称作node. 整个模型的做的工作其实是很多的.

### Inference-Optimized Transformer Kernels

这里指出了在做小batch和大batch的过程中可能出现的问题, 一般小batch会涉及到

- kernel-invocation overhead

- data-transfer overhead

- extremely small batch sizes suffers low computer efficiency

大batch的问题有

- kernel launch overhead

- data-transfer overhead

本章涉及到很多概念. 其中一个概念, 叫kernel, 如GeMM kernel, 实际上就是做GeMM的代码或库. 然后还有fusion, 具体而言, NN的运算可以理解为多个函数的嵌套, 如果能合并其中的一些嵌套, 那么就可以减少部分计算开销.

transformer中的fusion不好做的原因是, 其中的多种运算涉及到不同的tread block之间的数据迁移. 这样在启动新kernel时需要做global memory synchronization. 因此, 这里做的fusion是根据tile来的, 就是把数据不需要在tile之间交叉的放一块, 这样就能防止这样的数据迁移, 从而防止同步的开销.

### Inference-Adapted Dense Transformer Models

一个叫TP (tensor-slicing paralism), 一个叫PP (pipeline paralism). 后者其实很好理解, 就是当LLM去infer一个batch时, 其实都是一个transformer块接着一个, 你可以等一整个batch都算完, 再算下一个batch. 但这样前一个transformer块算完后就闲置了, 没有充分利用, 因此这里可以前一个块算完一个batch后, 立马算下一个batch, 这样反复可以达到目的. 前者有一个类似的, tensor paralism, 是我们熟知的把模型切割成多块放在不同的device里.

### 其他

还有一些就太难了, 看不太懂.

## Reference

- [知乎-DeepSpeed](https://zhuanlan.zhihu.com/p/629644249)
