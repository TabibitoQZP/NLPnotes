# GPU basic

主要参考2个文档

- [GPU Performance Background User&#x27;s Guide - NVIDIA Docs](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

- [Matrix Multiplication Background User&#x27;s Guide - NVIDIA Docs](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

前者更加基础, 涉及到GPU的基础知识, 后者则着重介绍矩阵乘法相关.

## GPU架构

基础架构如下

![](https://docscontent.nvidia.com/dita/00000186-1a08-d34f-a596-3f291b140000/deeplearning/performance/dl-performance-gpu-background/graphics/simple-gpu-arch.svg)

在GPU中, SM为流多处理器(streaming multiprocessor), 其有与之配套的L1缓存. L1, L2, DRAM相当于GPU中的存储器山结构. 如A100有108个SM, L2为40MB, DRAM为80GB.

在SM中支持多种操作, 如下

![](https://docscontent.nvidia.com/dita/00000186-1a08-d34f-a596-3f291b140000/deeplearning/performance/dl-performance-gpu-background/graphics/multi-add-op.svg)

如Volta针对CUDA Cores的FP64, 一个SM可以在一个时钟周期内做32次乘加操作(multi-add operations). 需要注意, 1次乘加操作相当于2次浮点运算.

还有就是CUDA Core和Tensor Core的区别

> CUDA provides a platform for general-purpose computing on GPUs, while Tensor Cores are specialized processing units designed for matrix operations in deep learning.

这些操作相当于对SM的一个更高层次的抽象. 只不过由于执行方式不同, 其速度也不一样.

用上述表格甚至可以计算一些GPU的运算速度, 以A100计算TF32为例, 其频率为1.41GHz, 故其FLOPS为

$$
108\times(512\times2)\times1.41\times10^9=1.56\times10^{12}FLOPS
$$

## GPU运行模型

这个没啥好说, 就是1个function可以分成多个tread, GPU将其组织成大小相同的tread block, 并分配到SM上. 通常, 一个tread block内的多个tread最好是有共享和同步需求的, 这样效率会比较高. 同时, 一个SM也可以并发执行多个tread block, 所以有所谓的tail效应, 如下

![](https://docscontent.nvidia.com/dita/00000186-1a08-d34f-a596-3f291b140000/deeplearning/performance/dl-performance-gpu-background/graphics/utilize-8sm-gpu.svg)

就是一堆SM开始是满载执行的, 但任务到了最后, 只剩下部分SM在执行了, 其他的执行完了. wave指的是并发执行的tread block的集合, 从图中看就是一行一行的.

## GEMM

GEMMs (General Matrix Multiplications), 通用的矩阵乘法, 可以定义为

$$
C=\alpha AB+\beta C
$$

这里的$\alpha, \beta$可以简单理解为权重, 同时$C$出现了2次, 这样可以代表一种残差连接, 从而在NN领域更具有通用性.

## Math And Memory Bounds

假设矩阵的shape为

$$
A_{M\times K}\\
B_{K\times N}\\
C_{M\times N}
$$

这里可以计算一下矩阵乘法的复杂度, 对于$C_{M\times N}$中的一个元素$C_{ij}$, 其要计算$K$次乘法, 后得到$K$个中间值, 而后这些中间值的加法, 若不考虑归并这类复杂的算法, 那么加法要计算$K-1$次, 所以在NV的官方文档中认为进行了$2MNK$次浮点运算. 但不管怎么样, 复杂度是$O(MNK)$是没问题的.

需要注意的是, 在NV的文档中, FLOPS是浮点运算次数的简写, 在一般计算机概念里, FLOPS指每秒浮点运算次数, 这两者有很大的不同.

需要完全理解这两个bound, 可能还要参考[GPU Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understand-perf). 这里有2个新概念, arithmetic intensity$AI$和ops:byte$OPB$. 下面一一解释

### arithmetic intensity

计算公式为

$$
AI=\frac{\bold{FLOPS}}{\bold{ByteAccess}}
$$

其中ByteAccess可以理解为显存中需要访问的地址数量, 但尴尬的是, 这个似乎看起来像$OPB$的定义, 但这个后面再解释. 以GEMM为例, 假设元素都按FP16, 即半精度来计算. 其通用的FLOPS可以计算为

$$
AI=\frac{2MNK}{2(MK+KN+MN)}=\frac{MNK}{MK+KN+MN}
$$

因此, arithmetic intensity是针对具体计算而异的.

### ops:byte

计算公式为

$$
OPB=\frac{BW_{math}}{BW_{mems}}
$$

计算和内存的bandwidth之比, 这个指标就是单纯的GPU特化的了.

### 两个bounds

解释完了上面的, 就可以定义两个bound了, 倘若

$$
AI>OPB
$$

可以理解为计算量较大, 是Math bound的, 倘若

$$
AI<OPB
$$

则内存开销较大, 是Mem bound的.

二者都会导致GPU的显存或者核心其中之一未跑满而另一方又成为瓶颈.

### Matrix-Vector Product

只要在上述公式中假设$M=1$即可, 结果为

$$
AI=\frac{NK}{K+NK+N}=\frac{1}{1+1/N+1/K}
$$

这个可能有点难理解, 但我们可以粗略地想象一下, 当$B$矩阵规模越大, 则$AI$会越接近1, 但不管怎样都是小于1的. 因此其总是很小, 是Mem Bound的.

## GPU执行矩阵运算

之前介绍了GPU的运行方式, 也介绍了矩阵乘法, 在GPU中, 计算矩阵的方式为先做划分, 将矩阵分成很多tile. 如下

![](https://docscontent.nvidia.com/dita/00000189-949d-d46e-abe9-bcdf9f8c0000/deeplearning/performance/dl-performance-matrix-multiplication/graphics/tiled-outer-prod.svg)

图画的还是很清晰的, blocks由$Mtile\times Ntile$的结果块界定, 每个blocks中需要做的是从$A, B$矩阵中加载数据, 然后送入SM中计算.

Tile的shape也能影响计算效率, NV的卡针对一些tile的shape做了优化. 比如(256, 128), (128, 256), (128, 128)等等.

## 各种quantization

因为从$A$, $B$矩阵确定了$C$矩阵的外形后, 就可以划分$C$矩阵并分配给SM算了, 前面已经说了NV对特定Tile的shape做了性能优化, 但分块时不能总是保证满足Tile的块大小需求. 这时候就会出现tile quantization现象.

![](https://docscontent.nvidia.com/dita/00000189-949d-d46e-abe9-bcdf9f8c0000/deeplearning/performance/dl-performance-matrix-multiplication/graphics/tile-quant.svg)

这里就是个示例, 其实看b, c可以看到, tile分配固定下来后, 对应的计算时间增长不多, 但TFLOPS增长很快, 因为刚超tiles的时候, 刚分配新SM时, TFLOPS立马掉下来.

还有就是假设Tile的数量不是SM数量的整数倍, 也会出现wave quantization现象, 就是算到u最后, 有SM还在算, 但有SM已经空置了.
