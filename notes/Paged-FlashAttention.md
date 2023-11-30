# PagedAttention and FlashAttention

## PagedAttention

这个大概的思路就是, KV cache需要使用连续的空间去存储, 且随着seq的增长其也会增长. 这种计算特性会导致做生成时KV cache需要预分配整块足够的内存空间. 从而造成大量的内部碎片和外部碎片.

PagedAttention其实就是实现KV cache的一个GPU内存分页. 联系torch的操作, 其实我们可以通过一系列方式使得一个tensor其不是分布在一整块空间内 (如部分索引等) . 这种情况下, 如果tensor的规模较大, 那么做矩阵乘法就会报错. 这里需要调用`.contigious()`方法来重新开辟一块内存空间, 并将此前的值连续存储.

因此, 原生的矩阵乘法是需要空间连续的, 但显然我们可以通过手动写程序在不重新分配空间的情况下实现前面所述的代码. 这本质就需要我们完成一个类似虚拟地址到物理地址的转换. 这本质也是内存分页的一种. 本论文用一个block来存储数个KV对, 这样在一个序列中的KV就可以不用连续的存储, 且除了最后一组可能有内部碎片外, 其余的不会有内部碎片, 这样也直接杜绝了外部碎片的产生.

## FlashAttention

这篇文章的编写甚至深入到了GPU详细的存储山结构层面, 考虑了SRAM从HBM加载数据可能造成的性能瓶颈. 具体而言, 在我们计算softmax时, 计算公式可以分解成如下

$$
\begin{align}
S&=QK^T\\
P&=\mathrm{softmax}(S)\\
O&=PV
\end{align}
$$

上述的每一个步骤, 都需要将输入从HBM中加载到SRAM中, 并将输出存回HBM. 需要注意因为SRAM较小且分布在SM上, 因此$Q$, $K$的加载是分block加载进来的 (按列分) , 然后在给定的SM上完成$S$的一块计算. 这里就涉及到一个问题, 就是softmax怎么分块算. 为了精度, 做softmax前所有值需要减去全局最大值, 这数学上不会影响softmax的结果, 但计算本身会更准确. 如果我们需要分块算, 则全局最大值是没办法知道的.

综上, 在一个SM中, 会分配到如下的block

$$
Q_{i,:}\\
K_{j,:}
$$

然后SM会计算一块$S$

$$
S_{i,j}=Q_{i,:}K_{j,:}^T
$$

这一小块的局部最大值和rescale后的exp和也可以计算

$$
\begin{aligned}
m(S_{i,j})&=\max_{row}S_{i,j}\\
f(S_{i,j})&=\exp[S_{i,j}-m(S_{i,j})]\\
l(S_{i,j})&=\sum_{row}f(S_{i,j})
\end{aligned}
$$

这样相当与每一块都维护了一个列向量, 保存这一小块中每一行的最大值和每一行rescale后的值. 最后所有的$S_{i,:}
$都算完以后, 可以得到全局的最大值

$$
m(S_{i,:})=\max_{j}(m(S_{i,j}))
$$

这时我们可以将各个SM中的$l$rescale一下相加, 得到总的$l$, 并rescale一下$S$, 用于计算真正的softmax

$$
\begin{aligned}
f(S_{i,j})&\leftarrow f(S_{i,j})\exp[m(S_{i,j})-m(S_{i,:})]\\
l(S_{i,j})&\leftarrow l(S_{i,j})\exp[m(S_{i,j})-m(S_{i,:})]\\
l(S_{i,:})&=\sum_{j}l(S_{i,j})\\
\mathrm{softmax}(S_{i,:})&=f(S_{i,:}) / l(S_{i,:})
\end{aligned}
$$

这种情况下, 就完成了尽可能少的访问HBM.

上述计算的问题在于, 计算attention的公式的反响传播可能出现问题. 原始的attention计算会把中间值$S$存起来, 而我们这存的是$f(S_{i,j})$, 最后存的是$O_{i,j}$. 也就是说整个attention矩阵的计算是in-place的. 考虑我们针对$O$计算$Q$, $K$, $V$中元素的偏导 (这些计算结果要被反向传播到更早的网络) . 原则上, 我们需要$S$, $P$这些中间的计算结果来辅助反向传播求导. 这里paper表示很容易通过存储的$m$和$l$来还原这两个值. 这部分就这么解决的, 甚至都没给伪代码.

## Reference

- [PagedAttention, FlashAttention](https://zhuanlan.zhihu.com/p/638468472)
