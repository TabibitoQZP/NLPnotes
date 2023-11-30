# GPTQ

## 理论补充

### Hesse矩阵

OBC算法的起源在于对于下列Loss的taylor展开

$$
\begin{aligned}
L&=|WX-\hat{W}X|^2\\
&=|\delta WX|^2
\end{aligned}
$$

而其中$W$是原始的权重, $\hat{W}$是量化矩阵权重, 上述矩阵的shape为

$$
\delta W_{d_{row}\times d_{col}}\\
X_{d_{col}\times N}
$$

这样乘出来也还是矩阵, 这种情况下算欧式距离平方实际上是矩阵中每个元素的距离平方和. 根据矩阵乘法的计算方式, 有2种分解方案, 按$\delta W$的行分解和$X$的列分解

$$
\begin{align}
L&=\sum |\delta W_{i,:}X|^2\\
&=\sum |\delta WX_{:,j}|^2
\end{align}
$$

上述分解都是很好推导的, OBC的解释是, 根据上述结果, 我们在对$W$优化时, 实际上根据(1), 我们只需要将$W$按行分开, 并按行优化即可. 这是OBC中的第一个贡献. 而后, 要做taylor展开, 取其中一个求和元素

$$
\begin{align}
L_i&=|\delta W_{i,:}X|^2\\
&=\delta W_{i,:}X(\delta W_{i,:}X)^T\\
&=\delta W_{i,:}XX^T\delta W_{i,:}^T\\
&=\frac{1}{2}\delta W_{i,:}H\delta W_{i,:}^T
\end{align}
$$

其实上述已经taylor展开完毕了... 它就和$y=x^2$一样, 其taylor展开就是它本身. 同时, 我们也得到了Hesse矩阵为

$$
H=2XX^T
$$

### OBQ

考虑$L_i$量化的本质, 指的是量化$W_{i,:}$中的一个值, 同时调整其他值到任意位置! 使得$L_i$的变动最小. 假设量化的是$W_{i,p}$, 则要加上约束条件

$$
\begin{align}
\delta W_{i,p}+\mathrm{quant}(W_{i,p})=W_{i,p}
\end{align}
$$

这里注意, $W_{i,j}$是确定的, 且给定$W_{i,j}$后, 其quant也是给定的 (取最近邻的那个量化) , 换言之, (7)中2项都是常数项. (1)与(6)结合, 我们需要求$L_{i}$的最小值, 此时因为(7)是约束条件, 因此要用拉格朗日乘子法, 将(6)改成

$$
\begin{align}
L_i=\frac{1}{2}\delta W_{i,:}H\delta W_{i,:}^T+
\lambda(\delta W_{i,p}+\mathrm{quant}(W_{i,p})-W_{i,p})
\end{align}
$$

求导结果为

$$
\begin{align}
\frac{\partial L_i}{\partial \delta W_{i,j}}&=
\frac{1}{2}\frac{\partial}{\partial \delta W_{i,j}}\delta W_{i,:}H\delta W_{i,:}^T\\
&=\frac{1}{2}[(\frac{\partial}{\partial \delta W_{i,j}}\delta W_{i,:})
H\delta W_{i,:}^T+
\delta W_{i,:}H
(\frac{\partial}{\partial \delta W_{i,j}}\delta W_{i,:}^T)]\\
&=\frac{1}{2}(e_{j}^TH\delta W_{i,:}^T+\delta W_{i,:}He_j)\\
&=\frac{1}{2}(H_{j,:}\delta W_{i,:}^T+\delta W_{i,:}H_{:,j})\\
&=\delta W_{i,:}H_{:,j}(j\neq p)
\end{align}
$$

如果$j=p$, 则额外加一条

$$
\begin{align}
\frac{\partial L_i}{\partial \delta W_{i,p}}=
\delta W_{i,:}H_{:,p}+\lambda
\end{align}
$$

(13)和(14)令其为0, 并整合成矩阵形式, 如下

$$
\delta W_{i,:}H=-\lambda e_p^T
$$

可以计算得到

$$
\delta W_{i,:}=-\lambda e_p^TH^{-1}
$$

这里的$\lambda$还是不知道, 要用(7)计算. 上述公式中可以计算得到

$$
\delta W_{i,p}=-\lambda [H^{-1}]_{p,p}
$$

进而

$$
\lambda=-\frac{\delta W_{i,p}}{[H^{-1}]_{p,p}}
$$

代入前面带$\lambda$的式子, 可以得到

$$
\begin{align}
\delta W_{i,:}&=-\frac{\delta W_{i,p}}{[H^{-1}]_{p,p}}[H^{-1}]_{p,:}\\
L_i&=\frac{1}{4}\frac{\delta W_{i,p}^2}{[H^{-1}]_{p,p}}
\end{align}
$$

比较重要的公式都拿出来了, 这里显然当$H$计算完成后, 我们可以简单用(16)算一下每个$p$位置量化后$L_i$的变化, 当$L_{i}$变化很小时, 则可以考虑量化这个值, 并用(15)去量化所有参数. 当量化了一个值以后, $H$的计算要排除掉$W_{i,p}$对应的$X$再算. 是需要重新计算的, 这里可以有下列简化

$$
\begin{align}
H^{-1}_{-p}=(H^{-1}-
\frac{1}{[H^{-1}]_{p,p}}[H^{-1}]_{:,p}[H^{-1}]_{p,:})_{-p}
\end{align}
$$

其中$-p$下标代表删除$p$行$p$列. 用上述公式可以避免重新计算, 这个的证明后面有时间再补充.

### GPTQ

上述优化已经很多了, 将一整个$W$的优化分成了按行的优化, 更新$H$的策略等, 但还是做不了对LLM的优化, 因为规模摆在这. 因此, 这里的优化要更激进一点.

GPTQ从OBQ来, 但再此基础上的优化没有过多的数学背景, 更感觉是基于某些实验甚至是直觉. 优化内容如下

1. OBQ每一行的优化是迭代的, 且迭代顺序是基于贪心, 每次找影响最小的先量化. GPTQ发现不用遵循这样的顺序, 对于量化效果的影响也是很小的

2. OBQ中迭代是一个一个的, 不能很好地利用GPU计算性能. 因此可以一个block一个block去迭代, 这样能大幅提高计算效率

3. Cholesky优化

这里方案1可以固定下处理顺序, 从而使得一整个$W$可以按照

$$
W_{:,i:i+block}
(i=0,block,2block,...)
$$

去优化, 优化一次, 做一次(17)去修改$H$. 这种优化还有一种好处, 就是$H$的修正可以in-place, 且修正后$H$的存储是连续的, 不会像之前的那种方式把$H$割的零零散散.

这里的Cholesky优化如下

$$
H^{-1}\leftarrow \mathrm{Cholesky}(H^{-1})^T
$$

其中的Cholesky优化没见过, 查代码是直接调的库, 后面再补充. 引入这个是为了防止递归计算引入的积累误差. 同时还有一个更重要的点, 就是引入这个以后, 不需要用(17)做$H$的修正了, 用同一个$H$, 量化一个block, 用(15)更新另一个block的weight, 直到更新完.

## 代码补充

参考autoGPTQ, 实际代码还是很长很复杂的, 但`gptq.py`量化核心是很简单的, 就是上述原理, 在paper中甚至有对应的algorithm帮助理解. 但除此之外的代码很难看, 要定义各种类去抽模型.

## SpQR

有部分权重对准确率影响很大, 要保持16bit的保存. 这里主要是开发了一种确定这类重要权重的方式.
