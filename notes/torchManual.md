# 模型补充

介绍一些模型以及torch操作, 模型主要聚焦于计算方式, 对于其运作原理不做介绍.

## tensor的操作

- `Tensor.view()`: 一个tensor是以连续的地址空间被编码在内存中的. 因此, 如何读取其中内容完全取决于我们对于这个tensor的外形定义. 即`t[4]`1维的tensor和`t[2][2]`2维的tensor, 二者在内存中编码没有本质区别, 可以通过`t.view(2, 2)`的方式统一化成$2\times 2$的方式输出

```python
t = torch.rand(1, 2, 4)
print(t)
print(t.view(1, 2, 2, 2))
```

- `Tensor.transpose()`: 转置是一个很常见的概念, 然而通常比较好理解的是2维条件下的转置, 实际的tensor可以在任意维度进行转置. 这个不要想太复杂了, 每个维度都是有各自的作用的, 实际上就是将维度对应的作用反了一下而已. 如常见的序列数据, 其维度通常为`batch, s, embDim = t.shape`. 分别对应batch, 序列长度, 序列嵌入的维度. 倘若我们做了`t.transpose(1, 2)`. 则维度序列变成了`batch, embDim, s = t.shape`. 注意, 这是开辟了新的地址空间将转置内容复制过去, 不是in-place的

```python
t = torch.rand(1, 2, 4)
print(t.shape)
print(t.transpose(1, 2).shape)
```
- `Tensor.size()`: 多数时候`Tensor.shape`可以和其互用, 但官方通常是使用`Tensor.size()`

## 函数操作

- `torch.softmax`

- `torch.matmul`: 一个比较麻烦的事情是, torch不一定只是2维的, 而这个函数的使用可以比较灵活, 如下都是支持的

```python
# vector x vector
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
torch.matmul(tensor1, tensor2).size() # []
# matrix x vector
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size() # [3]
# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size() # [10,3]
# batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
torch.matmul(tensor1, tensor2).size() # [10,3,5]
# batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
torch.matmul(tensor1, tensor2).size() # [10,3,5]
```

## nn操作

- `nn.Linear(in_features, out_features)`: 这个可以看source, 可以看到其会初始化weight, bias, 在forward中其实调用了`F.linear`. 这个函数是没有source的 (确切的说不是普通source) , 但其操作是$y=xA^T+b$. 官方文档对其定义为

  - 输入: `(*, H_in)`

  - 输出: `(*, H_out)`

  - 即线性层不关心高维的事情, 只关心最低维的变换

## self-attention

介绍self-attention以及multi-head self-attention用的都是Attention is all you need中的figure 2. 需要注意的是, 这里介绍的其实在文中被叫做Scaled Dot-Product Attention, 这里简单叫做self-attention.

输入为

$$
\begin{aligned}
&Q_{s_q\times d} \\
&K_{s\times d} \\
&V_{s\times d_v}
\end{aligned}
$$

没错, self-attention中接受3个数据! 而且务必要懂得各个的意思, $Q$是查询, $K$是key, $V$是value

$$
\begin{aligned}
Z_{s_q\times d_v}&=\bold{softmax}(\frac{Q\times K^T}{\sqrt{d}})_{s_q\times s}V
\end{aligned}
$$

这里的$\bold{softmax}$是针对最低维, 也就是按行做. 其中的$QK^T$的含义为, 每一行对应一个查询, 对应不同key的权重, softmax后对应概率.

## multihead self-attention

输入还是一样的

$$
\begin{aligned}
&Q_{s_q\times d} \\
&K_{s\times d} \\
&V_{s\times d_v}
\end{aligned}
$$

而后, 假设头个数为$h$, 则需要满足整除关系

$$
\begin{aligned}
h&|d \\
h&|d_v
\end{aligned}
$$

假定对应的整除结果为

$$
\begin{aligned}
t&=d/h \\
t_v&=d_v/h
\end{aligned}
$$

假设输入还是原来的$Q, K, V$. 则首先会对其进行切分, 并做线性变换

$$
\begin{aligned}
Q_{(i)s_q\times t} &= Q\times W_{Q(i)d\times t} \\
K_{(i)s\times t} &= K\times W_{K(i)d\times t} \\
V_{(i)s\times t_v} &= V\times W_{V(i)d_v\times t_v}
\end{aligned}
$$

则假设使用前面的self-attention为$\bold{SA}$, 会有

$$
Z_{(i)s_q\times t_v}=\bold{SA}(Q_{i}, K_{i}, V_{i})
$$

假设我们使用$h$个不同权重的self-attention层, 则最后有$h$个不同的$Z_i$, 可以将他们拼接得到总的$Z$, 如下

$$
Z_{s_q\times d_v}=\bold{concat}\{Z_1,Z_2,...,Z_h\}
$$

尽管此时$Z$与$X$的shape一样了, 但这仍然不是最后输出, 还需要过一个线性层, 如下

$$
O_{s_q\times d_v}=Z_{s_q\times d_v}W_{d_v\times d_v}
$$

至此, 一个multihead self-attention全部完成. 在transformer中, 取$d=d_v$.