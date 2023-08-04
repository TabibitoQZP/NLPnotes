# Transformer

下列用于对应的代码都是可以运行的, 但不考虑效率.

## Scaled Dot-Product Attention

这是一种attention的计算方式, 会给定如下参数

$$
\begin{aligned}
Q_{s_q\times d}\\
K_{s\times d}\\
V_{s\times d_v}
\end{aligned}
$$

而其计算只有1个公式, 在paper中已经给出

$$
Z_{s_q\times d_v}=\bold{softmax}(\frac{QK^T}{\sqrt{d}})_{s_q\times s}V
$$

怎么去理解呢? 首先是$Q$, 其有$s_q$条查询, 每条查询都用$d$个features来表示. 而$K$则表示这里有$s$条key, 其也是用$d$个features表示. 通过向量点乘的方式, 可以计算得到每条查询与每条key的关联程度, 即$QK^T$. 最后是$V$, 则是每条key对应的value. 因此, softmax是针对同一条查询的不同key做了分布计算. 最后与$V$相乘, 得到的就是每个query的到的value.

不同的查询可能对不同的key敏感程度不同, 这就产生了不同的attention. 需要注意的是, 这里并没有涉及到任何参数, 也就是说并不需要训练.

```python
def attention(Q, K, V, mask=None):
    """
    这个是论文里的单头自注意力, 其实是没有参数的
    Q: (b, sq, d)
    K: (b, s, d)
    V: (b, s, dv)
    注意根据矩阵运算规则, 这里的Q, K, V限制没有太严重
    """
    assert Q.size(0) == K.size(0) and Q.size(0) == V.size(0), \
        "should be same batch size."
    assert Q.size(2) == K.size(2), "Q, K should have same feature"
    assert K.size(1) == V.size(1), "K, V should have same seq len"

    b, sq, d = Q.size()
    sqrtD = d ** 0.5

    QK = torch.matmul(Q, K.transpose(-1, -2)) / sqrtD
    if mask is not None:
        QK = QK.masked_fill(mask == 0, -1e9)
    softQK = torch.softmax(QK, -1)  # (b, sq, s)
    return torch.matmul(softQK, V)  # (b, sq, dv)
```

## Multi-Head Attention

多头注意力引入的目的是, 前面的注意力模型没有参数, 输入还是不变的

$$
\begin{aligned}
Q_{s_q\times d}\\
K_{s\times d}\\
V_{s\times d_v}
\end{aligned}
$$

此时, 会有多头的概念, 假设头的个数为$h$, 则可以令

$$
\begin{aligned}
t&=d/h\\
t_v&=d_v/h
\end{aligned}
$$

而后, 将$Q, K, V$线性映射成$h$个, 如下

$$
\begin{aligned}
Q_{(i)s_q\times t}&=Q_{s_q\times d}W_{Q(i)d\times t}\\
K_{(i)s\times t}&=K_{s\times d}W_{K(i)d\times t}\\
V_{(i)s\times t_v}&=V_{s\times d_v}W_{V(i)d_v\times t_v}
\end{aligned}
$$

而后, 令前面的Attention为$DPA$, 可以计算对应的$h$个$Z$, 如下

$$
Z_{(i)s_q\times t_v}=DPA(Q_i, K_i, V_i)
$$

最后, concat一下所有的$Z$, 得到总的$Z$, 如下

$$
Z_{s_q\times d_v}=\bold{concat}\{Z_1,Z_2,...,Z_h\}
$$

这时候得到的还不是最终结果, 需要通过一个线性变换

$$
O_{s_q\times d_v}=Z_{s_q\times d_v}W_{d_v\times d_v}
$$

```python
class MultiHeadSelfAttention(nn.Module):
    """
    这个主要是理解论文figure 2中图的含义, 以及给出的公式.
    """

    def __init__(self, d, h) -> None:
        super().__init__()
        assert d % h == 0, "d % h != 0"

        self.d = d
        self.h = h
        self.t = d // h

        # 有3*h个线性变换层, 其权重矩阵size相同, 但权重不同
        self.LQ = [nn.Linear(self.d, self.t) for _ in range(self.h)]
        self.LK = [nn.Linear(self.d, self.t) for _ in range(self.h)]
        self.LV = [nn.Linear(self.d, self.t) for _ in range(self.h)]

        # 最后有个总的线性变换层
        self.lin = nn.Linear(self.d, self.d)

    def forward(self, Q, K, V, mask=None):
        """
        原始的接受就是Q, K, V, 且shape都一样, 为(b, s, d)
        """
        ZList = []
        for i in range(self.h):
            linQ = self.LQ[i](Q)  # (b,sq,d) -> (b,sq,t)
            linK = self.LK[i](K)  # (b,s,d) -> (b,s,t)
            linV = self.LV[i](V)  # (b,s,d) -> (b,s,t)
            ZList.append(attention(linQ, linK, linV, mask))

        Z = torch.cat(ZList, -1)

        return self.lin(Z)
```

需要指出, 在transformer中, $d=d_v$, 这是为了保证输入和输出的shape完全一致.

## Add Norm层

这里论文给了个通用公式

$$
\bold{LayerNorm}(x + \bold{Sublayer}(x))
$$

其中所谓的sublayer, 就是前面层处理过, 如attention层, 或者feed-forward层处理后的结果. 然后与原始的$x$相加, 做一次layernorm. 这里的layernorm是把$d$个feature做norm. 如下

```python
norm = nn.LayerNorm(d)
```

上述可以声明一个norm层.

## Feed Forward层

实际上就是个MLP, 定义如下

$$
O=\bold{ReLU}(xW_{d\times d_{ff}}+b_1)W_{d_{ff}\times d} + b_2
$$

这样最后会回到和$x$一样的shape.

## 位置编码

对于一个输入$Y_{s\times d}$, 其对应位置编码$PE$如下

$$
\begin{aligned}
PE_{pos,2i}&=\sin (pos/ 10000^{2i/d})\\
PE_{pos,2i+1}&=\cos (pos/ 10000^{2i/d})
\end{aligned}
$$

这个编码我暂时没有很好的解释, 但最终输入前需要把词嵌入与位置编码相加, 即

$$
X=Y+PE
$$

## 按总图拼接transformer

paper中有总图, 需要注意的是, attention层从左向右输入不是$Q, K, V$, 而是$V, K, Q$. 如果这个顺序搞错, 将会没法做...

定义Encoder

```python
class Encoder(nn.Module):
    def __init__(self, d, h, dff) -> None:
        super().__init__()
        self.d = d
        self.h = h
        self.dff = dff

        self.att = MultiHeadSelfAttention(d, h)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, dff),
            nn.ReLU(),
            nn.Linear(dff, d)
        )

    def forward(self, x):
        z = self.att(x, x, x)
        xzSum = x + z
        xzNorm = self.norm(xzSum)
        xzFF = self.mlp(xzNorm)
        fnSum = xzNorm + xzFF
        return self.norm(fnSum)
```

在有图的情况下没什么好说的.

定义Decoder

```python
class Decoder(nn.Module):
    """
    根据模型总图去复原, 注意总图中attention层输入依次是V, K, Q
    """

    def __init__(self, d, h, dff) -> None:
        super().__init__()

        self.d = d
        self.h = h
        self.dff = dff

        self.maskAtt = MultiHeadSelfAttention(d, h)
        self.att = MultiHeadSelfAttention(d, h)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, dff),
            nn.ReLU(),
            nn.Linear(dff, d)
        )

    def forward(self, x, encRet, mask=None):
        """
        这里是尤其特殊的, 因为前面的attention层都是自注意力, 这里引入交叉注意力
        encRet: (b, m, d)
        x: (b, n, d)
        """
        ret = self.maskAtt(x, x, x, mask)
        s = x + ret
        x = self.norm(s)  # (b, n, d)

        ret = self.att(x, encRet, encRet)  # (b, n, d), 输出序列长和原始长一样!
        s = x + ret
        x = self.norm(s)

        ret = self.mlp(x)
        s = x + ret
        return self.norm(s)
```

这里注意, Decoder是需要mask的, 因为这里有个mask attention层, 这个层的原理是, 让$i$号查询只与$i$以前的key相关, 从而造成一种时序的效果, 这个mask也就是个下三角全1矩阵.

最后是整个transformer

```python
class Transformer(nn.Module):
    def __init__(self, d, h, dff, en, dn, srcVoc, dstVoc) -> None:
        """
        srcVoc: 输入词的词库大小
        dstVoc: 输出词的词库大小
        """
        super().__init__()
        self.d = d
        self.h = h
        self.dff = dff
        self.en = en
        self.dn = dn

        # encoder, decoder组
        self.encoders = [Encoder(d, h, dff) for _ in range(en)]
        self.decoders = [Decoder(d, h, dff) for _ in range(dn)]

        # 词嵌
        self.srcEmb = nn.Embedding(srcVoc, d)
        self.dstEmb = nn.Embedding(dstVoc, d)

        # 分类
        self.cla = nn.Sequential(
            nn.Linear(d, dstVoc),
            nn.Softmax()
        )

    def forward(self, x, y):
        """
        x: (b, sx)
        y: (b, sy)
        """
        embX = self.srcEmb(x)  # (b, sx, d)
        embY = self.dstEmb(y)  # (b, sy, d)
        embX += locationEncoder(embX)
        embY += locationEncoder(embY)

        embOut = embX
        for i in range(self.en):
            embOut = self.encoders[i](embOut)

        mask = torch.ones((y.size(0), y.size(1), y.size(1)))
        for i in range(0, y.size(1)):
            mask[:, i, i+1:] = 0

        decOut = embY
        for i in range(self.dn):
            decOut = self.decoders[i](decOut, embOut, mask)

        return self.cla(decOut)
```

这个最后输出是每个token的分布, 对应机器翻译问题, 可以构建测试

```python
if __name__ == "__main__":
    d = 512
    h = 8
    dff = 2048
    en = 8
    dn = 6
    srcVoc = 128
    dstVoc = 256
    trans = Transformer(d, h, dff, en, dn, srcVoc, dstVoc)
    b = 8
    sx = 32
    sy = 64
    x = torch.randint(0, srcVoc, (b, sx))
    y = torch.randint(0, dstVoc, (b, sy))
    print(trans(x, y).shape)
```

## 总结

最重要的是对multi-head attention的理解, 而后, 需要知道模型示意图里的输入是$V, K, Q$. 慢慢的就能从模型图把一整个结构塑造出来.

## reference

- [Attention is all you need.](https://arxiv.org/abs/1706.03762)

- [code reference](https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb)