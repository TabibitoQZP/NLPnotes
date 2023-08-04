"""
参考: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


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


def locationEncoder(x):
    """
    x: (b, s, d)
    """
    b, s, d = x.size()
    out = torch.zeros((b, s, d), dtype=torch.float32)
    for pos in range(s):
        for i in range(d):
            if i % 2 == 0:
                out[:, pos, i] = np.sin(pos / 10000**(i/d))
            else:
                out[:, pos, i] = np.cos(pos / 10000**((i-1)/d))
    return out


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
