# TableGPT

## Set Transformer

不太好读, 但主要是模型都是用公式的方式给出的. 而且混了一些集合的特性科普, 就很费时间. transformer本身的部分不难就是了, 结构如下

![](https://production-media.paperswithcode.com/methods/a97c3a70-7657-4a08-bc35-1f16eeb8715d.jpg)

在上述中, c结构就是简单的attention is all you need中的encoder架构. 而b的话要注意输出通道, 要稍微看一下公式

$$
H=LN(X+MH(X,Y,Y;\omega))\\
MAB(X,Y)=LN(H+FF(H))
$$

这里的$MH$层如下

$$
MH(Q,K,V)
$$

显然, $SAB$和$MAB$的关系为

$$
SAB(X)=MAB(X,X)
$$

因为这里是集合操作, 因此用transformer会有一个$N^2$的复杂度. 为了规避这个问题, 这里创造了一个$ISAB$, 如下

$$
ISAB(X)=MAB(X, MAB(I,X))
$$

这里的$I$是可训练的参数, 相当于query被固定了下来.
