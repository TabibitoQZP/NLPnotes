## GPT

GPT的结构, 在论文里用几个公式就描述了, 其实非常的简单清晰, 只不过深究起来好像有不少错误的点. 假设给定的为$n$长度的token序列

$$
\mathcal{U}=\{u_1,u_2,...,u_n\}
$$

而后, 我们的模型需要得到一个对数极大似然

$$
L(\mathcal{U})=\sum_{i}\log P(u_i|u_{i-k},...,u_{i-1};\Theta)
$$

其中, $P$为我们模型的一个抽象, 其需要在给定前$k$个序列的情况下, 给出第$k+1$个序列的概率. 经过训练模型需要满足一整个序列满足要求的概率对数和最大.

对于模型具体如何构建, 论文也用三个公式表达了, 如下

$$
\begin{aligned}
h_0&=UW_e+W_p\\
h_l&=transformer_l(h_{l-1})(l=1,2,...,n)\\
P(U)&=softmax(h_nW_e^T)
\end{aligned}
$$

这里, $W_e$是个词嵌入, $W_p$是位置编码, $U$是序列向量$(u_{-k},...,u_{-1})$. 这样, 生成的$h_0$就可以作为transformer的输入, 这里一共有$n$个transformer层, 上一层的输出作为下一层的输入, 每一个transformer有自己的参数. 最后能得到$h_n$, 其维度为$s\times d$. 后面$h_nW_e^T$, 是为了把结果映射到序列序号上, 并softmax做一个分布.

上述代码的含义比较明确, 但具体深究起来就很不舒服, 词嵌入用矩阵表示本身就容易有歧义. 而且后面$W_e^T$不知道是不是词嵌入矩阵, 一不一样其实都可以解释.

最后, 这里的transformer和attention is all you need里的是不一样的! 事实上, attention is all you need里为机器翻译任务搭建的transformer模型是算复杂的, 涉及到自注意力和交叉注意力. 而这里没有交叉注意力, 使用的transformer较为简单.

GPT只使用transformer的decoder部分, 这就意味着不需要像此前一样输入`src`, `ref`两个参数. 只需要一个输入$x$就可以. 同时, 由于不再需要交叉注意力, 因此GPT与transformer的decoder也有区别

- Transformer Decoder : Masked multi-head self-attention + encoder-decoder multi-head self-attention + feed-forward

- GPT Decoder : Masked multi-head self-attention + feed-forward

论文中对模型结构也有图示. 最后, 这里所谓的Masked, 和此前的transformer没有区别, 就是普通的下三角阵. 注意的我这里表述的mask指的是应该被mask的内容, 如下

$$
\left(\begin{array}{ccccc} 
1 & 0 & 0 & 0 & 0\\
1 & 1 & 0 & 0 & 0\\
1 & 1 & 1 & 0 & 0\\
1 & 1 & 1 & 1 & 0\\
1 & 1 & 1 & 1 & 1
\end{array}\right)
$$

其中1代表要保留, 0代表舍弃, 舍弃的将不参与softmax计算. 网络上有另一种mask的表示方案, 就会表达成下列结果

$$
\left(\begin{array}{ccccc} 
0 & -inf & -inf & -inf & -inf\\
0 & 0 & -inf & -inf & -inf\\
0 & 0 & -inf & -inf & -inf\\
0 & 0 & 0 & -inf & -inf\\
0 & 0 & 0 & 0 & -inf
\end{array}\right)
$$

这种写法最后会把该矩阵与实际的$QK^T/\sqrt{d}$做加法, 理论上这二者没有区别, 因为数学上inf不会对softmax结果产生影响, 但实际中我们通常是加一个很大的数, 如$-10^9$. 这样在实际训练中反响传播还是不会忽略掉mask的部分, 尽管对最终结果的影响可能没有影响, 但会造成额外的训练开销.

这样的mask对于transformer (attention is all you need) 训练而言, 其自回归输入和输出要差1个token, 如

- src input: `我爱你!`

- target input: `<begin> I love you!`

- target output: `I love you! <end>`

对于预训练GPT而言, 其自回归输入输出也是差一个token, 如下

- input: `<begin> I love you!`

- output: `I love you! <end>`

可见, 对于GPT而言, 其生成能力会很强, 因为它训练的本质就是教会他如何生成语言, 即预测一个句子的下一个单词是什么.

## BERT

在bert里, 我们只使用transformer的encoder, 且其训练方案是很简单的. 因为在transformer的encoder里其实是不包括mask的, 因此是在数据集中做文章. 首先抽取$15\%$的token, 而后, 这$15\%$的token被按如下比例做替换

- `[mask]`: $80\%$

- 不变: $10\%$

- 错误词: $10\%$

使用mask的目的毫无疑问是预测这个词的实际内容; 替换成错误词是为了保证pre-training和fine-tuning相匹配, 因为在实际下游任务中, `[mask]`是不存在的. 注意这里保持不变不代表和剩余$85\%$等价, 因为即使不变还是需要参与loss和反向传播, 会对模型权重产生影响, 而剩余$85\%$只是作为上下文存在, 本身这个位置的输出是不参与模型训练的.
