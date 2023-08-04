# LayoutLMv3

## 结构

首先看一下结构, 这应该是笔记里第一个多模态 (multi-modal) 的论文... 图片是原论文的, 取自hugging face.

![structure](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png)

这里图片其实做的很好, 我们就不考虑什么多模态, 单看左边的部分, 将图片OCR处理成文字后, 其实就是个BERT, 其mask掉T1, T2, 然后需要在MLM (Masked Language Model) 部分预测其输出; 单看右边部分, 将图片resize成统一的$C\times H\times W$, 其实就是个BEiT, 其把图片打成patch, 然后mask掉V2, V3, 需要在MIM (Masked Image Model) 部分预测其输出. 实际上由于输入经过的是一个统一的transformer, 且没有mask attention, 因此两个模态会相互参考, 这就造成了模态的融合.

文章认为其的一个突出贡献是构造了一个WPA Head. 提出该head的原因是, 尽管在前面我们已经做了模态融合, 但实际上模型并没有学到一个明确的文本与patch对齐的知识. 因此, 该head的目的是: 预测每个没有被mask的token对应的patch是否被mask了. 听起来有点别扭, 但图其实也画了, T3, T4是没有被mask的token, 其对应的patch为V3, V4. 由于V3被mask了, 这里就会标注T3为unaligned, 同理T4标注为aligned.

## 细节

对于上述模型, 论文也用Loss函数来进一步注释, 假设词嵌入为$Y_{1:L}$, 图嵌入为$X_{1:M}$

$$
L_{MLM}(\theta)=-\sum_{l=1}^{L'}\log P_\theta(y_l|X^{M'},Y^{L'})\\
L_{MIM}(\theta)=-\sum_{m=1}^{M'}\log P_\theta(x_m|X^{M'},Y^{L'})\\
L_{WPA}(\theta)=-\sum_{l=1}^{L-L'}\log P_\theta(z_l|X^{M'},Y^{L'})
$$

假设被mask掉的词有$L'$个, 被mask掉的patch有$M'$个. 其中$y_l$为事件被mask掉的词是真实词, $x_m$为事件被mask掉的patch是正确的patch字典索引, $z_l$为事件align预测正确. 显然对于WPA的预测, 我们只需要预测没有被mask掉的, 即$L-L'$个.

总的loss是3个相加

$$
L=L_{MLM}+L_{MIM}+L_{WPA}
$$

具体而言, 训练时会mask掉$30\%$的文本, $40\%$的patch, 综上, 一个自监督模型就搭建完毕了.


## Reference

- [Paper](https://arxiv.org/abs/2204.08387)