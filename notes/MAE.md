# MAE

Masked AutoEncoder (MAE) , 注意这个和VAE完全不是一个东西...

## 介绍

MAE的基本思路很简单, 就是和BERT一样, 随机把一些patch给mask掉, 然后去预测这个patch里对应的像素. 在BERT中, 解码器本身是一个分类器, 用来预测mask掉的是一个什么样的token. 但在MAE中, 显然解码器得单独构造.

一个基本的工作流程如下

![MAE](https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png)

首先把input打成一个个patch, 而后, mask掉部分patch, 但具体操作就是把不mask掉的patch按位置拉直, 添加位置编码后输入encoder, 而后输出feature, 再在输入decoder前加入mask掉位置对应的feature, 后输入给decoder让其尝试重构图像.

这个和BEiT不一样, BEiT是会为每个patch赋予一个原始的离散值, 类似于语言中的token序列标号, 这样一整个训练也好预测也好, 就更BERT一点.

## 结构

分为编码器解码器, 在编码器部分, 会把图片打成patch, 然后随机挑选少量patch作为encoder的输入, 其余的视为被mask掉了. 每个patch先经过一个linear层做一次embedding, 而后与位置编码相加, 后输入encoder, 其实就是个标准的ViT.

而后是解码器, 对于mask掉的位置, 会给它一个统一的, 可训练的feature表示. 而后, 对其加上位置编码. 至于挑选出来的patch对应的feature, 文章没有说明其是否需要加位置编码. 而后, 再输入一个decoder里, 这个decoder的设计就比较灵活, 不强求transformer架构.

而这本质是个无监督训练模型, 由于设计时decoder从简, encoder从杂, 因此decoder能够学到足够的图像特征提取能力. 将该模型用于下游任务时, 只需要使用对应的解码器, 而无需使用编码器.