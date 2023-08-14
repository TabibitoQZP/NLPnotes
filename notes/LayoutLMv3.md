# LayoutLMv3

## 结构

首先看一下结构, 这应该是笔记里第一个多模态 (multi-modal) 的论文... 图片是原论文的, 取自hugging face.

![structure](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png)

这里图片其实做的很好, 我们就不考虑什么多模态, 单看左边的部分, 将图片OCR处理成文字后, 其实就是个BERT, 其mask掉T1, T2, 然后需要在MLM (Masked Language Model) 部分预测其输出; 单看右边部分, 将图片resize成统一的$C\times H\times W$, 其实就是个BEiT, 其把图片打成patch, 然后mask掉V2, V3, 需要在MIM (Masked Image Model) 部分预测其输出. 实际上由于输入经过的是一个统一的transformer, 且没有mask attention, 因此两个模态会相互参考, 这就造成了模态的融合.

文章认为其的一个突出贡献是构造了一个WPA Head. 提出该head的原因是, 尽管在前面我们已经做了模态融合, 但实际上模型并没有学到一个明确的文本与patch对齐的知识. 因此, 该head的目的是: 预测每个没有被mask的token对应的patch是否被mask了. 听起来有点别扭, 但图其实也画了, T3, T4是没有被mask的token, 其对应的patch为V3, V4. 由于V3被mask了, 这里就会标注T3为unaligned, 同理T4标注为aligned.

## loss

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

# 

## HuggingFace使用

需要指出, 懂了论文原理和会用还是有较大差距. 我们这里使用v3-base版本, 首先加载一下预训练模型. 这里试了一下, 使用transformer直接load是加载不下来的, 会有网络错误. 且直接用git去拖, lfs是拖不下来的. 这部分只能手动下, 在如下页面

[download link](https://huggingface.co/microsoft/layoutlmv3-base/tree/main)

下载到统一的`./model`目录下, 供后续使用. huggingface实际上给了如何使用layoutLM, 有2个版本, 一个是下载页面的

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("model")
```

还有就是layoutlmv3教程界面的. 其既有前者, 也有如下的加载方式. 下列加载方式实际上是随机加载权重的, 也可以`LayoutLMv3Model.from_pretrained("model")`加载预训练的权重

```python
from transformers import LayoutLMv3Model
model = LayoutLMv3Model()
```

后面的教程统一使用教程页面的内容. 同样由于网络问题, 实际上不能很方便的使用官网教程的操作. 在此之前, 需要下载数据集, 本地下载代码如下

```python
from datasets import load_dataset

dataset = load_dataset("nielsr/funsd-layoutlmv3")
dataset.save_to_disk("./dataset")
```

上述会把内容下载到本地`./dataset`目录下, 后将其压缩上传, 并解压到目录中. 使用`load_from_disk`, 给的操作如下

```python
from transformers import AutoProcessor, AutoModel
from datasets import load_from_disk
import numpy as np

processor = AutoProcessor.from_pretrained("./model", apply_ocr=False)
model = AutoModel.from_pretrained("./model")

dataset = load_from_disk("./dataset")["train"]
example = dataset[0:2]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]

encoding = processor(image, words, boxes=boxes, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

outputs = model(**encoding)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.size())
```

预处理阶段的输入包括, 图片本身(BHWC), 以及单词及其对应的box. 如果我们需要自己做dataset, 那么就需要整出这些东西来. 如果需要使用他自己的OCR, 将`apply_ocr`置为`True`, 这样的话, 就只需要输入image了. 如下

```python
from transformers import AutoProcessor, AutoModel
from PIL import Image

processor = AutoProcessor.from_pretrained("./model", apply_ocr=True)
model = AutoModel.from_pretrained("./model")

image = Image.open("./image.jpg")

encoding = processor(image, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

outputs = model(**encoding)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.size())
```

需要注意的是, 这样需要一些OCR识别库, 不过在错误提示中会给出.

另一方面, 就是所谓的`encoding`中包含的东西, 这个在教程中有解释. 重要的如下

- `input_ids`: 输入的序列加入cls以及padding后的token对应id序列, 这里需要注意的是, 文档中解释1是`[CLS]`. 但实际的输出来看, 1对应的是padding部分, 0对应的才是`[CLS]`

- `attention_mask`: 对于序列中padding的部分, 其mask为0, 否则为1

- `bbox`: 每个token在图片中都有一个对应的框, 这个对应的就是位置, 注意这里的图片会被resize成`224*224`, 这里的位置也是resize后的位置. 同时, 对于padding和`[CLS]`位置, 其bbox规定为`[0,0,0,0]`

- `pixel_values`: BCHW格式, 在这里的处理中, 会线性映射到-1到1

最后就是`outputs`, 其可以输出的东西有3个, 但后两个需要指定输出, 因此重要的是下列的隐藏层状态

- `last_hidden_state`: 格式为`batch_size, sequence_length, hidden_size`, 这里的序列长为, token长度+图片开始符`[SPE]`+图片打成patch的个数 (这里是196)

## Reference

- [paper](https://arxiv.org/abs/2204.08387)
- [huggingface](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)