# Huggingface中的layoutLMv3

主要通过huggingface中的layoutLMv3来学习一定的基本的transformers库的使用. 主要使用的参考文档是[LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3).

## 概览

这个模型在文档的multimodal models模块, 及多模态模型. 结构如下

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png)

具体介绍此前的笔记中有. 需要注意图中的细节, token的嵌入由`[CLS]`开头, 图片patch的嵌入由`[SPE]`开头. 在实际使用中, 只有token的嵌入或patch的嵌入都是可以的, 因为这个模型本身是multimodal的.

## 关于网络问题

实际使用中, 由于众所周知的原因. 因此huggingface的使用往往会遇到麻烦, 即预训练模型的下载和数据集的下载. 前者拉不下来对学习等有致命影响; 后者往往只是用作小规模测试, 但在学习中没有的话也有些不方便.

模型的话, 需要将[下载页面 (以layoutLMv3为例) ](https://huggingface.co/microsoft/layoutlmv3-base/tree/main)的内容全部下载下来, 存在同一个目录下 (假设就是项目的根目录下的`./model`) . 网络上有便捷的解决方案是使用`git lfs`去拉, 实测配好了还是拉不下来... 不过可以用git拉一部分, 然后缺的lfs手动下载. 下载完后打包发送到远程服务器. 而后, 如下就能加载了

```python
from transformers import LayoutLMv3Config, LayoutLMv3Model

config = LayoutLMv3Config()
model = LayoutLMv3Model(config).from_pretrained("./model")
```

而数据集, 则需要通过命令下载, 在网络可达的机器上使用如下指令

```python
from datasets import load_dataset

dataset = load_dataset("nielsr/funsd-layoutlmv3")
dataset.save_to_disk("./dataset")
```

这样, 数据集就保存在了`./dataset`目录下, 将其打包发送到远程服务器, 使用如下命令就可以加载了

```python
from datasets import load_from_disk

dataset = load_from_disk("./dataset")["train"]
example = dataset[0:2]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
```

需要注意的是, 这种操作加载下来的数据集组织结构需要自己稍微研究一下, 总体上和具体的数据集结构是类似的.

## 数据预处理

包括3个

- `LayoutLMv3ImageProcessor`: 对图片的预处理,实例化后默认会对图片做resize和normalize以及做OCR识别. 但做OCR需要加载`tesseract`插件, 倘若没有则会报错, 调用实例化对象时设置`apply_ocr=False`即可

- `LayoutLMv3Tokenizer`: 实例化的时候需要提供`vocab_file` 和`merges_file`. 实际中可以`from_pretrained`加载预训练模型对应的文件, 实际上打开此前的`./model`目录, 可以看到其中有`merges.txt`以及`vocab.json`文件. 调用实例化对象时实际上是较为复杂的, 分成`text`输入和`text_pair`输入. 这里所谓的`text_pair`是指做完token后的文本. 这里我们只考虑`text`的情况, 对于每一个输入的string, 都需要给定一个box

- `LayoutLMv3Processor`: 可以理解为总的处理模块, 实例化后既可以处理文本也可以处理图像, 这里需要注意的是, 其可以i添加NER模态, 即命名实体识别

对于上述操作, 这里抽象了一个总的类可以看看

```python
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3ImageProcessor, LayoutLMv3Tokenizer
from PIL import Image


class LayoutLMv3Proc:
    def __init__(self, modelPath="./model") -> None:
        config = LayoutLMv3Config()
        self.model = LayoutLMv3Model(config).from_pretrained(modelPath)

        self.imgProc = LayoutLMv3ImageProcessor()

        # Text processor need to load pretrained model.
        self.textProc = LayoutLMv3Tokenizer.from_pretrained(modelPath)

    def imageProcessor(self, imgPath, apply_ocr=False):
        """
        Need OCR plugin `tesseract`, or you need to set apply_ocr to false.
        Resize and normalize the imgage into BCHW.
        """
        img = Image.open(imgPath)
        return self.imgProc(img, apply_ocr=apply_ocr)

    def textProcessor(self, text, max_length=512):
        if type(text) == str:
            text = [text]
        # boxes are every token's box.

        boxes = []
        for lst in text:
            # we just assume every token are bounding in [0,0,0,0].
            boxes.append([[0, 0, 0, 0] for _ in range(len(lst))])
        # it will add [CLS], [END] to the start and end, and add [PAD] for padding.
        return self.textProc(text=text, boxes=boxes, padding="max_length", truncation=True, max_length=max_length)


if __name__ == "__main__":
    v3 = LayoutLMv3Proc()
    imgProcRet = v3.imageProcessor("./image/image.jpg")
    print(imgProcRet)
    textProcRet = v3.textProcessor(
        [["I", "Love", "You", "."], ["I", "Hate", "You", "."]])
    print(textProcRet)
```

为什么没有最后一个处理? 因为在文档中有相应操作且比较详细, 有时间再来补.
