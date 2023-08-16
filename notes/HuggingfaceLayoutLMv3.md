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

- `LayoutLMv3Processor`: 可以理解为总的处理模块, 实例化后既可以处理文本也可以处理图像, 这里需要注意的是, 其可以i添加NER模态, 即命名实体识别. 在介绍中, 其表明了这个模块提供了所有模型输入需要的东西

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

为什么没有最后一个样列? 因为实际上最后一个样列就是提供给前两个样列所需要的数据, 然后将各自处理好的数据整合返回, 因此在这写比较繁琐. 需要注意的是, 使用时同样可以输入`apply_ocr`参数, 倘若设置了使用OCR, 则无需输入额外的文本及其标注框.

## 模型使用

所有的预训练模型使用都大同小异, 关键在于掌握实例化后的输入以及输出. 模型有直接输出hidden states的, 也有用于下游任务的. 这里只介绍模型使用

```python
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3Processor
from datasets import load_from_disk


class ModelUse:
    def __init__(self, modelPath="./model", datasetPath="./dataset"):
        self.modelPath = modelPath
        self.data = load_from_disk(datasetPath)
        self.proc = LayoutLMv3Processor.from_pretrained(
            modelPath, apply_ocr=False)

        self.config = LayoutLMv3Config

    def dataShow(self):
        for k in self.data.keys():
            print(k)
            cnt = 0
            for data in self.data[k]:
                print(data)
                cnt += 1
                if cnt >= 3:
                    break

    def hiddenState(self):
        data = self.data["train"][:2]
        dataFormatted = {
            "images": data["image"],
            "text": data["tokens"],
            "boxes": data["bboxes"],
            "padding": "max_length",
            "max_length": 512,
            "truncation": True,
            "return_tensors": "pt"
        }
        dataProced = self.proc(**dataFormatted)
        model = LayoutLMv3Model.from_pretrained(self.modelPath)
        return model(**dataProced, output_hidden_states=True, output_attentions=True)


if __name__ == "__main__":
    mu = ModelUse()
    mu.dataShow()
    mu.hiddenState()
```

这里注意huggingface比较喜欢的传参方式, 使用字典.

## 实战: 在FUNSD上微调LayoutLMv3

官方写了一个非常详细 (甚至过于详细) 的微调版本[代码](https://github.com/huggingface/transformers/blob/main/examples/research_projects/layoutlmv3/run_funsd_cord.py). 包括各种版本检查, 参数输入等, 也提供了funetune完成后的权重. 我们这里自己写着用倒不用整那么多没用的, 参考main函数写就行了.

在写之前需要了解一下FUNSD (Form Understanding in Noisy Scanned Documents) . 看名字其实基本内容已经很清楚了, 这是个用于在噪声扫描情况下对表单的标注数据集. 表单中的内容主要是一些NER标注, 可视化一下就如下

![](https://guillaumejaume.github.io/FUNSD/img/two_forms.png)

比如这里蓝色部分代表表单问题, 绿色代表用户填写的信息, 橙色代表标题, 紫色代表一些其他信息 (纯属瞎猜) . 总之, 反映在具体的数据集上, 会有文本的bounding box坐标, 文本对应的token, 文本对应类别标签.

因此, 使用FUNSD来微调, 只能是针对NER类别标签进行token classification的任务. 幸运的是, huggingface提供了用于token classification的模型, 加载这个模型并加载预训练模型的权重, 则可以基于FUNSD来微调token classification模型. 如下

```python
import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import load_from_disk
from gpuGrab import GPUGrab


class FinetuneFUNSD:
    def __init__(self, modelPath="./model", datasetPath="./dataset", device="cuda"):
        """
        For the pretrained model, we do not change any configurations.
        """
        device = device.lower()
        if device == "cuda":
            gg = GPUGrab()
            self.device = f"{device}:{gg.autoCheck()[0]}"

        config = LayoutLMv3Config()  # use default configuration
        self.model = LayoutLMv3ForTokenClassification(
            config).from_pretrained(modelPath, num_labels=7).to(device)
        self.dataset = load_from_disk(datasetPath)
        self.proc = LayoutLMv3Processor.from_pretrained(
            modelPath, apply_ocr=False)

    def finnetune(self, epoch=16, batchSize=4):
        trainData = self.dataset["train"].map(lambda e: self.proc(
            e['image'], e['tokens'], boxes=e['bboxes'], word_labels=e['ner_tags'], padding="max_length", max_length=512, truncation=True), batched=True)

        print(trainData.features)
        trainData.set_format(type='torch', columns=[
                             "input_ids", "attention_mask", "bbox", "pixel_values", "labels"])

        # attention, every element should be the same size
        trainLoader = DataLoader(
            dataset=trainData, shuffle=True, drop_last=True, batch_size=batchSize)

        # optim
        optm = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        for e in range(epoch):
            print(e)
            for x in trainLoader:
                for k in x.keys():
                    x[k] = x[k].to(self.device)

                out = self.model(**x, output_hidden_states=False,
                                 output_attentions=False)
                optm.zero_grad()
                print(out.loss)
                out.loss.backward()
                optm.step()
        self.model.save_pretrained("./finetuned")


if __name__ == "__main__":
    ft = FinetuneFUNSD()
    ft.finnetune()
```

上面的`.map`命令要掌握, 因为从Dataset类中的原始数据很可能只是基础的数据类型, 需要转成tensor. 会使用`lambda`指令来定义简易函数. 看样例应该很好理解, 就不多解释了. 同时, `set_format`可以过滤出需要的栏目来. 因为前面的处理后原始数据也会给一个dict对应的key和value予以保留, 我们后面使用`**`符号传参不能传入未定义的变量, 因此需要取出必要的. 而后的训练则和pytorch一致了, 最后, 要模型保存也挺方便的, 用`.save_pretrained`保存到目录中即可, 加载也是从该目录加载.
