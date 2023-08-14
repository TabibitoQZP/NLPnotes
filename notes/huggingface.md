# Hugging Face

在NLP任务中, 我们需要学习transformer库的使用, 这个库之前我们主要是用它来拉预训练模型, 现在需要掌握一些具体的用法.

## Quick tour

在quick tour中, 其实很大程度上是学习`pipeline`的使用, 如下

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
out = classifier("We are very happy to show you the Transformers library.")
print(out)
"""
[{'label': 'POSITIVE', 'score': 0.9997994303703308}]
"""
```

可以看到, 尽管我们这里没有指定模型, 但指定了任务是`sentiment-analysis`, 因此会自动为我们选择一个模型, 并可以使用实例化的`classifier`作为分类器直接作用于句子, 并计算得到情感倾向. 这里还可以输入多个字符串构成的列表, 分类器会对每个句子都做分类.


## Reference

- [hugging face transformers](https://huggingface.co/docs/transformers/index)