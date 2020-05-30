
## 中文词向量

使用预训练的词向量，repo：https://github.com/embedding/chinese-word-vectors

由于数据为新闻类，所以使用其中的 Sogou News 的 SGNS (Word + Character + Ngram) 对应的词向量。

### 关于 unk 的处理

即对于未登录词的处理。

在 issue ( https://github.com/Embedding/Chinese-Word-Vectors/issues/54 )中发现，该预训练的词向量并没有对 unk 词做词嵌入，而作者的建议是使用整体词向量的均值或者存在下游模型时随机初始化一个 unk 向量。

这里采用随机初始化的方法，固定一个向量作为 unk 向量。

