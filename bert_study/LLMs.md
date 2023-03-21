# LLMs

## 一、模型一览

![](img/20230321223629.png)

## LLaMA

- 论文名称：LLaMA: Open and Efficient Foundation Language Models
- 作者：Meta/Facebook AI
- 论文地址：https://arxiv.org/pdf/2302.13971v1.pdf
- 论文github：https://github.com/facebookresearch/llama
- 时间：2023.02.27
- 论文介绍：介绍了一系列公开发布的语言模型，这些模型与最先进的基础模型具有竞争力。最值得注意的是，LLaMA-13B比GPT-3小10倍以上，并且LLaMA-65B与Chinchilla-70B和PaLM-540B具有竞争力。与之前的研究不同，通过仅在公开可用数据上进行训练，而无需使用专有数据集，就可以实现最先进的性能。向研究界发布这些模型将加速大型语言模型的开发，并有助于提高其鲁棒性，缓解毒性和偏见等已知问题。此外，像Chung等人（2022）一样观察到，根据指令微调这些模型会产生有希望的结果，计划在未来的工作中进一步研究这一点。最后，计划在未来发布在更大的预训练语料库上训练的更大模型，因为在扩展时看到了性能的不断提高。

## OPT

- 论文名称：OPT: Open Pre-trained Transformer Language Models
- 作者：Meta/Facebook AI
- 论文地址：https://arxiv.org/pdf/2205.01068.pdf
- 论文github：https://github.com/facebookresearch/metaseq
- 时间：
- 论文介绍：介绍了 OPT，这是一个自回归语言模型的集合，参数大小从 125M 到 175B 不等。 我们的目标是复制 GPT-3 类模型的性能和大小，同时在数据管理和训练效率方面应用最新的最佳实践。 我们描述了训练细节，评估了许多 NLP 和对话设置中的表现，并描述了关于偏见、毒性和仇恨言论的行为。 我们还描述了模型的许多其他限制，并讨论了负责任地发布模型的一系列广泛考虑因素。 我们相信整个 AI 社区将从共同努力为负责任的 LLM 制定指南中受益，我们希望广泛使用这些类型的模型将增加定义此类技术的伦理考虑的声音的多样性。
- 论文结论：
  - 结构：基于Transformers的模型结构，进行堆叠；
  - 模型大小：从125M 到 175B，大致匹配 GPT-3 类模型的性能和大小；
  - 训练语料：主要包含英语文本，存在少量非英语数据；
  - 数据量：GPT-2 BPE tokenizer，大约 180B 个tokens;
  - 训练效率：仅为GPT-3训练所需计算量的1/7；
  - 性能评估avg. accuracy：
    - Zero-shot: 与GPT-3 模型的表现相似，有些任务表现出高度不稳定
    - Multishot-shot: 大多数任务的性能与 GPT-3 大致相似，一些任务表现出不稳定的行为
  - 能力：诗歌生成，对话生成，few-shot翻译， 论文写作，算术，Python 编程。

## T5

- 论文名称：T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- 作者：Google
- 论文地址：https://arxiv.org/pdf/1910.10683.pdf
- 论文github：https://github.com/google-research/text-to-text-transfer-transformer
- 时间：
- 论文介绍：介引入一个统一的自然语言处理迁移框架，该框架将所有的自然言语处理任务统一为text-to-text形式。本文系统研究比较了 预训练目标、系统架构、未标记数据集和迁移方法等其他因素对数十个自然语言理解任务的影响。本文其实并没有引入新的模型或者新的方法，而是将现有的方法和技术做一次集大成，进行统一。此外，本文还引入一个新的数据集：Colossal Clean Crawled Corpus，名为C4。该数据集引入的初衷是为了探索尺度规模(包括模型规模和数据规模)在NLP中的影响。本文最终在文本摘要、问答、文本分类等多个基准任务上取得SOTA结果。
- 任务：机器翻译、问答、生成式摘要、文本分类(单句&双句)
- 数据格式：
  - 输入：参考GPT2，直接把任务名称当作prefix和输入拼在一起
  - 输出：分类任务(如推断)，需要输出"entailment", "neutral", "contradiction"这三种文本，否则都算错；回归任务输出str类型的浮点数。还有其他任务，
- 训练
  - 预训练
    - 参考SpanBERT，mask掉15%，平均长度为3的span
    - 训练更长步数，1百万步*1024个样本
    - 使用Multi-task预训练，即混入在无监督数据中混入一定比例的任务数据
  - 精调
    - 也是Multi-task，将所有GLUE/SuperGLUE的数据拼在一起变成精调一个task，减少过拟合，但同时也会牺牲一些精度
    - batch size减小到8
    - 其实最后同时进行了多任务精调和单独精调，根据dev集选择最好的结果
  - 解码
    - 大部分使用Greedy decoding，对于输出句子较长的任务使用beam search

## mT5

- 论文名称：mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer
- 作者：Google
- 论文地址：https://arxiv.org/pdf/2010.11934.pdf
- 论文github：https://huggingface.co/models?search=mt5
- 时间：2021
- 论文介绍：最近的“文本到文本传输转换器”（T5）利用统一的文本到文本格式和规模，在各种英语NLP任务上获得了最先进的结果。在本文中，我们介绍了 mT5，这是 T5 的多语言变体，它是在涵盖 101 种语言的基于 Common Crawl 的新数据集上预先训练的。我们详细介绍了mT5的设计和修改训练，并在许多多语言基准测试中展示了其最先进的性能。我们还描述了一种简单的技术，以防止在零镜头设置中“意外翻译”，其中生成模型选择（部分）将其预测翻译成错误的语言。这项工作中使用的所有代码和模型检查点都是公开的。

## UL2 and Flan-UL2

- 论文名称：UL2 and Flan-UL2: Unifying Language Learning Paradigms
- 作者：Google
- 论文地址：https://arxiv.org/pdf/2205.05131.pdf
- blog：https://www.yitay.net/blog/flan-ul2-20b
- 论文github：
  - https://huggingface.co/google/ul2
  - https://huggingface.co/google/flan-ul2
- 时间：2021
- 论文介绍：
  - 1、提出了一种新的降噪器混合 (MoD) 预训练，它将多个预训练任务混合。
  - 2、引入了模式切换，一种将下游任务行为与上游预训练相关联的方法。
- 论文效果：UL2 在大多数的有监督和少样本任务上始终优于 GPT 类模型和T5模型，在9个任务上优于 T5，归一化后的整体增益提升76.1%。最后，UL2 扩展到20B参数，并在60 个 NLP 任务进行了多样化的实验。结果表明，UL2 在其中的 50 个下游任务上都实现了SOTA的性能。




## 参考

1. [总结当下可用的大模型LLMs]()