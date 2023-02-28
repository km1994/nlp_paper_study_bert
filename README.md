# 【关于 NLP】 那些你不知道的事——预训练模型篇

> 作者：杨夕
> 
> 介绍：研读顶会论文，复现论文相关代码
> 
> NLP 百面百搭 地址：https://github.com/km1994/NLP-Interview-Notes
> 
> **[手机版NLP百面百搭](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=3&sn=5d8e62993e5ecd4582703684c0d12e44&chksm=1bbff26d2cc87b7bf2504a8a4cafc60919d722b6e9acbcee81a626924d80f53a49301df9bd97&scene=18#wechat_redirect)**
> 
> 推荐系统 百面百搭 地址：https://github.com/km1994/RES-Interview-Notes
> 
> **[手机版推荐系统百面百搭](https://mp.weixin.qq.com/s/b_KBT6rUw09cLGRHV_EUtw)**
> 
> 搜索引擎 百面百搭 地址：https://github.com/km1994/search-engine-Interview-Notes 【编写ing】
> 
> NLP论文学习笔记：https://github.com/km1994/nlp_paper_study
> 
> 推荐系统论文学习笔记：https://github.com/km1994/RS_paper_study
> 
> GCN 论文学习笔记：https://github.com/km1994/GCN_study
> 
> **推广搜 军火库**：https://github.com/km1994/recommendation_advertisement_search
![](other_study/resource/pic/微信截图_20210301212242.png)

> 手机版笔记，可以关注公众号 **【关于NLP那些你不知道的事】** 获取，并加入 【NLP && 推荐学习群】一起学习！！！

> 注：github 网页版 看起来不舒服，可以看 **[手机版NLP论文学习笔记](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=1&sn=14d34d70a7e7cbf9700f804cca5be2d0&chksm=1bbff26d2cc87b7b9d2ed12c8d280cd737e270cd82c8850f7ca2ee44ec8883873ff5e9904e7e&scene=18#wechat_redirect)**

- [【关于 NLP】 那些你不知道的事——预训练模型篇](#关于-nlp-那些你不知道的事预训练模型篇)
  - [理论学习篇](#理论学习篇)
    - [【关于 预训练模型】 那些的你不知道的事](#关于-预训练模型-那些的你不知道的事)
      - [【关于Bert】 那些的你不知道的事：Bert论文研读](#关于bert-那些的你不知道的事bert论文研读)
      - [【关于 Bert 模型压缩】 那些你不知道的事](#关于-bert-模型压缩-那些你不知道的事)
      - [【关于中文预训练模型】那些你不知道的事](#关于中文预训练模型那些你不知道的事)
      - [【关于 Bert trick】那些你不知道的事](#关于-bert-trick那些你不知道的事)

## 理论学习篇

### 【关于 预训练模型】 那些的你不知道的事

#### [【关于Bert】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/)：Bert论文研读

- [【关于Bert】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/T1_bert/)
  - 阅读理由：NLP 的 创世之作
  - 动机：word2vec 的多义词问题 && GPT 单向 Transformer && Elmo 双向LSTM 
  - 介绍：Transformer的双向编码器
  - 思路：
    - 预训练：Task 1：Masked LM && Task 2：Next Sentence Prediction
    - 微调：直接利用 特定任务数据 微调
  - 优点：NLP 所有任务上都刷了一遍 SOTA
  - 缺点：
    - [MASK]预训练和微调之间的不匹配
    - Max Len 为 512
  - [【关于SpanBert】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/T1_bert/)
    - 论文：SpanBERT: Improving Pre-training by Representing and Predicting Spans
    - 论文地址：https://arxiv.org/abs/1907.10529
    - github：https://github.com/facebookresearch/SpanBERT
    - 动机：旨在更好地表示和预测文本的 span;
    - 论文方法->扩展了BERT：
      - （1）屏蔽连续的随机 span，而不是随机标记；
      - （2）训练 span 边界表示来预测屏蔽 span 的整个内容，而不依赖其中的单个标记表示。
- [【关于 XLNet 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/T2_XLNet/)
  - 阅读理由：Bert 问题上的改进
  - 动机：
    - Bert 预训练和微调之间的不匹配
    - Bert 的 Max Len 为 512
  - 介绍：广义自回归预训练方法
  - 思路：
    - 预训练：
      - Permutation Language Modeling【解决Bert 预训练和微调之间的不匹配】
      - Two-Stream Self-Attention for Target-Aware Representations【解决PLM出现的目标预测歧义】 
      - XLNet将最先进的自回归模型Transformer-XL的思想整合到预训练中【解决 Bert 的 Max Len 为 512】
    - 微调：直接利用 特定任务数据 微调
  - 优点：
  - 缺点：
- [【关于 Bart】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/BART/)
  - 论文：Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension
  - 来源：Facebook 
  - 论文地址：https://mp.weixin.qq.com/s/42rYlyjQsh4loFKRdhJlIg
  - 开源代码：https://github.com/renatoviolin/Bart_T5-summarization
  - 阅读理由：Bert 问题上的改进
  - 动机：
    - BERT：用掩码替换随机 token，双向编码文档。由于缺失 token 被单独预测，因此 BERT 较难用于生成任务;
    - GPT：使用自回归方式预测 token，这意味着 GPT 可用于生成任务。但是，该模型仅基于左侧上下文预测单词，无法学习双向交互
  - 介绍：用于预训练序列到序列模型的去噪自动编码器
  - 思路：
    - 预训练：
      - (1) 使用任意噪声函数破坏文本;
        - Token Masking（token 掩码）：按照 BERT 模型，BART 采样随机 token，并用 [MASK]标记 替换它们；
        - Sentence Permutation（句子排列变换）：按句号将文档分割成多个句子，然后以随机顺序打乱这些句子；
        - Document Rotation（文档旋转）：随机均匀地选择 token，旋转文档使文档从该 token 开始。该任务的目的是训练模型识别文档开头；
        - Token Deletion（token 删除）：从输入中随机删除 token。与 token 掩码不同，模型必须确定缺失输入的位置；
        - Text Infilling（文本填充）：采样多个文本段，文本段长度取决于泊松分布 (λ = 3)。用单个掩码 token 替换每个文本段。长度为 0 的文本段对应掩码 token 的插入；
      - (2) 学习模型以重建原始文本。
      - Two-Stream Self-Attention for Target-Aware Representations【解决PLM出现的目标预测歧义】 
      - XLNet将最先进的自回归模型Transformer-XL的思想整合到预训练中【解决 Bert 的 Max Len 为 512】
    - 微调：
      - Sequence Classification Task 序列分类任务: 将相同的输入，输入到encoder和decoder中，最后将decoder的最后一个隐藏节点作为输出，输入到分类层（全连接层）中，获取最终的分类的结果;
      - Token Classification Task 序列分类任务: 将完整文档输入到编码器和解码器中，使用解码器最上方的隐藏状态作为每个单词的表征。该表征的用途是分类 token;
      - Sequence Generation Task 序列生成任务: 编码器的输入是输入序列，解码器以自回归的方式生成输出;
      - Machine Translation 机器翻译: 将BART的encoder端的embedding层替换成randomly initialized encoder，新的encoder也可以用不同的vocabulary。通过新加的Encoder，我们可以将新的语言映射到BART能解码到English(假设BART是在English的语料上进行的预训练)的空间. 具体的finetune过程分两阶段:
        1. 第一步只更新randomly initialized encoder + BART positional embedding + BART的encoder第一层的self-attention 输入映射矩阵。
        2. 第二步更新全部参数，但是只训练很少的几轮。
  - 优点：它使用标准的基于 Transformer 的神经机器翻译架构，尽管它很简单，但可以看作是对 BERT（由于双向编码器）、GPT（带有从左到右的解码器）和许多其他最近的预训练方案的泛化.
  - 缺点：
- [【关于 RoBERTa】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/T4_RoBERTa/) 
  - 阅读理由：Bert 问题上的改进
  - 动机：
    - 确定方法的哪些方面贡献最大可能是具有挑战性的
    - 训练在计算上是昂贵的的，限制了可能完成的调整量
  - 介绍：A Robustly Optimized BERT Pretraining Approach 
  - 思路：
    - 预训练：
      - 去掉下一句预测(NSP)任务
      - 动态掩码
      - 文本编码
    - 微调：直接利用 特定任务数据 微调
  - 优点：
  - 缺点：
- [【关于 ELECTRA 】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/ELECTRA/)
  - 阅读理由：Bert 问题上的改进 【不推荐阅读，存在注水！】
  - 动机：
    - 只有15%的输入上是会有loss
  - 介绍：判别器 & 生成器 【但是最后发现非 判别器 & 生成器】
  - 思路：
    - 预训练：
      - 利用一个基于MLM的Generator来替换example中的某些个token，然后丢给Discriminator来判别
    - 微调：直接利用 特定任务数据 微调
  - 优点：
  - 缺点： 
- [【关于 Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/ACL2020_UnsupervisedBert/)
  - 论文链接：https://arxiv.org/pdf/2004.14786.pdf
  - 代码链接：https://github.com/bojone/perturbed_masking
  - 动机
    - 通过引入少量的附加参数，probe learns 在监督方式中使用特征表示（例如，上下文嵌入）来 解决特定的语言任务（例如，依赖解析）。这样的probe  tasks 的有效性被视为预训练模型编码语言知识的证据。但是，这种评估语言模型的方法会因 probe 本身所学知识量的不确定性而受到破坏
  - Perturbed Masking 
    - 介绍：parameter-free probing technique
    - 目标：analyze and interpret pre-trained models，测量一个单词xj对预测另一个单词xi的影响，然后从该单词间信息中得出全局语言属性（例如，依赖树）。
  - 整体思想很直接，句法结构，其实本质上描述的是词和词之间的某种关系，如果我们能从BERT当中拿到词和词之间相互“作用”的信息，就能利用一些算法解析出句法结构。
- [【关于 GRAPH-BERT】 那些你不知道的事](https://github.com/km1994/nlp_paper_study/bert_study/T2020_GRAPH_BERT))
  - 论文名称：GRAPH-BERT: Only Attention is Needed for Learning Graph Representations
  - 论文地址：https://arxiv.org/abs/2001.05140
  - 论文代码：https://github.com/jwzhanggy/Graph-Bert
  - 动机
    - 传统的GNN技术问题：
      - 模型做深会存在suspended animation和over smoothing的问题。
      - 由于 graph 中每个结点相互连接的性质，一般都是丢进去一个完整的graph给他训练而很难用batch去并行化。
  - 方法：提出一种新的图神经网络模型GRAPH-BERT (Graph based BERT)，该模型只依赖于注意力机制，不涉及任何的图卷积和聚合操作。Graph-Bert 将原始图采样为多个子图，并且只利用attention机制在子图上进行表征学习，而不考虑子图中的边信息。因此Graph-Bert可以解决上面提到的传统GNN具有的性能问题和效率问题。
- [【关于自训练 + 预训练 = 更好的自然语言理解模型 】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study/bert_study/SelfTrainingImprovesPreTraining))
  - 论文标题：Self-training Improves Pre-training for Natural Language Understanding
  - 论文地址：https://arxiv.org/abs/2010.02194
  - 动机 
    - 问题一: do  pre-training and self-training capture the same information,  or  are  they  complementary?
    - 问题二: how can we obtain large amounts of unannotated data from specific domains?
  - 方法
    - 问题二解决方法：提出 SentAugment 方法 从 web 上获取有用数据；
    - 问题一解决方法：使用标记的任务数据训练一个 teacher 模型，然后用它对检索到的未标注句子进行标注，并基于这个合成数据集训练最终的模型。
- [【关于 Bart】 那些你不知道的事](https://github.com/km1994/nlp_paper_study/bert_study/BART)
  - 论文名称：Revisiting Pre-trained Models for Chinese Natural Language Processing 
  - 会议：EMNLP 2020
  - 论文地址：https://arxiv.org/abs/2004.13922
  - 论文源码地址：https://github.com/ymcui/MacBERT
  - 动机：主要为了解决与训练阶段和微调阶段存在的差异性
  - 方法：
    - MLM
      - 使用Whole Word Masking、N-gram Masking：single token、2-gram、3-gram、4-gram分别对应比例为0.4、0.3、0.2、0.1；
      - 由于finetuning时从未见过[MASK]token，因此使用相似的word进行替换。使用工具Synonyms toolkit 获得相似的词。如果被选中的N-gram存在相似的词，则随机选择相似的词进行替换，否则随机选择任意词替换；
      - 对于一个输入文本，15%的词进行masking。其中80%的使用相似的词进行替换，10%使用完全随机替换，10%保持不变。
    - NSP
      - 采用ALBERT提出的SOP替换NSP
- [【关于 SpanBERT 】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study/bert_study/SpanBERT)
  - 论文名称：SpanBERT: Improving Pre-training by Representing and Predicting Spans
  - 会议：EMNLP 2020
  - 论文地址：https://arxiv.org/abs/1907.10529
  - 论文源码地址：https://github.com/facebookresearch/SpanBERT
  - 动机：旨在更好地表示和预测文本的 span;
  - 论文方法->扩展了BERT：
    - （1）屏蔽连续的随机跨度，而不是随机标记；
    - （2）训练跨度边界表示来预测屏蔽跨度的整个内容，而不依赖其中的单个标记表示。
  - 实验结果：
    - SpanBERT始终优于BERT和我们更好调整的基线，在跨选择任务（如问题回答和共指消解）上有实质性的收益。特别是在训练数据和模型大小与BERT-large相同的情况下，我们的单一模型在1.1班和2.0班分别获得94.6%和88.7%的F1。我们还实现了OntoNotes共指消解任务（79.6\%F1）的最新发展，在TACRED关系抽取基准测试上表现出色，甚至在GLUE上也有所提高。
- [【关于 Flan-T5 】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study/bert_study/FlanT5)
  - 论文名称：Scaling Instruction-Finetuned Language Models
  - 会议：
  - 论文地址：https://arxiv.org/abs/2210.11416
  - 论文源码地址：https://huggingface.co/google/flan-t5-xxl
  - 动机：是否有这种方案：通过在超大规模的任务上进行微调，让语言模型具备了极强的泛化性能，做到单个模型就可以在1800多个NLP任务上都能有很好的表现呢？即 实现 **One model for ALL tasks**？
  - Flan-T5 介绍：这里的Flan 指的是（Instruction finetuning ），即"基于指令的微调"；T5是2019年Google发布的一个语言模型了。注意这里的语言模型可以进行任意的替换（需要有Decoder部分，所以不包括BERT这类纯Encoder语言模型），论文的核心贡献是提出一套多任务的微调方案（Flan），来极大提升语言模型的泛化性。
  - Flan-T5 实现机制
    - step 1: 任务收集：收集一系列监督的数据，这里一个任务可以被定义成<数据集，任务类型的形式>，比如“基于SQuAD数据集的问题生成任务”。需要注意的是这里有9个任务是需要进行推理的任务，即Chain-of-thought （CoT）任务。
    - step 2: 形式改写：因为需要用单个语言模型来完成超过1800+种不同的任务，所以需要将任务都转换成相同的“输入格式”喂给模型训练，同时这些任务的输出也需要是统一的“输出格式”。
    - 训练过程：采用恒定的学习率以及Adafactor优化器进行训练；同时会将多个训练样本“打包”成一个训练样本，这些训练样本直接会通过一个特殊的“结束token”进行分割。训练时候在每个指定的步数会在“保留任务”上进行模型评估，保存最佳的checkpoint。
  - 总结：
    - 微调很重要
    - 模型越大效果越好
    - 任务越多效果越好
    - 混杂CoT相关的任务很重要
    - 整合起来

#### [【关于 Bert 模型压缩】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/Bert_zip)

- [【关于 Bert 模型压缩】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/Bert_zip)
  - 阅读理由：Bert 在工程上问题上的改进 
  - 动机：
    - 内存占用；
    - 功耗过高；
    - 带来很高的延迟；
    - 限制了 Bert 系列模型在移动和物联网等嵌入式设备上的部署；
  - 介绍：BERT 瘦身来提升速度
  - 模型压缩思路：
    - 低秩因式分解：在输入层和输出层使用嵌入大小远小于原生Bert的嵌入大小，再使用简单的映射矩阵使得输入层的输出或者最后一层隐藏层的输出可以通过映射矩阵输入到第一层的隐藏层或者输出层；
    - 跨层参数共享：隐藏层中的每一层都使用相同的参数，用多种方式共享参数，例如只共享每层的前馈网络参数或者只共享每层的注意力子层参数。默认情况是共享每层的所有参数；
    - 剪枝：剪掉多余的连接、多余的注意力头、甚至LayerDrop[1]直接砍掉一半Transformer层
    - 量化：把FP32改成FP16或者INT8；
    - 蒸馏：用一个学生模型来学习大模型的知识，不仅要学logits，还要学attention score；
  - 优点：BERT 瘦身来提升速度
  - 缺点： 
    - 精度的下降
    - 低秩因式分解 and 跨层参数共享 计算量并没有下降；
    - 剪枝会直接降低模型的拟合能力；
    - 量化虽然有提升但也有瓶颈；
    - 蒸馏的不确定性最大，很难预知你的BERT教出来怎样的学生；
- [【关于 Distilling Task-Specific Knowledge from BERT into Simple Neural Networks】那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/BERTintoSimpleNeuralNetworks/)
  - 动机：
    - 随着 BERT 的横空出世，意味着 上一代用于语言理解的较浅的神经网络（RNN、CNN等） 的 过时？
    - BERT模型是真的大，计算起来太慢了？
    - 是否可以将BERT（一种最先进的语言表示模型）中的知识提取到一个单层BiLSTM 或 TextCNN 中？
  - 思路：
      1. 确定 Teacher 模型（Bert） 和 Student 模型（TextCNN、TextRNN）;
      2. 蒸馏的两个过程：
         1. 第一，在目标函数附加logits回归部分；
         2. 第二，构建迁移数据集，从而增加了训练集，可以更有效地进行知识迁移。
- [【关于 AlBert 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/T5_ALBERT/)
  - 模型压缩方法：低秩因式分解 + 跨层参数共享
  - 模型压缩方法介绍：
    - 低秩因式分解：
      - 动机：Bert的参数量大部分集中于模型的隐藏层架构上，在嵌入层中只有30,000词块，其所占据的参数量只占据整个模型参数量的小部分；
      - 方法：将输入层和输出层的权重矩阵分解为两个更小的参数矩阵；
      - 思路：在输入层和输出层使用嵌入大小远小于原生Bert的嵌入大小，再使用简单的映射矩阵使得输入层的输出或者最后一层隐藏层的输出可以通过映射矩阵输入到第一层的隐藏层或者输出层；
      - 优点：在不显著增加词嵌入大小的情况下能够更容易增加隐藏层大小；
    - 参数共享【跨层参数共享】：
      - 动机：隐藏层 参数 大小 一致；
      - 方法：隐藏层中的每一层都使用相同的参数，用多种方式共享参数，例如只共享每层的前馈网络参数或者只共享每层的注意力子层参数。默认情况是共享每层的所有参数；
      - 优点：防止参数随着网络深度的增加而增大；
  - 其他改进策略：
    - **句子顺序预测损失(SOP)**代替**Bert中的下一句预测损失(NSP)**：
      - 动机：通过实验证明，Bert中的下一句预测损失(NSP) 作用不大；
      - 介绍：用预测两个句子是否连续出现在原文中替换为两个连续的句子是正序或是逆序，用于进一步提高下游任务的表现
  - 优点：参数量上有所降低；
  - 缺点：其加速指标仅展示了训练过程，由于ALBERT的隐藏层架构**采用跨层参数共享策略并未减少训练过程的计算量**，加速效果更多来源于低维的嵌入层；
- [【关于 FastBERT】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/FastBERT/)
  - 模型压缩方法：知识蒸馏
  - 模型压缩方法介绍：
    - 样本自适应机制（Sample-wise adaptive mechanism）
      - 思路：
        - 在每层Transformer后都去预测样本标签，如果某样本预测结果的置信度很高，就不用继续计算了，就是自适应调整每个样本的计算量，容易的样本通过一两层就可以预测出来，较难的样本则需要走完全程。
      - 操作：
        - 给每层后面接一个分类器，毕竟分类器比Transformer需要的成本小多了
    - 自蒸馏（Self-distillation）
      - 思路：
        - 在预训练和精调阶段都只更新主干参数；
        - 精调完后freeze主干参数，用分支分类器（图中的student）蒸馏主干分类器（图中的teacher）的概率分布
      - 优点：
        - 非蒸馏的结果没有蒸馏要好
        - 不再依赖于标注数据。蒸馏的效果可以通过源源不断的无标签数据来提升
- [【关于 distilbert】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/distilbert/)
- [【关于 TinyBert】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/TinyBERT/)
  - 模型压缩方法：知识蒸馏
  - tinybert的创新点：学习了teacher Bert中更多的层数的特征表示；
  - 模型压缩方法介绍：
    - 基于transformer的知识蒸馏模型压缩
      - 学习了teacher Bert中更多的层数的特征表示；
      - 特征表示：
        - 词向量层的输出；
        - Transformer layer的输出以及注意力矩阵；
        - 预测层输出(仅在微调阶段使用)；
    - bert知识蒸馏的过程
      - 左图：整体概括了知识蒸馏的过程
        - 左边：Teacher BERT；
        - 右边：Student TinyBERT
        - 目标：将Teacher BERT学习到的知识迁移到TinyBERT中
      - 右图：描述了知识迁移的细节；
        - 在训练过程中选用Teacher BERT中每一层transformer layer的attention矩阵和输出作为监督信息
  
- [【关于 Perturbed Masking】那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/ACL2020_UnsupervisedBert)
  - 论文：Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT
  - 论文链接：https://arxiv.org/pdf/2004.14786.pdf
  - 代码链接：https://github.com/bojone/perturbed_masking
  - 动机： 通过引入少量的附加参数，probe learns 在监督方式中使用特征表示（例如，上下文嵌入）来 解决特定的语言任务（例如，依赖解析）。这样的probe  tasks 的有效性被视为预训练模型编码语言知识的证据。但是，这种评估语言模型的方法会因 probe 本身所学知识量的不确定性而受到破坏。
  - 方法介绍：
    - Perturbed Masking 
      - 介绍：parameter-free probing technique
      - 目标：analyze and interpret pre-trained models，测量一个单词xj对预测另一个单词xi的影响，然后从该单词间信息中得出全局语言属性（例如，依赖树）。
  - 思想：整体思想很直接，句法结构，其实本质上描述的是词和词之间的某种关系，如果我们能从BERT当中拿到词和词之间相互“作用”的信息，就能利用一些算法解析出句法结构。 

#### [【关于中文预训练模型】那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/Chinese/)

- [【关于ChineseBERT】那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/Chinese/ChineseBERT/)
  - 论文名称：ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information
  - 会议： ACL2021
  - 论文地址：https://arxiv.org/abs/2106.16038
  - 论文源码地址：https://github.com/ShannonAI/ChineseBert
  - 模型下载：https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main
  - 动机：最近的中文预训练模型忽略了中文特有的两个重要方面：字形和拼音，它们为语言理解携带重要的句法和语义信息。
  - 论文工作：提出了 ChineseBERT，它将汉字的 {\it glyph} 和 {\it pinyin} 信息合并到语言模型预训练中。
    - embedding 层：将 字符嵌入（char embedding）、字形嵌入（glyph embedding）和拼音嵌入（pinyin embedding） 做拼接；
    - Fusion Layer 层：将 拼接后的 embedding 向量 做 Fusion 得到 一个 d 维的 Fusion embedding;
    - 位置拼接：将 Fusion embedding 和 位置嵌入（position embedding）、片段嵌入（segment embedding）相加；
    - Transformer-Encoder层;
  - 改进点：
    - 在底层的融合层（Fusion Layer）融合了除字嵌入（Char Embedding）之外的字形嵌入（Glyph Embedding）和拼音嵌入（Pinyin Embedding），得到融合嵌入（Fusion Embedding），再与位置嵌入相加，就形成模型的输入；
    - 抛弃预训练任务中的NSP任务。 由于预训练时没有使用NSP任务，因此模型结构图省略了片段嵌入（segment embedding）。实际上下游任务输入为多个段落时（例如：文本匹配、阅读理解等任务），是采用了segment embedding；
  - 实验结果：在大规模未标记的中文语料库上进行预训练，提出的 ChineseBERT 模型在训练步骤较少的情况下显着提高了基线模型的性能。 porpsoed 模型在广泛的中文 NLP 任务上实现了新的 SOTA 性能，包括机器阅读理解、自然语言推理、文本分类、句子对匹配和命名实体识别中的竞争性能。

#### [【关于 Bert trick】那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/bert_trick/)

- [【关于 Bert 未登录词处理】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_bert/tree/master/bert_study/bert_trick/UNK_process)
  - 动机
    - 中文预训练BERT 对于 英文单词覆盖不全问题。由于  中文预训练BERT 主要针对中文，所以词表中英文单词比较少，但是一般英文单词如果简单的直接使用tokenize函数，往往在一些序列预测问题上存在一些对齐问题，或者很多不存在的单词或符号没办法处理而直接使用　unk　替换了，某些英文单词或符号失去了单词的预训练效果；
    - 专业领域（如医疗、金融）用词或中文偏僻词问题。NLP经常会用到预训练的语言模型，小到word2vector，大到bert。现在bert之类的基本都用char级别来训练，然而由于 Bert 等预训练模型都是采用开放域的语料进行预训练，所以对词汇覆盖的更多是日常用词，专业用词则覆盖不了，这时候该怎么处理？
  - 方法
    - 方法一：直接在 BERT 词表 vocab.txt 中替换 [unused]
    - 方法二：通过重构词汇矩阵来增加新词
    - 方法三：添加特殊占位符号 add_special_tokens
  - 方法对比
    - 方法一：
      - 优点：如果存在大量领域内专业词汇，而且已经整理成词表，可以利用该方法批量添加；
      - 缺点：因为 方法1 存在 未登录词数量限制（eg：cased模型只有99个空位，uncased模型有999个空位），所以当 未登录词 太多时，将不适用；
    - 方法二：
      - 优点：不存在 方法1 的 未登录词数量限制 问题；
    - 方法三：
      - 优点：对于一些 占位符（eg：<e></e>），方法一和方法二可能都无法生效，因为 <, e, >和 <e></e>均存在于 vocab.txt，但前三者的优先级高于 <e></e>，而 add_special_tokens会起效，却会使得词汇表大小增大，从而需另外调整模型size。但是，如果同时在词汇表vocab.txt中替换[unused]，同时 add_special_tokens，则新增词会起效，同时词汇表大小不变。

