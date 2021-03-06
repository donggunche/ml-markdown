
## 朴素贝叶斯讨论

> [@Time渐行渐远](https://github.com/Timehsw) [@那伊抹微笑](https://github.com/wangyangting) [@小瑶](https://github.com/chenyyx) [@如果迎着风就飞](https://github.com/orgs/apachecn/people/mikechengwei)

朴素贝叶斯就是用来求逆向概率的（已知）。

1. 根据训练数据集求（正向）概率。
2. 根据测试数据集求（逆向）概率（根据 贝叶斯公式）。
3. 求出的逆向概率，哪个大，就属于哪个类别。

### 疑问 1
通过训练集求出了各个特征的概率, 然后测试集的特征和之前求出来的概率相乘, 这个就代表这个测试集的特征的概率了.
有了这个基础后, 通过贝叶斯公式, 就可以得到这个测试集的特征属于哪个类别了, 他们相乘的依据是什么？

```
朴素贝叶斯？
条件独立性啊
朴素贝叶斯不是基于两个定理吗
一个是假设 条件独立性
一个是 贝叶斯定理
条件独立性  所以每个特征相乘得到的概率 就是这个数据的概率
```

### 疑问 2
凭啥测试集的特征乘以训练集的概率就是测试集的概率了.这么做的理论依据是什么？

```
朴素贝叶斯就是利用先验知识来解决后验概率，因为训练集中我们已经知道了每个单词在类别0和1中的概率，即p(w|c).
我们就是要利用这个知识去解决在出现这些单词的组合情况下，类别更可能是0还是1,即p(c|w).
如果说之前的训练样本少, 那么这个 p(w|c) 就更可能不准确, 所以样本越多我们会觉得这个 p(w|c) 越可信.
```
