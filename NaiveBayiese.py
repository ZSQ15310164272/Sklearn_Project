# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:53:47 2018

@author: ZSQ
"""

'''
朴素贝叶斯方法是基于贝叶斯定理的一组有监督学习算法，即“简单”地假设每对特征之间相互独立。
我们可以使用最大后验概率(Maximum A Posteriori, MAP) 来估计 P(y) 和 P(x_i \mid y) ;
前者是训练集中类别 y 的相对频率。
各种各样的的朴素贝叶斯分类器的差异大部分来自于处理 P(x_i \mid y) 分布时的所做的假设不
同。尽管其假设过于简单，在很多实际情况下，朴素贝叶斯工作得很好，特别是文档分类和垃圾邮
件过滤。这些工作都要求 一个小的训练集来估计必需参数。(至于为什么朴素贝叶斯表现得好的
理论原因和它适用于哪些类型的数据，请参见下面的参考。)相比于其他更复杂的方法，朴素贝叶
斯学习器和分类器非常快。 分类条件分布的解耦意味着可以独立单独地把每个特征视为一维分布
来估计。这样反过来有助于缓解维度灾难带来的问题。另一方面，尽管朴素贝叶斯被认为是一种
相当不错的分类器，但却不是好的估计器(estimator)，所以不能太过于重视从 predict_proba
 输出的概率。
'''

'''
高斯朴素贝叶斯
'''

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
       % (iris.data.shape[0],(iris.target != y_pred).sum()))

'''
其他的假设包括如多项分布朴素贝叶斯
MultinomialNB 实现了服从多项分布数据的朴素贝叶斯算法，也是用于文本分类(这个领域中数
据往往以词向量表示，尽管在实践中 tf-idf 向量在预测时表现良好)的两大经典朴素贝叶斯算
法之一。 分布参数由每类 y 的 \theta_y = (\theta_{y1},\ldots,\theta_{yn}) 向量决定
式中 n 是特征的数量(对于文本分类，是词汇量的大小) \theta_{yi} 是样本中属于类 y 中特
征 i 概率 P(x_i \mid y) 。
参数 \theta_y 使用平滑过的最大似然估计法来估计，即相对频率计数:
\hat{\theta}_{yi} = \frac{ N_{yi} + \alpha}{N_y + \alpha n}
式中 N_{yi} = \sum_{x \in T} x_i 是 训练集 T 中 特征 i 在类 y 中出现的次数
'''

'''
伯努利朴素贝叶斯
BernoulliNB 实现了用于多重伯努利分布数据的朴素贝叶斯训练和分类算法，即有多个特征，但
每个特征 都假设是一个二元 (Bernoulli, boolean) 变量。 因此，这类算法要求样本以二元值
特征向量表示；如果样本含有其他类型的数据， 一个 BernoulliNB 实例会将其二值化(取决于 
binarize 参数)。
伯努利朴素贝叶斯的决策规则基于
P(x_i \mid y) = P(i \mid y) x_i + (1 - P(i \mid y)) (1 - x_i)
与多项分布朴素贝叶斯的规则不同 伯努利朴素贝叶斯明确地惩罚类 y 中没有出现作为预测因子
的特征 i ，而多项分布分布朴素贝叶斯只是简单地忽略没出现的特征。
'''