# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 22:31:22 2018

@author: ZhaoGuangJun
"""

'''
多类和多标签算法
'''

'''
sklearn.multiclass 模块采用了 元评估器 ，通过把``多类`` 和 多标签 分类问题分解为 二
元分类问题去解决。这同样适用于多目标回归问题。

Multiclass classification 多类分类 意味着一个分类任务需要对多于两个类的数据进行分类
。比如，对一系列的橘子，
苹果或者梨的图片进行分类。多类分类假设每一个样本有且仅有一个标签：一个水果可以被归类
为苹果，也可以 是梨，但不能同时被归类为两类。

Multilabel classification 多标签分类 给每一个样本分配一系列标签。这可以被认为是预测不
相互排斥的数据点的属性，例如与文档类型相关的主题。一个文本可以归类为任意类别，例如可以
同时为政治、金融、 教育相关或者不属于以上任何类别。

Multioutput regression 多输出分类 为每个样本分配一组目标值。这可以认为是预测每一个样
本的多个属性，
比如说一个具体地点的风的方向和大小。

Multioutput-multiclass classification and multi-task classification **多输出-多
类分类和
多任务分类** 意味着单个的评估器要解决多个联合的分类任务。这是只考虑二分类的 multi-
label classification
和 multi-class classification 任务的推广。 此类问题输出的格式是一个二维数组或者一个
稀疏矩阵。
每个输出变量的标签集合可以是各不相同的。比如说，一个样本可以将“梨”作为一个输出变量的
值，这个输出变 量在一个含有“梨”、“苹果”等水果种类的有限集合中取可能的值；将“蓝色”或者
“绿色”作为第二个输出变量的值， 这个输出变量在一个含有“绿色”、“红色”、“蓝色”等颜色种类
的有限集合中取可能的值…

这意味着任何处理 multi-output multiclass or multi-task classification 任务的分类
器，在特殊的 情况下支持 multi-label classification 任务。Multi-task classification
 与具有不同模型公式 的 multi-output classification 相似。详细情况请查阅相关的分类器
 的文档。

所有的 scikit-learn 分类器都能处理 multiclass classification 任务， 但是 
sklearn.multiclass 提供的元评估器允许改变在处理超过两类数据时的方式，因为这会对分类
器的性能产生影响 （无论是在泛化误差或者所需要的计算资源方面）

下面是按照 scikit-learn 策略分组的分类器的总结，如果你使用其中的一个，则不需要此类中
的元评估器，除非你想要自定义的多分类方式。

固有的多类分类器:
sklearn.naive_bayes.BernoulliNB
sklearn.tree.DecisionTreeClassifier
sklearn.tree.ExtraTreeClassifier
sklearn.ensemble.ExtraTreesClassifier
sklearn.naive_bayes.GaussianNB
sklearn.neighbors.KNeighborsClassifier
sklearn.semi_supervised.LabelPropagation
sklearn.semi_supervised.LabelSpreading
sklearn.discriminant_analysis.LinearDiscriminantAnalysis
sklearn.svm.LinearSVC (setting multi_class=”crammer_singer”)
sklearn.linear_model.LogisticRegression (setting multi_class=”multinomial”)
sklearn.linear_model.LogisticRegressionCV (setting multi_class=”multinomial”)
sklearn.neural_network.MLPClassifier
sklearn.neighbors.NearestCentroid
sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
sklearn.neighbors.RadiusNeighborsClassifier
sklearn.ensemble.RandomForestClassifier
sklearn.linear_model.RidgeClassifier
sklearn.linear_model.RidgeClassifierCV
1对1的多类分类器:
sklearn.svm.NuSVC
sklearn.svm.SVC.
sklearn.gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_one”)
1对多的多类分类器:
sklearn.ensemble.GradientBoostingClassifier
sklearn.gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_rest”)
sklearn.svm.LinearSVC (setting multi_class=”ovr”)
sklearn.linear_model.LogisticRegression (setting multi_class=”ovr”)
sklearn.linear_model.LogisticRegressionCV (setting multi_class=”ovr”)
sklearn.linear_model.SGDClassifier
sklearn.linear_model.Perceptron
sklearn.linear_model.PassiveAggressiveClassifier
支持多标签分类的分类器:
sklearn.tree.DecisionTreeClassifier
sklearn.tree.ExtraTreeClassifier
sklearn.ensemble.ExtraTreesClassifier
sklearn.neighbors.KNeighborsClassifier
sklearn.neural_network.MLPClassifier
sklearn.neighbors.RadiusNeighborsClassifier
sklearn.ensemble.RandomForestClassifier
sklearn.linear_model.RidgeClassifierCV
支持多类-多输出分类的分类器:
sklearn.tree.DecisionTreeClassifier
sklearn.tree.ExtraTreeClassifier
sklearn.ensemble.ExtraTreesClassifier
sklearn.neighbors.KNeighborsClassifier
sklearn.neighbors.RadiusNeighborsClassifier
sklearn.ensemble.RandomForestClassifier
'''

'''
多标签分类格式
'''

from sklearn.preprocessing import MultiLabelBinarizer
y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
print(MultiLabelBinarizer().fit_transform(y))

'''
多类学习
'''
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

