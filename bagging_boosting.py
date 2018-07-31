# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 21:14:24 2018

@author: ZSQ
"""

'''
Bagging meta-estimator（Bagging 元估计器）
在集成算法中，bagging 方法会在原始训练集的随机子集上构建一类黑盒估计器的多个实例，然
后把这些估计器的预测结果结合起来形成最终的预测结果。 该方法通过在构建模型的过程中引入
随机性，来减少基估计器的方差(例如，决策树)。 在多数情况下，bagging 方法提供了一种非
常简单的方式来对单一模型进行改进，而无需修改背后的算法。 因为 bagging 方法可以减小过
拟合，所以通常在强分类器和复杂模型上使用时表现的很好（例如，完全决策树，fully 
developed decision trees），相比之下 boosting 方法则在弱模型上表现更好（例如，浅层
决策树，shallow decision trees）。

bagging 方法有很多种，其主要区别在于随机抽取训练子集的方法不同：

如果抽取的数据集的随机子集是样例的随机子集，我们叫做粘贴 (Pasting) [B1999] 。
如果样例抽取是有放回的，我们称为 Bagging [B1996] 。
如果抽取的数据集的随机子集是特征的随机子集，我们叫做随机子空间 (Random Subspaces) 
[H1998] 。
最后，如果基估计器构建在对于样本和特征抽取的子集之上时，我们叫做随机补丁 (Random 
Patches) [LG2012] 。
在 scikit-learn 中，bagging 方法使用统一的 BaggingClassifier 元估计器（或者 
BaggingRegressor ），输入的参数和随机子集抽取策略由用户指定。max_samples 和 
max_features 控制着子集的大小（对于样例和特征）， bootstrap 和 bootstrap_features 
控制着样例和特征的抽取是有放回还是无放回的。 当使用样本子集时，通过设置 oob_score=
True ，可以使用袋外(out-of-bag)样本来评估泛化精度。下面的代码片段说明了如何构造一个
 KNeighborsClassifier 估计器的 bagging 集成实例，每一个基估计器都建立在 50% 的样本
 随机子集和 50% 的特征随机子集上。
'''

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier, max_samples=0.5, 
                            max_features=0.5)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Settings
n_repeat = 50       # Number of iterations for computing expectations
n_train = 50        # Size of the training set
n_test = 1000       # Size of the test set
noise = 0.1         # Standard deviation of the noise
np.random.seed(0)

# Change this for exploring the bias-variance decomposition of other
# estimators. This should work well for estimators with high variance (e.g.,
# decision trees or KNN), but poorly for estimators with low variance (e.g.,
# linear models).
estimators = [("Tree", DecisionTreeRegressor()),
              ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor()))]

n_estimators = len(estimators)

# Generate data
def f(x):
    x = x.ravel()

    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))

        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

    X = X.reshape((n_samples, 1))

    return X, y

X_train = []
y_train = []

for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)

X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

# Loop over estimators to compare
for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_predict = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train[i], y_train[i])
        y_predict[:, i] = estimator.predict(X_test)

    # Bias^2 + Variance + Noise decomposition of the mean squared error
    y_error = np.zeros(n_test)

    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_predict[:, i]) ** 2

    y_error /= (n_repeat * n_repeat)

    y_noise = np.var(y_test, axis=1)
    y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
    y_var = np.var(y_predict, axis=1)

    print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
          " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                      np.mean(y_error),
                                                      np.mean(y_bias),
                                                      np.mean(y_var),
                                                      np.mean(y_noise)))

    # Plot figures
    plt.subplot(2, n_estimators, n + 1)
    plt.plot(X_test, f(X_test), "b", label="$f(x)$")
    plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

    for i in range(n_repeat):
        if i == 0:
            plt.plot(X_test, y_predict[:, i], "r", label="$\^y(x)$")
        else:
            plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

    plt.plot(X_test, np.mean(y_predict, axis=1), "c",
             label="$\mathbb{E}_{LS} \^y(x)$")

    plt.xlim([-5, 5])
    plt.title(name)

    if n == 0:
        plt.legend(loc="upper left", prop={"size": 11})

    plt.subplot(2, n_estimators, n_estimators + n + 1)
    plt.plot(X_test, y_error, "r", label="$error(x)$")
    plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
    plt.plot(X_test, y_var, "g", label="$variance(x)$"),
    plt.plot(X_test, y_noise, "c", label="$noise(x)$")

    plt.xlim([-5, 5])
    plt.ylim([0, 0.1])

    if n == 0:
        plt.legend(loc="upper left", prop={"size": 11})

plt.show()


'''
由随机树组成的森林
sklearn.ensemble 模块包含两个基于 随机决策树 的平均算法： RandomForest 算法和 
Extra-Trees 算法。 这两种算法都是专门为树而设计的扰动和组合技术（perturb-and-combine
techniques） [B1998] 。 这种技术通过在分类器构造过程中引入随机性来创建一组不同的分
类器。集成分类器的预测结果就是单个分类器预测结果的平均值。

与其他分类器一样，森林分类器必须拟合（fit）两个数组： 保存训练样本的数组（或稀疏或稠
密的）X，大小为 [n_samples, n_features]，和 保存训练样本目标值（类标签）的数组 Y，
大小为 [n_samples]
'''

from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)



'''
AdaBoost
模型 sklearn.ensemble 包含了流行的提升算法 AdaBoost, 这个算法是由 Freund and 
Schapire 在 1995 年提出来的 [FS1995].
AdaBoost 的核心思想是用反复修改的数据（校对者注：主要是修正数据的权重）来训练一系列
的弱学习器(一个弱学习器模型仅仅比随机猜测好一点, 比如一个简单的决策树),由这些弱学习器
的预测结果通过加权投票(或加权求和)的方式组合, 得到我们最终的预测结果。在每一次所谓的
提升（boosting）迭代中，数据的修改由应用于每一个训练样本的（新） 的权重 w_1, w_2, 
…, w_N 组成（校对者注：即修改每一个训练样本应用于新一轮学习器的权重）。 初始化时,将
所有弱学习器的权重都设置为 w_i = 1/N ,因此第一次迭代仅仅是通过原始数据训练出一个弱学
习器。在接下来的 连续迭代中,样本的权重逐个地被修改,学习算法也因此要重新应用这些已经修
改的权重。在给定的一个迭代中, 那些在上一轮迭代中被预测为错误结果的样本的权重将会被增
加，而那些被预测为正确结果的样本的权 重将会被降低。随着迭代次数的增加，那些难以预测的
样例的影响将会越来越大，每一个随后的弱学习器都将 会被强迫更加关注那些在之前被错误预测
的样例 

AdaBoost 既可以用在分类问题也可以用在回归问题中:

对于 multi-class 分类， AdaBoostClassifier 实现了 AdaBoost-SAMME 和 
AdaBoost-SAMME.R [ZZRH2009].
对于回归， AdaBoostRegressor 实现了 AdaBoost.R2 [D1997].
'''

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
print(scores.mean())


'''
Gradient Tree Boosting（梯度树提升）
Gradient Tree Boosting 或梯度提升回归树（GBRT）是对于任意的可微损失函数的提升算法的
泛化。 GBRT 是一个准确高效的现有程序， 它既能用于分类问题也可以用于回归问题。梯度树
提升模型被应用到各种领域，包括网页搜索排名和生态领域。
'''
'''
分类
GradientBoostingClassifier 既支持二分类又支持多分类问题。 下面的例子展示了如何训练
一个包含 100 个决策树弱学习器的梯度提升分类器:
'''

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0).fit(X_train, y_train)
print(clf.score(X_test, y_test))

'''
弱学习器(例如:回归树)的数量由参数 n_estimators 来控制；每个树的大小可以通过由参数 
max_depth 设置树的深度，或者由参数 max_leaf_nodes 设置叶子节点数目来控制。 
learning_rate 是一个在 (0,1] 之间的超参数，这个参数通过 shrinkage(缩减步长) 来控制
过拟合。
'''
'''
超过两类的分类问题需要在每一次迭代时推导 n_classes 个回归树。因此，所有的需要推导的
树数量等于 n_classes * n_estimators 。对于拥有大量类别的数据集我们强烈推荐使用 
RandomForestClassifier 来代替 GradientBoostingClassifier 。
'''


'''
 回归
对于回归问题 GradientBoostingRegressor 支持一系列 different loss functions ，这些
损失函数可以通过参数 loss 来指定；对于回归问题默认的损失函数是最小二乘损失函数（ 'ls' ）。
'''
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
     max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
print(mean_squared_error(y_test, est.predict(X_test))) 



'''
Voting Classifier（投票分类器）
VotingClassifier （投票分类器）的原理是结合了多个不同的机器学习分类器,并且采用多数表
决（majority vote）（硬投票） 或者平均预测概率（软投票）的方式来预测分类标签。 这样
的分类器可以用于一组同样表现良好的模型,以便平衡它们各自的弱点。

1.11.5.1. 多数类标签 (又称为 多数/硬投票)
在多数投票中，对于每个特定样本的预测类别标签是所有单独分类器预测的类别标签中票数占据
多数（模式）的类别标签。

例如，如果给定样本的预测是

classifier 1 -> class 1
classifier 2 -> class 1
classifier 3 -> class 2
类别 1 占据多数,通过 voting='hard' 参数设置投票分类器为多数表决方式，会得到该样本的
预测结果是类别 1 。

在平局的情况下,投票分类器（VotingClassifier）将根据升序排序顺序选择类标签。 例如，场
景如下:

classifier 1 -> class 2
classifier 2 -> class 1
这种情况下， class 1 将会被指定为该样本的类标签。
'''

from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                        voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], 
                      ['Logistic Regression', 'Random Forest', 
                       'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), 
          scores.std(), label))
    
'''
加权平均概率 （软投票）
与多数投票（硬投票）相比，软投票将类别标签返回为预测概率之和的 argmax 。
具体的权重可以通过权重参数 weights 分配给每个分类器。当提供权重参数 weights 时，收集
每个分类器的预测分类概率， 乘以分类器权重并取平均值。然后将具有最高平均概率的类别标签
确定为最终类别标签。
'''

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0,2]]
y = iris.target

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])

clf1 = clf1.fit(X,y)
clf2 = clf2.fit(X,y)
clf3 = clf3.fit(X,y)
eclf = eclf.fit(X,y)
