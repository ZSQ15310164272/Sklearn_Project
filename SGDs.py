# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 19:25:10 2018

@author: ZSQ
"""

'''
随机梯度下降(SGD) 是一种简单但又非常高效的方法，主要用于凸损失函数下线性分类器的判别
式学习，例如(线性) 支持向量机 和 Logistic 回归 。
'''

'''
SGDClassifier 类实现了一个简单的随机梯度下降学习例程, 支持不同的 loss functions（损
失函数）和 penalties for classification（分类处罚）。

具体的 loss function（损失函数） 可以通过 loss 参数来设置。 SGDClassifier 支持以下
的 loss functions（损失函数）：
loss="hinge": (soft-margin) linear Support Vector Machine （（软-间隔）线性支持向
量机），loss="modified_huber": smoothed hinge loss （平滑的 hinge 损失），loss=
"log": logistic regression （logistic 回归），and all regression losses below
（以及所有的回归损失）。

具体的惩罚方法可以通过 penalty 参数来设定。 SGD 支持以下 penalties（惩罚）:

penalty="l2": L2 norm penalty on coef_.
penalty="l1": L1 norm penalty on coef_.
penalty="elasticnet": Convex combination of L2 and L1（L2 型和 L1 型的凸组合）;
 (1 - l1_ratio) * L2 + l1_ratio * L1.
'''

from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
print(clf.predict([[2., 2.]]))
print(clf.coef_)
print(clf.intercept_)

# 使用 SGDClassifier.decision_function 来获得到此超平面的 signed distance (符号距离)
print(clf.decision_function([[2., 2.]]))

'''
使用 loss="log" 或者 loss="modified_huber" 来启用 predict_proba 方法, 其给出每个
样本 x 的概率估计 P(y|x) 的一个向量：
'''
clf = SGDClassifier(loss="log").fit(X, y)
print(clf.predict_proba([[1., 1.]]))


'''
SGDRegressor 类实现了一个简单的随机梯度下降学习例程，它支持用不同的损失函数和惩罚来
拟合线性回归模型。 SGDRegressor 非常适用于有大量训练样本（>10.000)的回归问题，对于
其他问题，我们推荐使用 Ridge ，Lasso ，或 ElasticNet 。

具体的损失函数可以通过 loss 参数设置。 SGDRegressor 支持以下的损失函数:
loss="squared_loss": Ordinary least squares（普通最小二乘法）,
loss="huber": Huber loss for robust regression（Huber回归）,
loss="epsilon_insensitive": linear Support Vector Regression（线性支持向量回归）.
Huber 和 epsilon-insensitive 损失函数可用于 robust regression（鲁棒回归）。不敏感
区域的宽度必须通过参数 epsilon 来设定。这个参数取决于目标变量的规模。
Huber 和 epsilon-insensitive 损失函数可用于 robust regression（鲁棒回归）。不敏感
区域的宽度必须通过参数 epsilon 来设定。这个参数取决于目标变量的规模。
'''







