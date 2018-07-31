# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 22:40:28 2018

@author: ZSQ
"""

'''
神经网络模型（有监督）
'''

'''
多层感知器
多层感知器(MLP) 是一种监督学习算法，通过在数据集上训练来学习函数 
f(\cdot): R^m \rightarrow R^o，其中 m 是输入的维数，o 是输出的维数。 给定一组特征 
X = {x_1, x_2, ..., x_m} 和标签 y ，它可以学习用于分类或回归的非线性函数。 与逻辑
回归不同的是，在输入层和输出层之间，可以有一个或多个非线性层，称为隐藏层。
'''

'''
分类
MLPClassifier 类实现了通过 Backpropagation 进行训练的多层感知器（MLP）算法。
MLP 在两个 array 上进行训练:大小为 (n_samples, n_features) 的 array X 储存表示训
练样本的浮点型特征向量; 大小为 (n_samples,) 的 array y 储存训练样本的目标值（类别标签）
'''

from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
print(clf.predict([[2., 2.], [-1., -2.]]))
[coef.shape for coef in clf.coefs_]

'''
目前， MLPClassifier 只支持交叉熵损失函数，通过运行 predict_proba 方法进行概率估计。

MLP 算法使用的是反向传播的方式。 更准确地说，它使用了通过反向传播计算得到的梯度和某种
形式的梯度下降来进行训练。 对于分类来说，它最小化交叉熵损失函数，为每个样本 x 给出一个
向量形式的概率估计 P(y|x)
'''
clf.predict_proba([[2., 2.], [1., 2.]])


'''
MLPClassifier 通过应用 Softmax 作为输出函数来支持多分类。

此外，该模型支持 多标签分类 ，一个样本可能属于多个类别。 对于每个类，原始输出经过 
logistic 函数变换后，大于或等于 0.5 的值将进为 1，否则为 0。 对于样本的预测输出，
值为 1 的索引位置表示该样本的分类类别:
'''
X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15,), random_state=1)

clf.fit(X, y)        
clf.predict([[1., 2.]])
clf.predict([[0., 0.]])     


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()            
