# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 21:03:25 2018

@author: ZSQ
"""

'''
Decision Trees (DTs) 是一种用来 classification 和 regression 的无参监督学习方法。
其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值。
'''

'''
分类
DecisionTreeClassifier 是能够在数据集上执行多分类的类,与其他分类器一样，
DecisionTreeClassifier 采用输入两个数组：数组X，用 [n_samples, n_features] 的方式
来存放训练样本。整数值数组Y，用 [n_samples] 来保存训练样本的类标签:
'''

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([[2., 2.]]))
print(clf.predict_proba([[2., 2.]]))


'''
DecisionTreeClassifier 既能用于二分类（其中标签为[-1,1]）也能用于多分类（其中标签为
[0,…,k-1]）。使用Lris数据集，我们可以构造一个决策树，如下所示:
'''
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
print(clf.predict(iris.data[:1, :]))
print(clf.predict_proba(iris.data[:1, :]))


'''
决策树通过使用 DecisionTreeRegressor 类也可以用来解决回归问题。如在分类设置中，拟合
方法将数组X和数组y作为参数，只有在这种情况下，y数组预期才是浮点值:
'''

from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
print(clf.predict([[1, 1]]))


# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()



