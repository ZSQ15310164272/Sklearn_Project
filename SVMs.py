# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:38:14 2018

@author: ZSQ
"""

'''
支持向量机 (SVMs) 可用于以下监督学习算法 分类, 回归 和 异常检测.
'''

from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[0., 0.]]))

print(clf.support_)
print(clf.n_support_)


'''
多分类SVM
SVC, NuSVC 和 LinearSVC 能在数据集中实现多元分类
SVC 和 NuSVC 是相似的方法, 但是接受稍许不同的参数设置并且有不同的数学方程(在这部分看
数学公式). 另一方面, LinearSVC 是另一个实现线性核函数的支持向量分类. 记住 LinearSVC
不接受关键词 kernel, 因为它被假设为线性的. 它也缺少一些 SVC 和 NuSVC 的成员(members)
比如 support_ .和其他分类器一样, SVC, NuSVC 和 LinearSVC 将两个数组作为输入: 
[n_samples, n_features] 大小的数组 X 作为训练样本, [n_samples] 大小的数组 y 作为类别标签(字符串或者整数)
'''

'''
SVC 和 NuSVC 为多元分类实现了 “one-against-one” 的方法 (Knerr et al., 1990) 如果 
n_class 是类别的数量, 那么 n_class * (n_class - 1) / 2 分类器被重构, 而且每一个从
两个类别中训练数据. 为了给其他分类器提供一致的交互, decision_function_shape 选项允
许聚合 “one-against-one” 分类器的结果成 (n_samples, n_classes) 的大小到决策函数
'''

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes

'''
LinearSVC 实现 “one-vs-the-rest” 多类别策略, 从而训练 n 类别的模型. 如果只有两类, 
只训练一个模型
'''
lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y) 
dec = lin_clf.decision_function([[1]])
dec.shape[1]


'''
Regression
支持向量分类的方法可以被扩展用作解决回归问题. 这个方法被称作支持向量回归.
支持向量分类生成的模型(如前描述)只依赖于训练集的子集,因为构建模型的 cost 
function 不在乎边缘之外的训练点. 类似的,支持向量回归生成的模型只依赖于训
练集的子集, 因为构建模型的 cost function 忽略任何接近于模型预测的训练数据.
支持向量分类有三种不同的实现形式: SVR, NuSVR 和 LinearSVR. 在只考虑线性
核的情况下, LinearSVR 比 SVR 提供一个更快的实现形式, 然而比起 SVR 和 
LinearSVR, NuSVR 实现一个稍微不同的构思(formulation).细节参见 实现细节.
与分类的类别一样, fit方法会调用参数向量 X, y, 只在 y 是浮点数而不是整数型.
'''
from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y) 
print(clf.predict([[1,1]]))


'''
自定义核函数 SVM
'''

import numpy as np
from sklearn import svm
def my_kernel(X, Y):
    return np.dot(X, Y.T)

clf = svm.SVC(kernel=my_kernel)


'''
在适应算法中，设置 kernel='precomputed' 和把 X 替换为 Gram 矩阵。 此时，必须要提供
在 所有 训练矢量和测试矢量中的内核值。
'''
import numpy as np
from sklearn import svm
X = np.array([[0, 0], [1, 1]])
y = [0, 1]
clf = svm.SVC(kernel='precomputed')
# 线性内核计算
gram = np.dot(X, X.T)
clf.fit(gram, y) 
print(clf.predict(gram))






























