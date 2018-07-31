# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 19:59:55 2018

@author: ZSQ
"""

'''
sklearn.neighbors 提供了 neighbors-based (基于邻居的) 无监督学习以及监督学习方法的
功能。 无监督的最近邻是许多其它学习方法的基础，尤其是 manifold learning (流行学习) 
和 spectral clustering (谱聚类)。 neighbors-based (基于邻居的) 监督学习分为两种： 
classification （分类）针对的是具有离散标签的数据，regression （回归）针对的是具有
连续标签的数据。

最近邻方法背后的原理是从训练样本中找到与新点在距离上最近的预定数量的几个点，然后从这
些点中预测标签。 这些点的数量可以是用户自定义的常量（K-最近邻学习）， 也可以根据不同
的点的局部密度（基于半径的最近邻学习）。距离通常可以通过任何度量来衡量： standard 
Euclidean distance（标准欧式距离）是最常见的选择。Neighbors-based（基于邻居的）方法
被称为 非泛化 机器学习方法， 因为它们只是简单地”记住”了其所有的训练数据（可能转换为一
个快速索引结构，如 Ball Tree 或 KD Tree）
'''

'''
1.6.1. 无监督最近邻
NearestNeighbors （最近邻）实现了 unsupervised nearest neighbors learning（无监督
的最近邻学习）。 它为三种不同的最近邻算法提供统一的接口：BallTree, KDTree, 还有基于 
sklearn.metrics.pairwise 的 brute-force 算法。算法的选择可通过关键字 'algorithm' 
来控制， 并必须是 ['auto', 'ball_tree', 'kd_tree', 'brute'] 其中的一个。当默认值设
置为 'auto' 时，算法会尝试从训练数据中确定最佳方法。
'''
from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
nbrs.kneighbors_graph(X).toarray()


'''
scikit-learn 实现了两种不同的最近邻分类器：KNeighborsClassifier 基于每个查询点的 k 
个最近邻实现，
其中 k 是用户指定的整数值。RadiusNeighborsClassifier 基于每个查询点的固定半径 r 内
的邻居数量实现， 其中 r 是用户指定的浮点数值。
k -邻居分类是 KNeighborsClassifier 下的两种技术中比较常用的一种。k 值的最佳选择是高
度依赖数据的：
通常较大的 k 是会抑制噪声的影响，但是使得分类界限不明显。

如果数据是不均匀采样的，那么 RadiusNeighborsClassifier 中的基于半径的近邻分类可能是
更好的选择。

用户指定一个固定半径 r，使得稀疏邻居中的点使用较少的最近邻来分类。
对于高维参数空间，这个方法会由于所谓的 “维度灾难” 而变得不那么有效。

基本的最近邻分类使用统一的权重：分配给查询点的值是从最近邻的简单多数投票中计算出来的
在某些环境下，最好对邻居进行加权，使得更近邻更有利于拟合。可以通过 weights 关键字来实
现。

默认值 weights = 'uniform' 为每个近邻分配统一的权重。而 weights = 'distance' 分配
权重与查询点的距离成反比。 或者，用户可以自定义一个距离函数用来计算权重。
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()


'''
 最近邻回归
最近邻回归是用在数据标签为连续变量，而不是离散变量的情况下。分配给查询点的标签是由它
的最近邻标签的均值计算而来的。

scikit-learn 实现了两种不同的最近邻回归：KNeighborsRegressor 基于每个查询点的 k 个
最近邻实现， 其中 k 是用户指定的整数值。RadiusNeighborsRegressor 基于每个查询点的固
定半径 r 内的邻点数量实现， 其中 r 是用户指定的浮点数值。

基本的最近邻回归使用统一的权重：即，本地邻域内的每个邻点对查询点的分类贡献一致。 在某
些环境下，对邻点加权可能是有利的，使得附近点对于回归所作出的贡献多于远处点。 这可以通
过 weights 关键字来实现。默认值 weights = 'uniform' 为所有点分配同等权重。 而 
weights = 'distance' 分配的权重与查询点距离呈反比。 或者，用户可以自定义一个距离函数
用来计算权重。
'''

# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

plt.show()


'''
最近质心分类
该 NearestCentroid 分类器是一个简单的算法, 通过其成员的质心来表示每个类。 实际上, 
这使得它类似于 sklearn.KMeans 算法的标签更新阶段. 它也没有参数选择, 使其成为良好的
基准分类器. 然而，它确实受到非凸类的影响，即当类有显著不同的方差时。所以这个分类器假
设所有维度的方差都是相等的。 对于没有做出这个假设的更复杂的方法, 请参阅线性判别分析 
'''
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))
