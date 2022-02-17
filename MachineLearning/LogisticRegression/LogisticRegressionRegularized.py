"""
Created by Chloe on 2022/1/14
逻辑回归--正则化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op


def mapFeature(x1, x2, power):
    """
    要素映射
    :param x1:数据集的第一个特征
    :param x2:数据集的第二个特征
    :param power: 最大的阶数
    :return: 要素映射的结果，为方面观察，是一个pd的DataFrame
    """
    out = {}
    for i in range(0, power + 1):
        for j in range(i + 1):
            out["f{}{}".format(i - j, j)] = np.power(x1, i - j) * np.power(x2, j)
    return pd.DataFrame(out)


def sigmoid(hx):
    """
    sigmoid函数
    :param hx:数据集的特征
    :return: sigmoid值求解
    """
    g = 1 / (1 + np.exp(-hx))
    return g


def hypothesis(theta, hx):
    """
    假设函数
    :param theta: 函数的系数矩阵（列向量）
    :param hx: 数据集的特征（行向量）
    :return: 假设函数的值
    """
    h = sigmoid(hx.dot(theta))
    return h


def costFunction(theta, hx, hy, hLam=1):
    """
    代价函数的求解
    :param theta: 假设函数的系数矩阵
    :param hx: 数据集的特征
    :param hy: 数据集的特征对应的值
    :param hLam: 正则化参数
    :return: 代价函数求解值
    """
    hm = hx.shape[0]
    thetaReg = np.delete(theta, 0, axis=0)
    J = -1 / hm * (hy.T.dot(np.log(hypothesis(theta, hx))) + (1 - hy).T.dot(np.log(1 - hypothesis(theta, hx)))) \
        + hLam / (2 * hm) * thetaReg.T.dot(thetaReg)
    return J


def gradient(theta, hx, hy, hLam=1):
    """
    代价函数的梯度求解
    :param theta: 假设函数的系数矩阵
    :param hx: 数据集的特征
    :param hy: 数据集的特征对应的值（0或1）
    :param hLam: 正则化参数
    :return: 代价函数的梯度值
    """
    theta = np.reshape(theta, (hx.shape[1], 1))
    hm = hx.shape[0]
    Reg = hLam / hm * theta  # 正则化项
    Reg[0] = 0  # theta[0]不参与, 因此此项为1
    grad = 1 / hm * hx.T.dot(hypothesis(theta, hx) - hy) + Reg
    return grad


if __name__ == '__main__':
    """
    加载数据
    """
    filename = 'ex2data2.txt'
    df = pd.read_csv(filename, names=['Test1', 'Test2', 'Result'])
    X = np.array(df[['Test1', 'Test2']])
    X = np.insert(X, 0, values=1, axis=1)
    y = np.array(df[['Result']])
    m = X.shape[0]  # m为训练集的数据数量

    """
    可视化数据
    """
    # 数据分类
    positive = [i for i in range(m) if y[i] == 1]  # positive存储通过测试的元件的索引
    negative = [j for j in range(m) if y[j] == 0]  # negative存储未通过测试的元件的索引

    # 绘制分类的图
    type1 = plt.scatter(X[positive, 1], X[positive, 2], c='b', marker='+')
    type2 = plt.scatter(X[negative, 1], X[negative, 2], c='g', marker='x')
    plt.legend((type1, type2), ('Pass', 'No Pass'))
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('Plot of training data')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LogisticRegression\\Figure\\RegularizedScatterTrainingData.png")
    plt.show()

    """
    mapFeature
    """
    # 要素映射，即将两个要素映射到高次项上
    XOut = mapFeature(X[:, 1], X[:, 2], 6)
    XOut = np.array(XOut)  # 转换为np数组

    """
    正则化梯度下降的过程
    """
    # 初始化参数
    n = XOut.shape[1]  # n为特征量+1
    # lam = 1
    thetaInit = np.zeros((n, 1))

    # 代价函数的检验1
    JInit = costFunction(thetaInit, XOut, y)
    gradInit = gradient(thetaInit, XOut, y)
    print("当使用初始的theta值带入代价函数求得J和梯度分别为:")
    print("J:", JInit)
    print("grad:\n", gradInit)

    # 代价函数的检验2
    thetaTest = np.ones((n, 1))
    lamTest = 10
    JTest = costFunction(thetaTest, XOut, y, lamTest)
    gradTest = gradient(thetaTest, XOut, y, lamTest)
    print("\n当使用测试的theta值带入代价函数求得J和梯度分别为:")
    print("J:", JTest)
    print("grad:\n", gradTest)

    # 利用optimize中的minimize实现无约束多变量函数的最小值求解
    result = op.minimize(fun=costFunction, x0=thetaInit, args=(XOut, y), method='TNC', jac=gradient)
    JRes = result.fun  # JRes是更新后的J值
    thetaRes = np.reshape(result.x, (n, 1))  # thetaRes是更新后的theta值
    print("\n利用optimize中的minimize后求得的梯度值和theta分别为:")
    print("更新后的J:", JRes)
    print("更新后的theta:\n", thetaRes)

    """
    绘制DecisionBoundary
    """
    XPlot = np.linspace(np.min(XOut[:, 1]), np.max(XOut[:, 1]), 1000)
    xx, yy = np.meshgrid(XPlot, XPlot)  # np.meshgrid()是二维坐标网
    zz = mapFeature(xx.ravel(), yy.ravel(), 6).values  # ravel()将数组拉为一维数组
    zz = zz.dot(thetaRes)
    zz = np.reshape(zz, xx.shape)
    boundary = plt.contour(xx, yy, zz, 0)
    type1 = plt.scatter(X[positive, 1], X[positive, 2], c='b', marker='+')
    type2 = plt.scatter(X[negative, 1], X[negative, 2], c='g', marker='x')
    plt.legend((type1, type2), ('Pass', 'No Pass'))
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('Decision Boundary of training data')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LogisticRegression\\Figure\\RegularizedDecisionBoundary.png")
    plt.show()

    """
    对更新的theta作质量评估
    """
    # 更新y值, 此y值由更新的theta值求得
    yAccuracy = hypothesis(thetaRes, XOut)
    for i in range(m):
        if yAccuracy[i] >= 0.5:
            yAccuracy[i] = 1
        else:
            yAccuracy[i] = 0

    # 将yAccuracy与训练集提供的y值作比较，求得模型的Accuracy
    Accuracy = np.mean(np.double(y == yAccuracy)) * 100  # np.mean(y_test==y_pred)可以很好的求出accuracy，(==返回False或True数组)
    print("\n经过评估得出本LogisticRegression模型的准确性为:", Accuracy)
