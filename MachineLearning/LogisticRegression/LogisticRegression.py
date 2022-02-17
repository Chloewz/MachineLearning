"""
Created by Chloe on 2022/1/12
逻辑回归--binary--直线
可改进--在函数中使用shape，从而不用手动输入参数m或n
可改进--画图也列入函数之中
可改进--将X统一为（n+1）列的矩阵
attention!--使用optimize中的minimize时，输入和输出的theta都是行向量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op  # 代替matlab中的fminunc的优化方法，是一种求解无约束多变量函数的最小值方法


# sigmoid函数
def sigmoid(hx):
    """
    sigmoid函数
    :param hx:数据集的特征
    :return: sigmoid值求解
    """
    g = 1 / (1 + np.exp(-hx))
    return g


# 假设函数
def hypothesis(hx, thetaForFunc):
    """
    假设函数
    :param hx:数据集的特征（行向量）
    :param thetaForFunc: 函数的系数矩阵（列向量）
    :return: 假设函数的值
    """
    h = sigmoid(np.dot(hx, thetaForFunc))
    # h = sigmoid(np.dot(thetaForFunc, hx.T))
    return h


# 代价函数
def costFunction(thetaForFunc, hx, hy):
    """
    代价函数的求解
    :param thetaForFunc:假设函数的系数矩阵
    :param hx: 数据集的特征
    :param hy: 数据集的特征对应的值
    :return: 代价函数求解值
    """
    hm = hx.shape[0]
    hJ = -1 / hm * (np.dot(hy.T, np.log(hypothesis(hx, thetaForFunc) + 1e-6)) +
                    np.dot((1 - hy).T, np.log(1 - hypothesis(hx, thetaForFunc) + 1e-6)))
    return hJ


# 代价函数的梯度
def gradient(thetaForFunc, hx, hy):
    """
    代价函数的梯度求解
    :param thetaForFunc:假设函数的系数矩阵
    :param hx:数据集的特征
    :param hy:数据集的特征对应的值（0或1）
    :return:代价函数的梯度值
    """
    thetaForFunc = np.reshape(thetaForFunc, (hx.shape[1], 1))  # 需将theta数组维度转为(3,1),否则会报错(optimize输入的是(1,3))
    # 否则就需要在函数外面定义时就把theta定义为(1,3),那样要改全部函数
    hm = hx.shape[0]
    hgrad = 1 / hm * (np.dot(hx.T, hypothesis(hx, thetaForFunc) - hy))
    return hgrad


if __name__ == '__main__':
    """
    数据准备
    """
    # 读取数据
    filename = 'ex2data1.txt'
    df = pd.read_csv(filename, names=['Exam1', 'Exam2', 'Admitted'])
    X = np.array(df[['Exam1', 'Exam2']])
    y = np.array(df[['Admitted']])
    m, n = X.shape  # m为训练集的数量, n为特征的数量

    """
    可视化数据
    """
    # 将数据分类
    positive = [i for i in range(m) if y[i] == 1]  # positive存储被录取的学生的索引
    negative = [j for j in range(m) if y[j] == 0]  # negative存储未被录取的学生的索引

    # 绘制分类的图
    type1 = plt.scatter(X[positive, 0], X[positive, 1], c='b', marker='+')
    type2 = plt.scatter(X[negative, 0], X[negative, 1], c='g', marker='x')
    plt.legend((type1, type2), ('Admitted', 'No Admitted'))
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Plot of training data')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LogisticRegression\\Figure\\ScatterTrainingData.png")
    plt.show()

    """
    利用梯度下降进行逻辑回归
    """
    X = np.insert(X, 0, values=1, axis=1)  # 在前面加一列
    thetaInit = np.zeros((n + 1, 1))

    # 代价函数的检验
    JInit = costFunction(thetaInit, X, y)
    gradInit = gradient(thetaInit, X, y)
    print("当使用初始的theta值带入代价函数求得J和梯度分别为:")
    print("J:", JInit)
    print("grad:\n", gradInit)

    # 利用optimize中的minimize实现无约束多变量函数的最小值求解
    result = op.minimize(fun=costFunction, x0=thetaInit, args=(X, y), method='TNC', jac=gradient)  # result存储了优化结果
    JRes = result.fun  # JRes中是优化后的代价函数值
    thetaRes = np.reshape(result.x, (n + 1, 1))  # thetaRes中是优化后的系数值
    print("\n利用optimize中的minimize后求得的梯度值和theta分别为:")
    print("更新后的J:", JRes)
    print("更新后的theta:\n", thetaRes)

    """
    绘制plotDecisionBoundary
    """
    XPlot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 1000)
    YPlot = -(thetaRes[0] + thetaRes[1] * XPlot) / thetaRes[2]
    plt.figure()
    boundary = plt.plot(XPlot, YPlot, '-')
    type1 = plt.scatter(X[positive, 1], X[positive, 2], c='b', marker='+')
    type2 = plt.scatter(X[negative, 1], X[negative, 2], c='g', marker='x')
    plt.legend((type1, type2), ('Admitted', 'No Admitted'))
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Plot of Decision Boundary')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LogisticRegression\\Figure\\PlotDecisionBoundary.png")
    plt.show()

    """
    利用更新的theta进行预测
    """
    XPred = [1, 45, 85]  # 预测的考生exam1成绩为45, exam2成绩为85
    yPred = hypothesis(XPred, thetaRes)
    print("\n当考生的exam1和exam2的成绩分别为%d, %d时, 他被录取的概率为: %.4f" % (XPred[1], XPred[2], yPred))

    """
    对更新的theta作质量评估
    """
    # 更新y值, 此y值由更新的theta值求得
    yAccuracy = hypothesis(X, thetaRes)
    for i in range(m):
        if yAccuracy[i] >= 0.5:
            yAccuracy[i] = 1
        else:
            yAccuracy[i] = 0

    # 将yAccuracy与训练集提供的y值作比较，求得模型的Accuracy
    Accuracy = np.mean(np.double(y == yAccuracy)) * 100  # np.mean(y_test==y_pred)可以很好的求出accuracy，(==返回False或True数组)
    print("\n经过评估得出本LogisticRegression模型的准确性为:", Accuracy)
