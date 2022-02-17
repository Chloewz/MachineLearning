"""
Created by Chloe on 2022/1/9
线性拟合--梯度下降--多变量
可改进--将迭代过程存放至costFunction函数外（单次返回单次求得的代价函数值和偏导值）
可改进--涉及需要输入样本数量的函数，这个参数可以取消，在函数内部用一个shape求取就行了
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 假设函数
def hypothesis(hx, thetaForFunc):
    """
    假设函数
    :param hx:数据集的特征 （行向量）
    :param thetaForFunc:函数的系数矩阵（列向量）
    :return: 假设值
    """
    h = np.dot(hx, thetaForFunc)  # h=theta.*x
    return h


# 代价函数
def computeCostMulti(hx, hy, hm, thetaForFunc):
    """
    线性拟合中的代价函数
    :param hx: 数据集的特征
    :param hy: 数据集的特征对应的值
    :param hm: 数据集的数据个数
    :param thetaForFunc: 假设函数的系数列矩阵
    :return: 计算机J值
    """
    J = 1 / (2 * hm) * np.dot(np.transpose(hypothesis(hx, thetaForFunc) - hy), (hypothesis(hx, thetaForFunc) - hy))
    return J


# 梯度下降
def gradientDescentMulti(hx, hy, hm, haplha, thetaForFunc, iterationsForFunc):
    """
    梯度下降过程的实现
    :param hx: 数据集的特征
    :param hy: 数据集的特征对应的值
    :param hm: 数据集的数据个数
    :param haplha: learning rate
    :param thetaForFunc: 初始的系数
    :param iterationsForFunc: 设置的迭代次数
    :return: 更新后的系数
    """
    JHistoryForFunc = np.zeros((iterationsForFunc, 1))  # 存储J计算值的数据
    for iteration in range(iterationsForFunc):
        S = 1 / hm * np.dot(np.transpose(hx), hypothesis(hx, thetaForFunc) - hy)
        thetaForFunc = thetaForFunc - haplha * S
        JHistoryForFunc[[iteration]] = computeCostMulti(hx, hy, hm, thetaForFunc)
    return thetaForFunc, JHistoryForFunc


# 最小二乘求解
def normalEquations(hx, hy):
    """
    最小二乘法求解系数
    :param hx: 数据集的特征
    :param hy: 数据集的特征对应的值
    :return: 最小二乘法计算出的系数
    """
    thetaNorm = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(hx), hx)), np.transpose(hx)), hy)
    return thetaNorm


# 特征缩放
def featureScaling(hx):
    """
    特征缩放函数
    :param hx: 数据集的要素
    :return: 特征缩放的结果
    """
    hmean = np.mean(hx)  # hmean是hx的平均值
    hsigma = np.std(feature)  # hsigma是hx的标准差
    result = np.array((hx - hmean) / hsigma)
    return result


if __name__ == '__main__':
    """
    数据准备
    """
    # 读取数据
    filename = 'ex1data2.txt'
    df = pd.read_csv(filename, names=['size', 'number', 'prices'])  # size为房间尺寸，number为卧室数量

    """
    特征标准化
    """
    feature = df[['size', 'number']]
    m, n = feature.shape  # m为训练集数据数量, n为特征的数量
    mean = np.mean(feature)
    sigma = np.std(feature)
    featureNorm = featureScaling(feature)

    """
    梯度下降过程实现
    """
    theta = np.zeros((n + 1, 1))  # 输入初始的系数
    alpha = 0.01  # learning rate
    iterations = 400  # 迭代次数
    feature = np.insert(np.array(feature), 0, values=1, axis=1)  # 在前面加一列1
    featureNorm = np.insert(featureNorm, 0, values=1, axis=1)
    y = df['prices'].values.reshape((m, 1))

    theta, JHistory = gradientDescentMulti(featureNorm, y, m, alpha, theta, iterations)
    print("梯度下降法得出系数:\n", theta, "\n")

    """
    利用更新的theta做预测
    """
    predX = [1650, 3]  # 代表1650的size和3间卧室
    predX = (predX - mean) / sigma  # 标准化该数据
    predX = np.insert(np.array(predX), 0, 1)
    predictionGrad = hypothesis(predX, theta)
    print("梯度下降法预测结果: ", predictionGrad, "\n")

    """
    可视化J
    """
    plt.plot(range(iterations), JHistory)
    plt.xlabel('iterations')
    plt.ylabel('value of J')
    plt.title('the trajectory of J')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LinearRegression\\Figure\\JTrajectory.png")
    plt.show()

    """
    最小二乘求解
    """
    thetaNormE = normalEquations(feature, y)
    print("最小二乘法得出的系数:\n", thetaNormE, "\n")

    """
    最小二乘预测
    """
    predictionNorm = hypothesis([1, 1650, 3], thetaNormE)
    print("最小二乘法预测的结果: ", predictionNorm, "\n")
