"""
Created by Chloe on 2022/1/8
线性拟合--梯度下降--单变量
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# h是假设函数
def hypothesis(hx, thetaForFunc):
    h = thetaForFunc[0] + thetaForFunc[1] * hx
    return h


# J是代价函数
def computeCost(hx, hy, hm, thetaForFunc):
    J = 1 / (2 * hm) * sum(np.square((hypothesis(hx, thetaForFunc) - hy)))
    return J


# 梯度下降过程
def gradientDescent(hx, hy, hm, halpha, thetaForFunc, iterationsForFunc):
    """
    :param hx: 数据集中的变量
    :param hy: 数据集中的变量对应的值
    :param hm: 数据集中的数据个数
    :param halpha: learning rate
    :param thetaForFunc: 预测函数的系数
    :param iterationsForFunc: 希望的迭代次数
    :return:更新后的系数
    """
    temp = thetaForFunc  # 为达到更新theta的同时性，使用temp数组暂存
    for iteration in range(iterationsForFunc):
        temp[0] = thetaForFunc[0] - halpha * 1 / hm * sum(hypothesis(hx, thetaForFunc) - hy)
        temp[1] = thetaForFunc[1] - halpha * 1 / hm * sum((hypothesis(hx, thetaForFunc) - hy) * hx)
        thetaForFunc[0] = temp[0]
        thetaForFunc[1] = temp[1]
    return thetaForFunc


if __name__ == '__main__':
    """
    数据准备
    """
    # 读取数据
    filename = 'ex1data1.txt'
    df = pd.read_csv(filename, names=['x', 'y'])
    x = df['x']
    y = df['y']
    m = df.shape[0]  # m为训练集数据的数量

    # 绘制数据散点图
    plt.scatter(x, y)
    plt.title('scatter of dataset')
    plt.xlabel('Populations of city in 10000s')
    plt.ylabel('Profit in $10000s')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LinearRegression\\Figure\\Scatter.png")
    plt.show()

    """
    梯度下降过程实现
    """
    theta = np.zeros((2, 1))    # 输入初始的系数
    alpha = 0.01                # learning rate
    iterations = 1500           # 迭代次数

    theta = gradientDescent(df['x'], df['y'], m, alpha, theta, iterations)

    """
    绘制出此预测图
    """
    plt.scatter(x, y)
    x_pre = np.arange(5, 24, 0.01)
    y_pre = hypothesis(x_pre, theta)
    plt.plot(x_pre, y_pre, 'r')
    plt.title('result of predictions')
    plt.xlabel('Populations of city in 10000s')
    plt.ylabel('Profit in $10000s')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LinearRegression\\Figure\\Predictions.png")
    plt.show()

    """
    进行简单的预测
    """
    pre1 = hypothesis(3.5, theta)
    print(pre1 * 10000)
    pre2 = hypothesis(7, theta)
    print(pre2 * 10000)

    """
    可视化J
    """
    # 选定展示的参数范围
    theta0Vals = np.linspace(-10, 10, 100)
    theta1Vals = np.linspace(-1, 4, 100)
    JVals = np.zeros((np.shape(theta0Vals)[0], np.shape(theta1Vals)[0]))    # 初始化J值存储表格

    # 计算每一个对应的J值
    for i in range(np.shape(theta0Vals)[0]):
        for j in range(np.shape(theta1Vals)[0]):
            t = [theta0Vals[i], theta1Vals[j]]
            JVals[i, j] = computeCost(df['x'], df['y'], m, t)

    # 绘制出J的三维图像
    x_surf, y_surf = np.meshgrid(theta0Vals, theta1Vals)    # 设置二维表格
    fig = plt.figure()
    ax = plt.axes(projection='3d')  # 开启三维坐标轴
    ax.plot_surface(x_surf, y_surf, JVals)  # 绘制J的三维图像
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('J')
    plt.title('J three-dimension fig')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LinearRegression\\Figure\\J_threeDimension.png")
    plt.show()

    # 绘制二维等高线图
    plt.contour(x_surf, y_surf, JVals, 30)
    plt.scatter(theta[0], theta[1])
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('contour of J')
    plt.savefig("D:\\PythonFiles\\MachineLearning\\LinearRegression\\Figure\\contour.png")
    plt.show()
