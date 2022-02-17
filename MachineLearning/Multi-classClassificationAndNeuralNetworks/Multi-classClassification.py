"""
Created by Chloe on 2022/1/16
多分类(一对多分类器)--手写识别
可完善--选择随机图像时如果遇到不是正方形图像，计算像素数错误导致的无法展示的报错提示
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import random
import math
import scipy.io as scio  # scio中的loadmat函数可以读取.mat格式的文件
import scipy.optimize as op


def randomlySelected(hImg, hNum):
    """
    从图片集hImg中随机选取hNum个图片
    :param hImg: 图片集
    :param hNum: 选取数
    :return: 选取结果的矩阵, 行为选取数, 列为像素数
    """
    hm = hImg.shape[0]  # 图片总数
    hn = hImg.shape[1]  # 每张图片的像素数
    resImg = np.zeros((hNum, hn))
    for i in range(hNum):
        index = random.randint(0, hm)  # 产生随机数
        resImg[i] = hImg[index]
    return resImg


def rowToImg(hRow, hm, hn):
    """
    将数据集中的一行数据转换成hm*hn的灰度图片矩阵
    :param hRow: 输入的本行数据
    :param hm: 输出的图片每列像素数
    :param hn: 输出的图片每行像素数
    :return: 灰度图片
    """
    resImg = np.zeros((hm, hn))
    for i in range(hm):
        for j in range(hn):
            resImg[i, j] = hRow[i * hn + j]
    return resImg


def VisualRandomImg(hImg, exampleWidth):
    """
    可视化随机的图像
    :param hImg:可视化的图像
    :param exampleWidth:每个小图片的每行像素数
    :return:0, 运行此函数即展示图像
    """
    hm = hImg.shape[0]  # 图片总数
    hn = hImg.shape[1]  # 每张图片的像素数
    # exampleWidth = int(np.sqrt(hn))  # 一张图片的每行像素数
    exampleHeight = int(hn / exampleWidth)  # 一张图片的每列像素数
    displayRows = math.floor(np.sqrt(hm))  # 每行展示图片数
    displayCols = math.ceil(hm / displayRows)  # 每列展示图片数
    for i in range(displayRows):
        for j in range(displayCols):
            hRowImg = rowToImg(hImg[i * displayRows + j], exampleHeight, exampleWidth).T  # 直接生成的图片是倒置的, 需要转置
            plt.subplot(displayRows, displayCols, i * displayRows + j + 1)  # subplot的序列不能从0开始
            plt.imshow(hRowImg, cmap='gray')
            plt.axis('off')
    plt.savefig(
        "D:\\PythonFiles\\MachineLearning\\Multi-classClassificationAndNeuralNetworks\\Figure\\VisualizeData.png",
        bbox_inches='tight')
    plt.show()
    return 0


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


def costFunction(theta, hx, hy, hLam=0.1):
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


def gradient(theta, hx, hy, hLam=0.1):
    """
    代价函数的梯度求解
    :param theta: 假设函数的系数矩阵
    :param hx: 数据集的特征
    :param hy: 数据集的特征对应的值（0或1）
    :param hLam: 正则化参数
    :return: 代价函数的梯度值
    """
    # theta = np.reshape(theta, (theta.shape[0], 1))
    theta = np.reshape(theta, (1, theta.shape[0]))  # (1, 401)
    hy = np.reshape(hy, (hx.shape[0], 1))  # 发现此处在运算时hy转为了(5000, 5000), 故添加此步
    hm = hx.shape[0]
    Reg = hLam / hm * theta.T  # 正则化项
    Reg[0] = 0  # theta[0]不参与, 因此此项为1
    grad = 1 / hm * hx.T.dot(hypothesis(theta.T, hx) - hy) + Reg  # (401, 1)
    return grad


def convert(hy):
    """
    一对多的分类器中，采用一个向量表示y，其中y的索引值为1对应的就是该Label
    :param hy:输入的y
    :return:转换后的y
    """
    hm = hy.shape[0]
    hn = len(np.unique(hy))
    hResult = np.zeros((hm, hn))
    for i in range(hm):
        hResult[[i], [hy[i] % 10]] = 1
    return hResult


def oneVsAll(hx, hy, hNumLabel, hLam):
    """
    一对多分类器的构造
    :param hx:数据的特征
    :param hy:数据的标签
    :param hNumLabel:标签的数量
    :param hLam:正则化参数值
    :return:一对多分类器的参数
    """
    hm, hn = hx.shape  # hm为训练集的数量, hn为特征的数量
    hAllTheta = np.zeros((hNumLabel, hn + 1))  # (10, 401)
    hXMulti = np.insert(hx, 0, values=1, axis=1)  # (5000, 401)
    hyMulti = convert(hy)  # (5000, 10)
    hThetaInit = np.zeros((hn + 1, 1))  # (401, 1)
    for i in range(hNumLabel):
        print("learning class:", i)
        result = op.minimize(fun=costFunction, x0=hThetaInit, args=(hXMulti, hyMulti[:, i], hLam), method='TNC',
                             jac=gradient)
        hThetaLabel = np.reshape(result.x, (1, hn + 1))
        hAllTheta[i, :] = hThetaLabel
    return hAllTheta


def predictOneVsAll(hTheta, hx):
    """
    一对多分类器的预测
    :param hTheta:模型的参数
    :param hx: 想要预测的数据集的特征
    :return: 预测的结果
    """
    hm = hx.shape[0]  # 训练集的数量
    hXPred = np.insert(hx, 0, values=1, axis=1)
    h = np.argmax(hypothesis(hTheta.T, hXPred), axis=1)     # np.argmax返回指定行/列最大值的索引, axis=1代表按行查找, axis=0列
    for i in range(hm):
        if h[i] == 0:
            h[i] = 10
    hyAccuracy = np.reshape(h, (hm, 1))     # 上述计算完成后是一个行向量, 转换为一个列向量
    return hyAccuracy


if __name__ == '__main__':
    startTime = time.perf_counter()
    """
    加载数据
    """
    print("Loading and Visualizing Data...")
    filename = 'ex3data1.mat'
    data = scio.loadmat(filename)
    # print(data.keys())  # data.keys()查看该字典的键, 方便后续得到字典的value
    X = data['X']  # 字典方式得到数据, X为ndarray, (5000, 400), 5000张图片, 每行存储一张图片
    y = data['y']  # y为ndarray, (5000, 1), 每行存储图片对应的标签, 其中Label为10代表图片的数字是0

    """
    设置初始化参数
    """
    inputLayerSize = 400  # 输入照片格式为20*20, 每张照片400像素
    numLabel = 10  # 共10个标签, 从1-10(10代表0)
    m = np.size(X, 0)  # m存储训练集的数量
    n = np.size(X, 1)  # n存储训练集的特征数量

    """
    可视化图片
    """
    # 从图片集中随机选取100张图片可视化
    randomImg = randomlySelected(X, 100)
    VisualRandomImg(randomImg, 20)
    print("=" * 40)

    """
    逻辑回归梯度下降--使用正则化逻辑回归
    """
    # 模型测试
    print("Testing costFunction() and gradient() with regularization")
    thetaTest = np.array([[-2], [-1], [1], [2]])
    XTest = np.array([[1, 0.1, 0.6, 1.1],
                      [1, 0.2, 0.7, 1.2],
                      [1, 0.3, 0.8, 1.3],
                      [1, 0.4, 0.9, 1.4],
                      [1, 0.5, 1, 1.5]])
    YTest = np.array(([1], [0], [1], [0], [1]))
    lamTest = 3
    JTest = costFunction(thetaTest, XTest, YTest, lamTest)
    gradTest = gradient(thetaTest, XTest, YTest, lamTest)
    print('Cost: ', JTest)
    print('Expected cost: 2.534819')
    print('Gradient:\n', gradTest)
    print('Expected gradients:\n[0.146561 -0.548558 0.724722 1.398003]')
    print('=' * 40)

    # One-vs-All(一对多分类器)
    print("Training One-vs-All Logistic Regression...")
    lamInit = 0.1
    modelTheta = oneVsAll(X, y, numLabel, lamInit)
    print('=' * 40)

    """
    利用模型进行预测
    """
    yAccuracy = predictOneVsAll(modelTheta, X)
    Accuracy = np.mean(np.double(y == yAccuracy)) * 100
    print("Test Multi-classClassification Accuracy:", Accuracy)
    print('=' * 40)

    endTime = time.perf_counter()
    print("Finished Time:", endTime - startTime)
