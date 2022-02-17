"""
Created by Chloe on 2022/1/17
神经网络--手写识别--Feedforward Propagation Algorithm
可改进--自己写的rowToImg其实就是np.reshape()--还需要改其他函数, 此py不做修改, 后面再改// 后续已改进--NeuralNetworkLearning.py
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import random
import math


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
            # hRowImg = np.reshape(hImg[i * displayRows + j], exampleHeight, exampleWidth)
            plt.subplot(displayRows, displayCols, i * displayRows + j + 1)  # subplot的序列不能从0开始
            plt.imshow(hRowImg, cmap='gray')
            plt.axis('off')
    plt.savefig(
        "D:\\PythonFiles\\MachineLearning\\Multi-classClassificationAndNeuralNetworks\\Figure\\VisualizeDataNeural.png",
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


def predict(hTheta1, hTheta2, hx):
    """
    三层的神经网络预测
    :param hTheta1:inputLayer到hiddenLayer的参数
    :param hTheta2:hiddenLayer到outputLayer的参数
    :param hx: inputLayer数据
    :return: outputLayer结果, 但此函数经过了转换
    """
    hm = hx.shape[0]  # 数据集的数据数量
    # hNumLabels = hTheta2.shape[0]  # 标签数量
    hXPred = np.insert(hx, 0, values=1, axis=1)

    # 计算hiddenLayer
    ha2 = hypothesis(hTheta1.T, hXPred)
    ha2 = np.insert(ha2, 0, values=1, axis=1)  # 插入一列, 即bias unit

    # 计算outputLayer
    ha3 = hypothesis(hTheta2.T, ha2)

    # 输出结果矩阵
    hPred = np.argmax(ha3, axis=1) + 1
    for i in range(hm):
        if hPred[i] == 0:
            hPred[i] = 10
    hPred = np.reshape(hPred, (hm, 1))  # 上述计算完成后是一个行向量, 转换为一个列向量
    return hPred


if __name__ == '__main__':
    """
    输入神经网络初始化参数
    """
    inputLayerSize = 400  # 输入层400个单元, 即一张图像有400个像素组成
    hiddenLayerSize = 25  # 隐藏层25个单元
    numLabels = 10  # 输出层包括10个标签

    """
    加载并可视化数据
    """
    print("Loading and Visualizing Data ...")
    filenameData = 'ex3data1.mat'
    data = scio.loadmat(filenameData)
    # print(data.keys())  # data.keys()查看该字典的键, 方便后续得到字典的value
    X = data['X']  # 字典方式得到数据, X为ndarray, (5000, 400), 5000张图片, 每行存储一张图片
    y = data['y']  # y为ndarray, (5000, 1), 每行存储图片对应的标签, 其中Label为10代表图片的数字是0
    m, n = X.shape  # m存储训练集的数据数量, n存储训练集的特征数量

    # 从图片集中随机选取100张图片可视化
    randomImg = randomlySelected(X, 100)
    VisualRandomImg(randomImg, 20)
    print('=' * 40)

    """
    加载参数
    """
    print("Loading Saved Neural Network Parameters ...")
    filenameWeight = 'ex3weights.mat'
    para = scio.loadmat(filenameWeight)
    # print(para.keys())
    theta1 = para['Theta1']  # (25, 401)
    theta2 = para['Theta2']  # (10, 26)
    print('=' * 40)

    """
    模型预测
    """
    yPred = predict(theta1, theta2, X)
    Accuracy = np.mean(np.double(y == yPred)) * 100
    print("Test NeuralNetwork Accuracy:", Accuracy)
    print('=' * 40)

    """
    展示模型预测效果
    """
    print("Displaying Random Selected Picture And Prediction")
    # 以10次随机抽取为例
    for i in range(10):
        index = random.randint(0, m)
        ix = np.reshape(X[index], (1, n))
        iPred = predict(theta1, theta2, ix)
        print("Neural Network Predict: ", iPred % 10)
        print("The Label is: ", y[index] % 10)
        iImg = ix.reshape(20, 20).T
        plt.figure()
        plt.imshow(iImg, cmap='gray')
        plt.axis('off')
        plt.show()

        s = input('输入q退出, 回车则继续')
        if s == 'q':
            break
    print('=' * 40)
