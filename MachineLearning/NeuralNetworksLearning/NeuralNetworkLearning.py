"""
Created by Chloe on 2022/1/19
神经网络学习--backpropagation algorithm
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
        hIndex = random.randint(0, hm)  # 产生随机数
        resImg[i] = hImg[hIndex]
    return resImg


def VisualRandomImg(hImg, exampleWidth, hSaveFig):
    """
    可视化随机的图像
    :param hSaveFig: 保存生成图片的地址
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
            hRowImg = np.reshape(hImg[i * displayRows + j], (exampleHeight, exampleWidth)).T  # 直接生成的图片是倒置的, 需要转置
            plt.subplot(displayRows, displayCols, i * displayRows + j + 1)  # subplot的序列不能从0开始
            plt.imshow(hRowImg, cmap='gray')
            plt.axis('off')
    plt.savefig(hSaveFig, bbox_inches='tight')
    plt.show()
    return 0


if __name__ == '__main__':
    """
    初始化Neural Network参数
    """
    inputLayerSize = 400  # 输入层400个单元, 每张图片像素为20*20
    hiddenLayerSize = 25  # 隐藏层25个单元,
    numLabel = 10  # 输出层包括10个标签

    """
    加载和可视化数据
    """
    print("Loading and Visualizing Data...")
    # 加载训练集数据
    filenameData = 'ex4data1.mat'
    data = scio.loadmat(filenameData)
    # print(data.keys())
    X = data['X']  # (5000, 400)
    y = data['y']  # (5000, 1)
    m, n = X.shape  # m=5000, n=400

    # 随机选取100个数据图像展示
    saveAddress = "D:\\PythonFiles\\MachineLearning\\NeuralNetworksLearning\\Figure\\VisualizeRandomPicture.png"
    randomResult = randomlySelected(X, 100)
    VisualRandomImg(randomResult, 20, hSaveFig=saveAddress)
    print("=" * 40)

    """
    加载参数feedforward propagation
    """
    print("Loading Saved Neural Network Parameters...")
    filenameWeight = 'ex4weights.mat'
    Weight = scio.loadmat(filenameWeight)
    # print(Weight.keys())
    theta1 = Weight['Theta1']  # (25, 401)
    theta2 = Weight['Theta2']  # (10, 26)
    print("=" * 40)


