"""
Created by Chloe on 2022/1/19
神经网络学习--backpropagation algorithm
可优化--feedforward propagation适合所有层数的神经网络
出错why--J代价值计算为9.022, 吴恩达处为0.28--已解决--不能将第一列和最后一列简单互换，而是应将最后一列提至第一列
注意--np.concatenate数组合并的用法, 第一个变量两个需要合并的数组需要用括号括起, 后面才是变量axis, 否则会报错only integer scalar arrays
                                                                                    can be converted to a scalar index
question--随机初始权重最后的预测结果很低，但是给的参数能够很好的预测
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import random
import math
import scipy.optimize as op
import time


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


def visualRandomImg(hImg, exampleWidth, hSaveFig):
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


def rowToMatrix(hNNParam, hInputLayerSize, hHiddenLayerSize, hNumLabel):
    """
    将一列的参数转换为两个矩阵(只适用于此类仅含一层隐藏层的神经网络)
    :param hNNParam: 所有参数组成的列矩阵
    :param hInputLayerSize: 输入层层数
    :param hHiddenLayerSize: 隐藏层层数
    :param hNumLabel: 输出层层数
    :return: 输入层到隐藏层参数、隐藏层到输出层参数
    """
    hTheta1Size = hHiddenLayerSize * (hInputLayerSize + 1)  # (25, 401)
    hTheta1 = np.matrix(np.reshape(hNNParam[:hTheta1Size], (hHiddenLayerSize, hInputLayerSize + 1)))
    hTheta2 = np.matrix(np.reshape(hNNParam[hTheta1Size:], (hNumLabel, hHiddenLayerSize + 1)))
    return hTheta1, hTheta2


def matrixToRow(hTheta1, hTheta2):
    """
    将两个矩阵合并为一个列矩阵, 即参数部分展开, 仅适用于此类仅含一层隐藏层的神经网络
    :param hTheta1: 输入层到隐藏层的参数
    :param hTheta2: 隐藏层到输出层的参数
    :return: 合并后的列矩阵
    """
    hTheta1Row = hTheta1.reshape((hTheta1.shape[0] * hTheta1.shape[1], 1))
    hTheta2Row = hTheta2.reshape((hTheta2.shape[0] * hTheta2.shape[1], 1))
    hNNParam = np.concatenate((hTheta1Row, hTheta2Row), axis=0)  # axis=0, 纵向合并
    return hNNParam


def convert(hy):
    """
    对y的格式进行一个转换，将每一个y转换为仅包含0或1的向量
    :param hy: 数据集的标签集
    :return hResult: 转换后的标签集
    """
    hm = hy.shape[0]  # 5000
    hn = len(np.unique(hy))  # 10
    hResult = np.zeros((hm, hn))  # (5000, 10)
    for i in range(hm):
        hResult[[i], [hy[i] % 10]] = 1
    # hResult[:, [0, 9]] = hResult[:, [9, 0]]
    return hResult


def sigmoid(hx):
    """
    sigmoid函数
    :param hx:数据集的特征
    :return: sigmoid值求解
    """
    g = 1 / (1 + np.exp(-hx))
    return g


def feedforwardPropagation(hX, hTheta1, hTheta2):
    """
    前向传播算法，只适合拥有一个隐藏层的神经网络
    :param hX: 数据集
    :param hTheta1: 输入层到隐藏层的参数
    :param hTheta2: 隐藏层到输出层的参数
    :return h: 前向传播计算结果
    """
    hm = hX.shape[0]  # 5000
    ha1 = np.insert(hX, 0, values=np.ones(hm), axis=1)  # (5000, 401), 前插一列
    hz2 = ha1.dot(hTheta1.T)  # (5000, 25)
    ha2 = np.insert(sigmoid(hz2), 0, values=np.ones(hm), axis=1)  # (5000, 26), 前插一列
    hz3 = ha2.dot(hTheta2.T)  # (5000, 10)
    h = sigmoid(hz3)  # (5000, 10)

    hEnd = np.array(h[:, 9])
    h = np.array(np.delete(h, obj=9, axis=1))
    h = np.concatenate((hEnd, h), axis=1)
    return ha1, hz2, ha2, hz3, h


def costFunction(hNNParam, hInputLayerSize, hHiddenLayerSize, hNumLabel, hX, hy, hLam=0):
    """
    代价函数值计算
    :param hNNParam: 参数集合，应为一列矩阵，首先是输入层到隐藏层的参数，其次是隐藏层到输出层的参数
    :param hInputLayerSize: 输入层的层数
    :param hHiddenLayerSize: 隐藏层层数
    :param hNumLabel: 标签数目，即输出层的层数
    :param hX: 数据集的特征
    :param hy: 数据集的特征对应的值
    :param hLam: 正则化参数
    :return hJ: 代价值
    """
    hm = hX.shape[0]  # 5000
    hyCon = convert(hy)  # (5000, 10)

    # 计算参数大小
    hTheta1, hTheta2 = rowToMatrix(hNNParam, hInputLayerSize, hHiddenLayerSize, hNumLabel)

    # Feedforward Propagation求出hypothesis
    ha1, hz2, ha2, hz3, hXFeed = feedforwardPropagation(hX, hTheta1, hTheta2)  # (5000, 10)

    # 求解代价值
    hJ = 0
    for i in range(hm):
        first = np.multiply(hyCon[i, :], np.log(hXFeed[i, :]))  # np.multiply为计算内积
        second = np.multiply((1 - hyCon[i, :]), np.log(1 - hXFeed[i, :]))
        hJ += np.sum(first + second)
    hJ = -hJ / hm

    # 加入正则项部分(正则化参数默认为0)
    hReg = hLam / (2 * hm) * (np.sum(np.power(hTheta1[:, 1:], 2)) + np.sum(np.power(hTheta2[:, 1:], 2)))
    hJ += hReg
    return hJ


def sigmoidGradient(hz):
    """
    求解sigmoid函数的梯度
    :param hz: sigmoid函数
    :return: sigmoid函数的梯度
    """
    hg = np.zeros(np.size(hz))  # 保证可以求出任何格式的sigmoid函数的梯度, 无论矩阵或者向量
    hg = np.multiply(sigmoid(hz), (1 - sigmoid(hz)))  # numpy数组的广播性, 对单个元素的操作可以广播至整个数组
    return hg


def randomInitializeWeights(hLayerInput, hLayerOutput):
    """
    随机初始化权重, 包括一个随机数项和扰动项
    :param hLayerInput: 输入层数
    :param hLayerOutput: 输出层数
    :return: 大小为(LOut, LIn+1)的随机初始权重
    """
    hw = np.zeros((hLayerOutput, hLayerInput + 1))
    hwSize = hLayerOutput * (hLayerInput + 1)
    hEpsilonInit = 0.12  # 加入扰动项的随机值为0.12
    hw = np.matrix(np.random.randn(hwSize) * 2 * hEpsilonInit - hEpsilonInit)
    # hw = np.matrix(np.random.randn(hwSize))
    return hw


def backpropagation(hNNParam, hInputLayerSize, hHiddenLayerSize, hNumLabel, hX, hy, hLam=0):
    """
    向后传播, 求出代价值和梯度
    :param hNNParam: 参数集合，应为一列矩阵，首先是输入层到隐藏层的参数，其次是隐藏层到输出层的参数
    :param hInputLayerSize: 输入层的层数
    :param hHiddenLayerSize: 隐藏层层数
    :param hNumLabel: 标签数目，即输出层的层数
    :param hX: 数据集的特征
    :param hy: 数据集的特征对应的值
    :param hLam: 正则化参数
    :return hJ: 代价值、hgrad: 梯度值
    """
    hm = hX.shape[0]  # 5000
    hyCon = convert(hy)  # (5000, 10)

    # 计算参数大小
    hTheta1, hTheta2 = rowToMatrix(hNNParam, hInputLayerSize, hHiddenLayerSize, hNumLabel)

    # Feedforward Propagation求出hypothesis
    ha1, hz2, ha2, hz3, hXFeed = feedforwardPropagation(hX, hTheta1, hTheta2)

    # 求解代价值
    hJ = 0
    for i in range(hm):
        first = np.multiply(hyCon[i, :], np.log(hXFeed[i, :]))  # np.multiply为计算内积
        second = np.multiply((1 - hyCon[i, :]), np.log(1 - hXFeed[i, :]))
        hJ += np.sum(first + second)
    hJ = -hJ / hm

    # 加入正则项部分(正则化参数默认为0)
    hReg = hLam / (2 * hm) * (np.sum(np.power(hTheta1[:, 1:], 2)) + np.sum(np.power(hTheta2[:, 1:], 2)))
    hJ += hReg

    # Back Propagation求解梯度
    hDelta1 = hDelta2 = 0  # 求梯度所用的为delta
    for i in range(m):
        # 取出第i个样本(xi, yi)
        ha1i = np.matrix(ha1[i, :])  # (1, 401)
        hz2i = hz2[i, :]  # (1, 25)
        ha2i = ha2[i, :]  # (1, 26)
        hi = hXFeed[i, :]  # (1, 10), 即a3
        hyi = hyCon[i, :]  # (1, 10)

        hError3 = hi - hyi  # (1, 10), 各层误差记为Error
        hz2i = np.insert(hz2i, 1, values=1, axis=1)  # (1, 26)

        hError2 = np.multiply((hTheta2.T.dot(hError3.T)), sigmoidGradient(hz2i))  # (1, 26)

        hDelta1 += (hError2[:, 1:]).T.dot(ha1i)  # (25, 401)
        hDelta2 += np.matrix(hError3).T.dot(ha2i)  # (10, 26)

    hDelta1 = np.array(hDelta1 / hm)
    hDelta2 = np.array(hDelta2 / hm)
    hDelta1[:, 1:] += (hTheta1[:, 1:] * hLam) / hm
    hDelta2[:, 1:] += (hTheta2[:, 1:] * hLam) / hm

    hgrad = np.concatenate((np.ravel(hDelta1), np.ravel(hDelta2)))  # np.concatenate方法的前一个变量，两个合并数组需要用()括起
    # np.ravel将数组拉为一维数组
    return hJ, hgrad


def predict(hTheta1, hTheta2, hX):
    """
    输入参数，对其进行预测
    :param hTheta1: 输入层到隐藏层的参数
    :param hTheta2: 隐藏层到输出层的参数
    :param hX: 特征矩阵
    :return: hPred: 预测结果，直接为数字
    """
    hm = hX.shape[0]
    _, _, _, _, hy = feedforwardPropagation(hX, hTheta1, hTheta2)
    hPred = np.argmax(np.array(hy), axis=1)
    for i in range(hm):
        if hPred[i] == 0:
            hPred[i] = 10
    hPred = np.reshape(hPred, (hm, 1))
    return hPred


if __name__ == '__main__':
    startTime = time.perf_counter()
    """
    初始化Neural Network参数
    """
    inputLayerSize = 400  # 输入层400个单元, 每张图片像素为20*20
    hiddenLayerSize = 25  # 隐藏层25个单元,
    numLabel = 10  # 输出层包括10个标签

    """
    Part 1: Loading and Visualizing Data
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
    visualRandomImg(randomResult, 20, hSaveFig=saveAddress)
    print("=" * 40)

    """
    Part 2: Loading Parameters feedforward propagation
    """
    print("Loading Saved Neural Network Parameters...")
    filenameWeight = 'ex4weights.mat'
    Weight = scio.loadmat(filenameWeight)
    # print(Weight.keys())
    theta1 = Weight['Theta1']  # (25, 401)
    theta2 = Weight['Theta2']  # (10, 26)
    # 将theta值改为列矩阵. 最优化算法也需要把参数修改为列矩阵
    nnParam = matrixToRow(theta1, theta2)
    print("=" * 40)

    """
    Part 3: Compute Cost (Feedforward)
    """
    print("Feedforward Using Neural Network...")
    lamTest = 0  # 模型测试阶段, 忽略正则化参数
    JTest = costFunction(nnParam, inputLayerSize, hiddenLayerSize, numLabel, X, y)
    print("Cost: ", JTest)
    print("Expected cost: 0.287629")
    print("=" * 40)

    """
    Part 4: Implement Regularization
    """
    print("Checking Cost Function (Regularization)...")
    lamReg = 1
    JRegTest = costFunction(nnParam, inputLayerSize, hiddenLayerSize, numLabel, X, y, lamReg)
    print("CostReg: ", JRegTest)
    print("Expected cost: 0.383770")
    print("=" * 40)

    """
    Part 5: Sigmoid Gradient
    """
    print("Evaluating sigmoid gradient...")
    zTest = np.array([-1, -0.5, 0, 0.5, 1])
    gTest = sigmoidGradient(zTest)
    print("sigmoidGradient: ", gTest)
    print("=" * 40)

    """
    Part 6: Initializing Parameters
    """
    print("Initializing Neural Network Parameters...")
    initWeights1 = randomInitializeWeights(inputLayerSize, hiddenLayerSize)
    initWeights2 = randomInitializeWeights(hiddenLayerSize, numLabel)
    # 转换为列矩阵
    initNNParam = matrixToRow(initWeights1, initWeights2)
    print("=" * 40)

    """
    Part 7: Implement Backpropagation
    """
    print("Checking Backpropagation...")  # 此步暂时忽略
    print("=" * 40)

    """
    Part 8: Implement Regularization
    """
    print("Checking Backpropagation (Regularization)...")  # 此步暂时忽略
    print("=" * 40)

    """
    Part 9: Training NN
    """
    print("Training Neural Network...")
    lambdaBack = 1
    # J, grad = backpropagation(initNNParam, inputLayerSize, hiddenLayerSize, numLabel, X, y, lambdaBack)
    # 使用op.minimize实现梯度下降
    result = op.minimize(fun=backpropagation, x0=initNNParam,
                         args=(inputLayerSize, hiddenLayerSize, numLabel, X, y, lambdaBack), method='TNC', jac=True,
                         options={'maxiter': 300})  # jac=True的情况下, scipy默认将第一项作为cost, 第二项作为gradient
    resultJ = result.fun
    resultNNParam = result.x
    resultTheta1, resultTheta2 = rowToMatrix(resultNNParam, inputLayerSize, hiddenLayerSize, numLabel)

    # 运用算法算出的权重做出预测
    yPred = np.array(predict(resultTheta1, resultTheta2, X))
    Accuracy = np.mean(np.double(y == yPred)) * 100
    print("Test NeuralNetwork Accuracy:", Accuracy)
    print('=' * 40)

    endTime = time.perf_counter()
    print("Finished Time:", endTime - startTime)
