"""
Created by Chloe on 2022/3/17
svm支持向量机练习
借鉴--绘制边界时参考网上直接使用了等值面的方法, 更加通用(不仅是线性, 对非线性也适用)
attention--sklearn里的svm相关的model建立后, 其输入的参数矩阵必须为二维的, 如果不是可以用reshape(-1,1)将一维的修改为二维的
attention--使用fit或者其他svm模型算法的时候, 如果输入的y是一维向量, 会产生一个友好的提示, 建议我们将y的形状更改为(n_samples, ),
            这个时候修改一下就好了, 只需要使用y.ravel()
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import svm
import time


def plotData(hX, hy, hPlotTitle, hPlotAdd):
    """
    plots the data points with + for the positive examples and o for the negative examples
    :param hPlotTitle: 绘制散点图的标题
    :param hPlotAdd: 绘制散点图的保存地址
    :param hX: 数据集的特征
    :param hy: 数据集的标签, 0或1
    :return: 数据集的分布
    """
    hm = hX.shape[0]  # 数据数量
    # hPos和hNeg存储符合要求的数据索引
    hPos = [i for i in range(hm) if hy[i] == 1]
    hNeg = [j for j in range(hm) if hy[j] == 0]

    # 绘图
    hType1 = plt.scatter(hX[hPos, 0], hX[hPos, 1], c='b', marker='+')
    hType2 = plt.scatter(hX[hNeg, 0], hX[hNeg, 1], c='r', marker='x')
    plt.legend((hType1, hType2), ('Pos', 'Neg'))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(hPlotTitle)
    plt.savefig(hPlotAdd)
    plt.show()

    return 0


def visualizeBoundary(hX, hy, hModel, hFitTitle, hFitAdd):
    """
    plots a linear decision boundary learned by the SVM
    :param hy: 数据集的标签
    :param hFitAdd: 拟合边界图像的储存地址
    :param hFitTitle: 拟合边界图像的标题
    :param hX: 数据集的特征
    :param hModel: SVM拟合的结果模型
    :return: 选择边界的图像
    """
    hXMax, hXMin = np.max(hX[:, 0]), np.min(hX[:, 0])  # 横坐标的最大值与最小值
    hYMax, hYMin = np.max(hX[:, 1]), np.min(hX[:, 1])  # 纵坐标的最大值与最小值
    hxx, hyy = np.meshgrid(np.linspace(hXMin, hXMax, 1000), np.linspace(hYMin, hYMax, 1000))  # 直接绘制为坐标网
    hzz = hModel.predict(np.concatenate((hxx.ravel().reshape(-1, 1), hyy.ravel().reshape(-1, 1)), axis=1))
    # hzz = hModel.predict(np.concatenate((hxx.ravel().reshape(-1, 1), hyy.ravel().reshape(-1, 1)), axis=1))

    # 绘制拟合的图像
    plt.contour(hxx, hyy, hzz.reshape(hxx.shape))
    plotData(hX, hy, hFitTitle, hFitAdd)
    return 0


def GaussianKernel(hx1, hx2, hSigma):
    """
    returns a radial basis function kernel between x1 and x2
    :param hx1: 第一个数据
    :param hx2: 第二个数据
    :param hSigma: 标准差
    :return: 高斯核的值
    """
    hKernel = np.exp(-np.sum(np.power(hx1 - hx2, 2)) / (2 * hSigma ** 2))
    return hKernel


def dataset3Params(hX, hy, hXVal, hyVal, hC, hSigma):
    """
    returns your choice of C and sigma for Part 3 of the exercise where you select the optimal (C, sigma) learning
    parameters to use SVM with RBF kernel
    :param hX: 数据训练集的特征
    :param hy: 数据训练集的标签
    :param hXVal: 数据验证集的特征
    :param hyVal: 数据验证集的标签
    :param hC: SVM的参数C集合
    :param hSigma: SVM的参数sigma集合
    :return: hMinC, hMinSigma: 选择的得分最高的SVM参数C和sigma
    """
    hMinError = -1
    hMinC = 9999
    hMinSigma = 9999

    # 对不同的C和sigma组合逐一比较
    for hi in range(len(hC)):
        for hj in range(len(hSigma)):
            hCurC = hC[hi]  # 暂存
            hCurSigma = hSigma[hj]  # 暂存
            hCurGamma = 1 / (2 * hCurSigma ** 2)
            hModel = svm.SVC(C=hCurC, kernel='rbf', gamma=hCurGamma)
            hModel.fit(hX, hy.ravel())
            hScore = hModel.score(hXVal, hyVal.ravel())

            # 对每次的得分进行比较, 选择得分最高的一次
            if hScore > hMinError:
                hMinError = hScore
                hMinC = hCurC
                hMinSigma = hCurSigma
    return hMinC, hMinSigma


if __name__ == '__main__':
    startTime = time.perf_counter()
    """
    Part 1: Loading and Visualizing Data
    """
    print("Loading and Visualizing Data...")
    # 加载数据
    filenameTrain = 'ex6data1.mat'
    dataTrain = scio.loadmat(filenameTrain)
    # print(data)
    XTrain = dataTrain['X']
    yTrain = dataTrain['y']

    # 绘制训练集数据
    PlotTrainTitle = "Example Dataset 1"
    PlotTrainAddress = "D:\\PythonFiles\\MachineLearning\\SupportVectorMachine\\Figure\\TrainDataScatter.png"
    plotData(XTrain, yTrain, PlotTrainTitle, PlotTrainAddress)
    print('=' * 40)

    """
    Part 2: Training Linear SVM
    """
    # 加载数据
    filenameLinear = 'ex6data1.mat'
    dataLinear = scio.loadmat(filenameLinear)
    XLinear = dataLinear['X']
    yLinear = dataLinear['y']

    # 训练svm模型
    print("Training Linear SVM...")
    modelLinear = svm.SVC(C=1, kernel='linear')
    modelLinear.fit(XLinear, yLinear.ravel())

    # 绘制svm模型的训练边界
    PlotLinearTitle = "Figure 2: SVM Decision Boundary with C=1 (Example Dataset 1)"
    PlotLinearAddress = "D:\\PythonFiles\\MachineLearning\\SupportVectorMachine\\Figure\\LinearBoundary.png"
    visualizeBoundary(XLinear, yLinear, modelLinear, PlotLinearTitle, PlotLinearAddress)
    print("Plot the Linear Decision Boundary already done")
    print('=' * 40)

    """
    Part 3: Implementing Gaussian Kernel
    """
    print("Evaluating the Gaussian Kernel...")
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    testGaussianKernel = GaussianKernel(x1, x2, sigma)
    print("Expected Test Gaussian Kernel Value: 0.324652")
    print("Computed Test Gaussian Kernel Value: ", testGaussianKernel)
    print('=' * 40)

    """
    Part 4: Visualizing Dataset 2
    """
    print("Loading and Visualizing Data2...")
    # 加载数据
    filenameGaussian = 'ex6data2.mat'
    dataGaussian = scio.loadmat(filenameGaussian)
    # print(dataGaussian)
    XGaussian = dataGaussian['X']
    yGaussian = dataGaussian['y']

    # 绘制数据集数据点
    plotGaussianTitle = 'Figure 4: Example Dataset 2'
    plotGaussianAddress = "D:\\PythonFiles\\MachineLearning\\SupportVectorMachine\\Figure\\GaussianDataScatter.png"
    plotData(XGaussian, yGaussian, plotGaussianTitle, plotGaussianAddress)
    print("=" * 40)

    """
    Part 5: Training SVM with RBF Kernel (Dataset 2)
    """
    print("Training SVM with RBF kernel...")
    # 设置SVM参数
    CGaussian = 1
    sigmaGaussian = 0.1
    gammaGaussian = 1 / (2 * sigmaGaussian ** 2)
    modelGaussian = svm.SVC(C=CGaussian, kernel='rbf', gamma=gammaGaussian)
    modelGaussian.fit(XGaussian, yGaussian.ravel())

    # 绘制SVM的训练选择边界
    plotGaussianBoundaryTitle = 'Figure 5: SVM (Gaussian Kernel) Decision Boundary (Example Dataset 2)'
    plotGaussianBoundaryAddress = "D:\\PythonFiles\\MachineLearning\\SupportVectorMachine\\Figure\\GaussianBoundary.png"
    visualizeBoundary(XGaussian, yGaussian, modelGaussian, plotGaussianBoundaryTitle, plotGaussianBoundaryAddress)
    print("Plot the Gaussian Decision Boundary already done")
    print('=' * 40)

    """
    Part 6: Visualizing Dataset 3
    """
    print("Loading and Visualizing Dataset3...")
    # 加载数据
    filenameTest = 'ex6data3.mat'
    dataTest = scio.loadmat(filenameTest)
    # print(dataTest)
    XTest = dataTest['X']
    yTest = dataTest['y']
    XValTest = dataTest['Xval']
    yValTest = dataTest['yval']

    # 绘制训练集的图像
    plotTestTitle = 'Figure 6: Example Dataset 3'
    plotTestAddress = 'D:\\PythonFiles\\MachineLearning\\SupportVectorMachine\\Figure\\TestScatterPlot.png'
    # plotData(XTest, yTest, plotTestTitle, plotTestAddress)
    print('=' * 40)

    """
    Part 7: Training SVM with RBF Kernel (Dataset 3)
    """
    # 尝试不同的SVM参数, 包括C和sigma
    print("Selecting the suitable parameters...")
    CTry = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigmaTry = CTry
    CEnd, sigmaEnd = dataset3Params(XTest, yTest, XValTest, yValTest, CTry, sigmaTry)
    print("Selected the Most Suitable Parameters: ")
    print("C: ", CEnd)
    print("sigma: ", sigmaEnd)

    # 使用选择出的C和sigma训练SVM模型
    print("Training SVM with RBF kernel through selected parameters...")
    gammaEnd = 1 / (2 * sigmaEnd ** 2)
    modelTest = svm.SVC(C=CEnd, kernel='rbf', gamma=gammaEnd)
    modelTest.fit(XTest, yTest.ravel())
    scoreTest = modelTest.score(XValTest, yValTest.ravel())
    print("Training End! The Model Score: ", scoreTest * 100)

    # 画出SVM模型的DecisionBoundary
    plotTestBoundaryTitle = 'Figure 7: SVM (Gaussian Kernel) Decision Boundary (Example Dataset 3)'
    plotTestBoundaryAddress = 'D:\\PythonFiles\\MachineLearning\\SupportVectorMachine\\Figure\\TestDecisionBoundary.png'
    visualizeBoundary(XTest, yTest, modelTest, plotTestBoundaryTitle, plotTestBoundaryAddress)
    print("Alright, Plot the Decision Boundary already done")
    print('=' * 40)

    endTime = time.perf_counter()
    print("Using time: ", endTime - startTime)
