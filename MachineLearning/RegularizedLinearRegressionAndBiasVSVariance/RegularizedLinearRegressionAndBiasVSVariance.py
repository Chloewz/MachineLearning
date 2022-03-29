"""
Created by Chloe on 2022/2/27
对线性规划正则化理解和偏差方差的理解
attention--np.mean()和np.std是参数的使用, 尤其是axis和ddof
可改进--多项式拟合画图是否有更加简便的方法?
可改进--输出的列表更加美观
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.optimize as op
import time


def hypothesis(hTheta, hX):
    """
    线性拟合的假设函数
    :param hX: 训练数据的特征集
    :param hTheta: 特征集的系数
    :return: hy: 假设函数的值, 即系数乘x
    """
    hy = hX.dot(hTheta)
    return hy


def linearRegCostFunction(hTheta, hX, hy, hLam):
    """
    线性规划的代价函数和梯度值求解
    :param hX: 训练数据的特征集
    :param hy: 训练数据特征集对应的值
    :param hTheta: 系数
    :param hLam: 正则化参数
    :return: hJ: 代价函数的值, 也就是后面的误差; hgrad: 对应代价函数的梯度
    """
    hm = hX.shape[0]
    hTheta = np.reshape(hTheta, (hX.shape[1], 1))
    hJ = 1 / (2 * hm) * (hypothesis(hTheta, hX) - hy).T.dot(hypothesis(hTheta, hX) - hy) + hLam / (2 * hm) * (
            hTheta.T.dot(hTheta) - hTheta[0] ** 2)

    hgrad = 1 / hm * hX.T.dot(hypothesis(hTheta, hX) - hy) + hLam / hm * hTheta
    hgrad[0] = hgrad[0] - hLam / hm * hTheta[0]
    return hJ, hgrad


def trainLinearReg(hX, hy, hLam):
    """
    最小化线性模型的参数
    :param hX: 训练数据的特征集
    :param hy: 训练数据的特征集对应的值
    :param hLam: 正则化参数
    :return: hTheta: 根据一定函数规则求出的参数值
    """
    hTheta = np.zeros((hX.shape[1], 1))
    hresult = op.minimize(fun=linearRegCostFunction, x0=hTheta, args=(hX, hy, hLam), method='TNC',
                          jac=True)
    # hJ = hresult.fun
    hTheta = hresult.x
    return hTheta


def learningCurve(hX, hy, hXVal, hyVal, hLam):
    """
    得出学习曲线的误差, 包括训练集和验证集的曲线, (训练集和验证集长度一致)
    :param hX: 训练集的特征集
    :param hy: 训练集的特征集对应的值
    :param hXVal: 验证集的特征集
    :param hyVal: 验证集的特征集对应的值
    :param hLam: 正则化参数
    :return: hErrorTrain: 训练集的误差; hErrorVal: 验证集的误差
    """
    hm = hX.shape[0]  # 此处为训练集的数量
    hErrorTrain = np.zeros((hm, 1))  # 训练集误差
    hErrorVal = np.zeros((hm, 1))  # 验证集误差

    # 利用for循环求参数, 然后存储
    for i in range(1, hm + 1):  # 需要从1开始, 否则无法计算J值, 会报错, ZeroDivisionError, 除数为0的错误. 且hm+1为计算全部训练集得出的
        hTheta = trainLinearReg(hX[:i, :], hy[:i, :], hLam)  # 首先根据训练集求出参数theta, 是根据样本数量变化的
        hErrorTrain[i - 1], _ = linearRegCostFunction(hTheta, hX[:i, :], hy[:i, :], hLam)  # 求出训练集的J
        hErrorVal[i - 1], _ = linearRegCostFunction(hTheta, hXVal, hyVal, hLam)  # 求出验证集的J
    return hErrorTrain, hErrorVal


def polyFeatures(hX, hp):
    """
    将X映射至高阶多项式中, 其中不包括x的零次方, 并且只包含单个数据的整次方
    :param hX: 数据的特征集
    :param hp: 多项式的系数
    :return: hXPoly: X映射至高阶多项式后的矩阵
    """
    hm = hX.shape[0]
    hXPoly = np.zeros((hm, hp))
    for i in range(hm):
        for j in range(1, hp + 1):
            hXPoly[i, j - 1] = hX[i] ** j
    return hXPoly


def featureNormalize(hXPoly):
    """
    Normalizes the feature in X
    :param hXPoly: 高阶数据集
    :return: hXPolyNorm: Normalization后的数据集; hMu: 数据集的均值; hSTD: 数据集的标准差
    """
    hMu = np.mean(hXPoly, axis=0)  # axis=0为压缩行, 对各列求平均值
    hSTD = np.std(hXPoly, axis=0, ddof=1)  # ddof=1时计算的是无偏样本标准差, 即除以n-1。此参数默认等于0, 计算的是有偏的, 即除以n
    hXPolyNorm = (hXPoly - hMu) / hSTD
    return hXPolyNorm, hMu, hSTD


def validationCurve(hLamVec, hX, hy, hXVal, hyVal):
    """
    Generate the train and validation errors needed to plot a validation curve that we can use to select lambda
    :param hLamVec: 提供的正则化参数集合
    :param hX: 训练集的特征集
    :param hy: 训练集的特征集对应的值
    :param hXVal: 验证集的特征集
    :param hyVal: 验证集的特征集对应的值
    :return: hErrorTrainVec: 求得的训练集的误差集合, 对应于正则化参数集合; hErrorValVec: 求得的验证集的误差集合
    """
    hErrorTrainVec = np.zeros((hLamVec.shape[0], 1))
    hErrorValVec = np.zeros((hLamVec.shape[0], 1))

    # 利用lamVec中的每一项循环求出对应的参数theta, 并计算相应的误差
    for i in range(hLamVec.shape[0]):
        hLam = hLamVec[i]
        hTheta = trainLinearReg(hX, hy, hLam)
        hErrorTrainVec[i], _ = linearRegCostFunction(hTheta, hX, hy, 0)
        hErrorValVec[i], _ = linearRegCostFunction(hTheta, hXVal, hyVal, 0)
    return hErrorTrainVec, hErrorValVec


if __name__ == '__main__':
    """
    Part 1 : Loading and Visualizing Data
    """
    print("Loading and Visualizing Data...")
    # 加载数据
    filename = 'ex5data1.mat'
    data = scio.loadmat(filename)
    # print(data)
    X = data['X']  # 训练集, (12, 1)
    y = data['y']  # (12, 1)
    XTest = data['Xtest']  # 测试集, (21, 1)
    yTest = data['ytest']  # (21, 1)
    XVal = data['Xval']  # 验证集, (21, 1)
    yVal = data['yval']  # (21, 1)
    X = np.insert(X, 0, values=1, axis=1)  # 前插一列
    XVal = np.insert(XVal, 0, values=1, axis=1)
    XTest = np.insert(XTest, 0, values=1, axis=1)

    m = X.shape[0]  # 训练集的数目

    # 绘制可视化图像
    plt.scatter(X[:, 1], y, c='r', marker='x')
    plt.title('Figure 1: Data')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.savefig(
        "D:\\PythonFiles\\MachineLearning\\RegularizedLinearRegressionAndBiasVSVariance\\Figure\\Scatter.png")
    plt.show()
    print('=' * 40)

    """
    Part 2 : Regularized Linear Regression Cost 
    """
    thetaLine = np.ones((2, 1))
    JLine, _ = linearRegCostFunction(thetaLine, X, y, 1)
    print("J Value: ", JLine)
    print("Expected J Value: 303.993192")
    print('=' * 40)

    """
    Part 3 : Regularized Linear Regression Gradient
    """
    JLine, gradLine = linearRegCostFunction(thetaLine, X, y, 1)
    print("grad value: ", gradLine)
    print("Expected grad Value: \n[-15.303016; 598.250744]")
    print('=' * 40)

    """
    Part 4 : Train Linear Regression
    """
    lamLine = 0
    thetaLine = trainLinearReg(X, y, lamLine)
    print(thetaLine)

    # 绘制拟合曲线图
    plt.scatter(X[:, 1], y, c='r', marker='x')
    plt.title('Figure 2: Linear Fit')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X[:, 1], hypothesis(thetaLine, X))
    plt.savefig(
        "D:\\PythonFiles\\MachineLearning\\RegularizedLinearRegressionAndBiasVSVariance\\Figure\\Fit.png")
    plt.show()
    print('=' * 40)

    """
    Part 5 : Learning Curve For Linear Regression
    """
    LamLearn = 0  # 求学习曲线时不需要计算正则项, 后面通过训练集得到参数时仍需要加入正则项
    ErrorTrain, ErrorVal = learningCurve(X, y, XVal, yVal, LamLearn)

    # 绘制学习曲线图
    xx = np.array([i for i in range(m)])
    plt.plot(xx, ErrorTrain, c='b')
    plt.plot(xx, ErrorVal, c='r')
    plt.title("Learning Curve for Linear Regression")
    plt.legend(["Train", "Cross Validation"])
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Error")
    plt.savefig(
        "D:\\PythonFiles\\MachineLearning\\RegularizedLinearRegressionAndBiasVSVariance\\Figure\\LearningCurve.png")
    plt.show()

    # 打印误差的表格
    print("# TrainingExamples\tTrainError\t\t\tCrossValidationError")
    for i in range(m):
        print("    \t{0}\t\t\t{1}\t\t{2}".format(i + 1, ErrorTrain[i], ErrorVal[i]))
    print('=' * 40)

    """
    Part 6 : Feature Mapping For Polynomial Regression
    """
    power = 8  # 多项式的阶数为8

    # 将X映射到高阶多项式并规范化(Normalize)
    XPoly = polyFeatures(X[:, 1], power)
    XPolyNorm, mu, std = featureNormalize(XPoly)
    XPolyNorm = np.insert(XPolyNorm, 0, values=1, axis=1)

    # 验证集与测试集的映射
    XValPoly = polyFeatures(XVal[:, 1], power)
    XValPolyNorm = (XValPoly - mu) / std
    XValPolyNorm = np.insert(XValPolyNorm, 0, values=1, axis=1)
    XTestPoly = polyFeatures(XTest[:, 1], power)
    XTestPolyNorm = (XTestPoly - mu) / std
    XTestPolyNorm = np.insert(XTestPolyNorm, 0, values=1, axis=1)

    print("Normalized Training Examples 1: ")
    print(XPolyNorm[0, :])
    print('=' * 40)

    """
    Part 7 : Learning Curve For Polynomial Regression
    """
    lamPoly = 0
    thetaPoly = trainLinearReg(XPolyNorm, y, lamPoly)
    xxPoly = np.linspace(-100, 100, 50)
    xxPolyNorm = polyFeatures(xxPoly, power)
    for i in range(xxPolyNorm.shape[0]):
        for j in range(xxPolyNorm.shape[1]):
            xxPolyNorm[i, j] = (xxPolyNorm[i, j] - mu[j]) / std[j]
    xxPolyNorm = np.insert(xxPolyNorm, 0, values=1, axis=1)
    yyPoly = hypothesis(thetaPoly, xxPolyNorm)

    # 绘制多项式拟合曲线图
    plt.scatter(X[:, 1], y, c='r')
    plt.plot(xxPoly, yyPoly, c='b')
    plt.title("Polynomial Regression Fit Lambda = 0")
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlim((-100, 100))
    plt.ylim((-10, 40))
    plt.savefig(
        "D:\\PythonFiles\\MachineLearning\\RegularizedLinearRegressionAndBiasVSVariance\\Figure\\PolyFit.png")
    plt.show()

    # 绘制学习曲线图
    ErrorTrainPoly, ErrorValPoly = learningCurve(XPolyNorm, y, XValPolyNorm, yVal, lamPoly)
    xxPolyLearn = np.array([i for i in range(m)])
    plt.plot(xxPolyLearn, ErrorTrainPoly, c='b')
    plt.plot(xxPolyLearn, ErrorValPoly, c='r')
    plt.title("Polynomial Regression Learning Curve Lambda = 0")
    plt.legend(["Train", "Cross Validation"])
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.savefig(
        "D:\\PythonFiles\\MachineLearning\\RegularizedLinearRegressionAndBiasVSVariance\\Figure\\PolyLearnCurve.png")
    plt.show()

    # 打印误差的表格
    print("Polynomial Regression Lambda = ", lamPoly)
    print("# TrainingExamples\tTrainError\t\t\tCrossValidationError")
    for i in range(m):
        print("    \t{0}\t\t\t{1}\t\t{2}".format(i + 1, ErrorTrainPoly[i], ErrorValPoly[i]))
    print('=' * 40)

    """
    Part 8 : Validation For Selecting Lambda
    """
    lamVec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T
    ErrorTrainVec, ErrorValVec = validationCurve(lamVec, XPolyNorm, y, XValPolyNorm, yVal)

    # 绘制lambda与误差变化趋势图
    plt.plot(lamVec, ErrorTrainVec, c='b')
    plt.plot(lamVec, ErrorValVec, c='r')
    plt.title("Error change with lambda")
    plt.legend(["Train", "Cross Validation"])
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.savefig(
        "D:\\PythonFiles\\MachineLearning\\RegularizedLinearRegressionAndBiasVSVariance\\Figure\\ErrorLam.png")
    plt.show()

    print("lambda\t\t\tTrainError\t\t\t\tValidationError")
    for i in range(lamVec.shape[0]):
        print("  %.3f\t\t\t%.4f\t\t\t\t%.4f" % (lamVec[i], ErrorTrainVec[i], ErrorValVec[i]))
    print('=' * 40)
