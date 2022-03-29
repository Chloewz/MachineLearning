"""
Created by Chloe on 2022/3/24
svm支持向量机邮件分类--是否为垃圾邮件
耗费时间: 将邮件转换为特征向量和后续提取权重之后对于字典的操作
查阅--对邮件的处理可以使用正则表达式
TypeError: expected string or bytes-like object--re.sub()遇到, 原因是在输入正则表达式过程中没有找到相应的字符串。转化为str即可
查阅--将读取的txt文件转换为字典--按行读取, 形成嵌套列表再使用dict()创建字典
查阅--for循环遍历字典, for i in dict, 自动识别dict是一个字典, i就是dict的键, dict[i]就是他的值
出错--将邮件向量映射为数字的时候, 如果在原文本向量上修改时会出错。 通过建立一个新列表存储映射数值解决这个问题
查阅--sklearn里的model.coef_可以查看权重
查阅--将二维数组/多维数组转换为一维--array.flatten()
查阅--字典的键值对互换--zip()的用法
查阅--字典按值排序
此code的model.predict使用的是reshape(1, -1), 目的就是让他变成两列, 万不可一概而过
"""

import numpy as np
import time
import nltk.stem.porter as stem
import re
import scipy.io as scio
from sklearn import svm


def processEmail(hEmailContents):
    """
    preprocesses a the body of an email and returns a list of word indices(指数、目录)
    :param hEmailContents: 邮件内容
    :return: hEmailList: 邮件内容列表, 经过一定规则处理, 元素是单词
    """
    hEmailContents = hEmailContents.lower()
    hEmailContents = re.sub(r'<.*>', '', hEmailContents)  # 移除html标签
    hEmailContents = re.sub(r'http[s]?://.+', 'httpaddr', hEmailContents)  # 移除url
    hEmailContents = re.sub(r'[\S]+@[\w]+.[\w]+', 'emailaddr', hEmailContents)  # 移除邮箱
    hEmailContents = re.sub(r'[\$][0-9]+', 'dollar number', hEmailContents)  # 移除$, 解决dollar和number连接的问题
    hEmailContents = re.sub(r'\$', 'dollar number', hEmailContents)  # 移除单个$
    hEmailContents = re.sub(r'[0-9]+', 'number', hEmailContents)  # 移除数字
    hEmailContents = re.sub(r'[\W]+', ' ', hEmailContents)  # 移除字符
    hEmailList = hEmailContents.split(' ')

    if hEmailList[0] == '':  # 分开时会导致开始空格出多出两个空字符, 分别在最开始和最后
        hEmailList = hEmailList[1:-1]
    hPorterStemmer = stem.PorterStemmer()
    for i in range(len(hEmailList)):
        hEmailList[i] = hPorterStemmer.stem(hEmailList[i])  # 提取每个单词的词干
    return hEmailList


def mappingEmail(hEmailList, hVocab):
    """
    将邮件单词列表映射为编号
    :param hVocab: 词汇表, 已提供
    :param hEmailList: 邮件的单词列表
    :return: hWordIndices: 单词的映射表, 将每一个单词转换为了词汇表对应的编号, 并且删去了没有对照的单词
    """
    hWordIndices = []
    for i in range(len(hEmailList)):
        for j in hVocab:
            if hEmailList[i] == hVocab[j]:  # 删除没有对照的单词, 不显示其编号
                hWordIndices.append(int(j) + 1)
    return hWordIndices


def emailFeatures(hWordIndices, hVocab):
    """
    takes in a wordIndices vector and produces a feature vector from the word indices
    :param hVocab: 词汇表, 已提供
    :param hWordIndices: 邮件的单词映射表
    :return: hFeatures: 邮件的特征向量
    """
    hFeatures = np.zeros((len(hVocab),))
    for i in range(len(hWordIndices)):
        index = int(hWordIndices[i])
        hFeatures[index - 1] = 1
    return hFeatures


def getVocabList(hCoe, hVocab):
    """
    reads the fixed vocabulary list in vacab.txt and returns a cell array of words
    :param hCoe: 由model.coef_的到来的系数权重列表
    :param hVocab: 词汇表
    :return: hVocabList: 根据权重大小进行排列的词汇表
    """
    hVocabList = {}
    for hkey, hVal in hVocab.items():
        hVocabList[hVal] = hkey  # 原词汇字典的键值对互换

    # 将系数的列表每一项赋给词汇字典的值
    if len(hVocabList) == len(hCoe):
        hVocabList = dict(zip(hVocabList.keys(), hCoe))
    else:
        print("Match Error!")

    # 将词汇列表按照权重大小倒序排列
    hVocabList = sorted(hVocabList.items(), key=lambda x: x[1], reverse=True)
    hVocabList = dict(hVocabList)
    return hVocabList


if __name__ == '__main__':
    startTime = time.perf_counter()
    """
    Part 1: Email Processing
    """
    print("Preprocessing sample email (emailSample1.txt)")
    # 提取特征
    # 读取邮件
    filenameEmail = 'EmailTXT/emailSample1.txt'
    f = open(filenameEmail, 'r')
    emailContents = f.read()
    f.close()

    # 将邮件按照一定的原则处理, 输出结果为列表. 列表的每个元素是单词
    emailList = processEmail(emailContents)
    print("Email List:\n", emailList)

    # 将邮件单词映射为数字编号
    filenameVocab = 'EmailTXT/vocab.txt'

    with open(filenameVocab, 'r') as f:
        dic = []  # 嵌套列表
        for line in f.readlines():
            line = line.strip('\n')  # 去掉换行符
            v = line.split('\t')  # 将每一个以制表符为分隔符的行转换成列表
            dic.append(v)  # 加入一个临时的列表中
    vocab = dict(dic)  # dict()用于创建一个字典
    # vocab = list(vocab.values())

    wordIndices = mappingEmail(emailList, vocab)
    print("Word Indices:\n", wordIndices)
    print('=' * 40)

    """
    Part 2: Feature Extraction
    """
    print("Extracting features from sample emails (emailSample1.txt)")
    features = emailFeatures(wordIndices, vocab)
    print("Length of feature vector:\n", len(features))
    print("Number of non-zeros entries:\n", np.sum(features > 0))
    print('=' * 40)

    """
    Part 3: Train Linear SVM for Spam Classification
    """
    # 使用的是提供的现成的数据来拟合模型
    filenameTrain = 'spamTrain.mat'
    dataTrain = scio.loadmat(filenameTrain)
    XTrain = dataTrain['X']
    yTrain = dataTrain['y']

    print("Training Linear SVM (Spam Classification)...")
    CTrain = 0.1
    modelTrain = svm.SVC(C=CTrain, kernel='linear')
    modelTrain.fit(XTrain, yTrain.ravel())
    scoreTrain = modelTrain.score(XTrain, yTrain.ravel())
    print("Score of Training Set: ", scoreTrain)

    """
    Part 4: Test Spam Classification
    """
    filenameTest = 'spamTest.mat'
    dataTest = scio.loadmat(filenameTest)
    XTest = dataTest['Xtest']
    yTest = dataTest['ytest']

    print("Evaluating the trained Linear SVM on a test set...")
    scoreTest = modelTrain.score(XTest, yTest.ravel())
    print("Score of Test Set: ", scoreTest)
    print('=' * 40)

    """
    Part 5: Top Predictors of Spam
    """
    # 对权重排序并获得词汇列表
    coefficients = modelTrain.coef_  # 获取系数
    # coefficients = np.reshape(coefficients, (coefficients.shape[1], 1)).tolist()  # 转换为一个列表
    coefficients = coefficients.flatten().tolist()  # tolist()将array数组转换为列表
    vocabList = getVocabList(coefficients, vocab)

    # 打印前十五个权重单词及其对应的权重
    print("The weights rank the top fifteen coefficients and their words: ")
    for i, (key, val) in enumerate(vocabList.items()):
        print({key, val})
        if i == 14:
            print("Printed Over")
            break
    print('=' * 40)

    """
    Part 6: Try Your Own Emails
    """
    filenamePred = 'EmailTXT/emailSample2.txt'
    with open(filenamePred, 'r') as f:
        emailPredContents = f.read()
    emailPredList = processEmail(emailPredContents)
    wordIndicesPred = np.array(mappingEmail(emailPredList, vocab))
    featuresPred = emailFeatures(wordIndicesPred, vocab)
    predict = modelTrain.predict(featuresPred.reshape(1, -1))
    print("The Predict Result: ", predict)
    print('=' * 40)

    endTime = time.perf_counter()
    print("Using time: ", endTime - startTime)
