from __future__ import division
from math import exp
import pandas as pd
from numpy import *
from random import normalvariate
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


# 加载相应的数据集
def load_data(data_path):
    data = pd.read_csv(data_path)
    label = data['label'][:350000]
    data = data.iloc[:350000, 1:]
    # 要把类别数据变成数字索引才能进行训练
    sparse_feats = []
    dense_feats = []
    for f in data.columns:
        if f[0] == 'C':
            sparse_feats.append(f)
            data[f].fillna('-1', inplace=True)
            vec = data[f].unique().tolist()
            le = LabelEncoder()
            le.fit(vec)
            data[f] = le.transform(data[f].values.tolist())
        else:
            dense_feats.append(f)
            data[f].fillna(0, inplace=True)

    # 这里设置一样random_state，目的是为了保持和下面的GBDT模型一致
    for i in range(len(label)):
        if label[i] == 0:
            label[i] = -1
        else:
            label[i] = 1
    return data, label


def load_train_data(data, labelMat):
    global min_max_scaler
    X_train = np.array(data)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    return X_train_minmax, labelMat


def laod_test_data(data, labelMat):
    X_test = np.array(data)
    X_tset_minmax = min_max_scaler.transform(X_test)
    return X_tset_minmax, labelMat


def sigmoid(inx):
    return 1. / (1. + exp(-max(min(inx, 10), -10)))


def FM_function(dataMatrix, classLabels, k, iter):
    m, n = shape(dataMatrix)
    alpha = 0.01
    w = zeros((n, 1))
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))
    for it in range(iter):
        for x in range(m):
            inter_1 = np.dot(dataMatrix[x] ,v)
            inter_2 = np.dot(multiply(dataMatrix[x], dataMatrix[x]) ,multiply(v, v))
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            p = w_0 + np.dot(dataMatrix[x] ,w) + interaction  #
            loss = sigmoid(classLabels[x] * p[0][0]) - 1
            w_0 = w_0 - alpha * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (
                        dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
    return w_0, w, v


def Assessment(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):
        allItem += 1
        inter_1 = np.dot(dataMatrix[x] ,v)
        inter_2 = np.dot(multiply(dataMatrix[x], dataMatrix[x]) ,multiply(v, v))
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + np.dot(dataMatrix[x] ,w) + interaction
        pre = sigmoid(p[0][0])
        result.append(pre)
        if pre < 0.5 and classLabels[x] == 1:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1:
            error += 1
        else:
            continue
    return float(error) / allItem


if __name__ == '__main__':
    # -------读取数据----------
    trainData = 'train.txt'
    testData = 'test.txt'
    total_data, total_label = load_data("./data/criteo_sampled_data.csv")
    x_train, x_test, y_train, y_test = train_test_split(total_data, total_label, test_size=0.3)
    # ------模型训练----
    dataTrain, labelTrain = load_train_data(x_train, y_train)
    labelTrain = np.array(labelTrain)
    dataTest, labelTest = laod_test_data(x_test, y_test)
    labelTest = np.array(labelTest)
    w_0, w, v = FM_function(mat(dataTrain), labelTrain, 15, 20)
    result = Assessment(dataTest, labelTest, w_0, w, v)
    print("the final acc is : {}".format(result))

