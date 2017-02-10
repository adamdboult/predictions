#!/usr/bin/env python3
#############
# Libraries #
#############
import numpy as np
from numpy import inf
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import json

#######################
# Model specification #
#######################
#dataSource = "data.json"
dataSource = sys.argv[1]

yCutoff = [0]
feat = 6

runClass = 1

hiddenLayers = 0
hiddenLayerLength = 0

#######################
# Algorithm constants #
#######################
costDiffCutoff = 1000000
costDiffCount = 100
trainProp = 0.6
cvProp = 0.2

alpha = 1
thetaRange = 1
alphaReduce = 0.1
thetaRangeReduce = 0.1

gradEps = 0.0001

#############
# Functions #
#############
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def poly_size_f(feat, d):
    return int(math.factorial((feat - 1) + d) / (math.factorial(feat - 1) * math.factorial(d)))

def polynomial(X, d):
    feat = X.shape[1]
    m = X.shape[0]
    poly_size = poly_size_f(feat, d)

    X_prime = np.zeros((m, poly_size))

    poly_array_temp = np.int8(np.array(np.zeros((feat, 1), dtype = np.int)))
    poly_array_temp[feat - 1] = d

    for i in range(0, poly_size):
        new_poly_array = np.ones((m, 1))
        for j in range(0, poly_array_temp.shape[0]):
            for k in range(0, poly_array_temp[j]):
                new_poly_array = new_poly_array * X[:, j: j + 1]

        X_prime[:, i: i + 1] = new_poly_array

        poly_array_index = 0
        for j in range(1, feat):
            if (int(poly_array_temp[j]) > 0):
                if (poly_array_index == 0):
                    poly_array_index = j

        poly_array_fall = np.zeros((feat, 1))
        poly_array_fall[0: poly_array_index - 1] = 1
        poly_array_fall_v = - poly_array_fall * poly_array_temp

        poly_array_suck_v = np.zeros((feat, 1))
        poly_array_suck_v[poly_array_index - 1] = np.sum(poly_array_fall * poly_array_temp) + 1

        poly_array_push_v = np.zeros((feat, 1))
        poly_array_push_v[poly_array_index] = - 1

        poly_array_temp = poly_array_temp + poly_array_suck_v + poly_array_push_v + poly_array_fall_v
        poly_array_temp = np.int8(poly_array_temp)

    return (X_prime)

def minMax(X):
    x_min = np.amin(X, axis = 0)
    x_max = np.amax(X, axis = 0)

    return (x_min, x_max)

def normalise(X, x_max, x_min):
    feat = X.shape[1]
    mul = np.ones((1, feat))
    add = np.zeros((1, feat))
    diff = x_max - x_min
    for i in range(0, feat):
        if (diff[i] > 0):
            mul[0: 1, i: i + 1] = 1 / diff[i]
            add[0: 1, i: i + 1] = - x_min[0: 1, i: i + 1] / diff[i]

    X_prime = X * mul + add

    return (X_prime)

def thetaInit(feat, outcomes, thetaRange, hiddenLayers, hiddenLayerLength):
    layers = hiddenLayers + 1
    nextLength = hiddenLayerLength
    thisLenth = feat
    theta = []
    thetaCandidate = []
    for layer in range(0, layers):
        if (layer == hiddenLayers):
            nextLength = outcomes
        thetaCandidate.append(np.random.rand(feat, nextLength) * 2 * thetaRange - thetaRange)
        theta.append(np.zeros((feat, nextLength)) * 2 * thetaRange - thetaRange)
        thisLength = nextLength
    return theta, thetaCandidate

def getHypothesis(X, theta, runClass):
    if (runClass == 1):
        hypothesis = X
        for i in range(0, len(theta)):
            if (i > 0):
                oneCol = np.ones((m, 1))
                hypothesis = np.append(oneCol, hypothesis, axis = 1)
            hypothesis = sigmoid(np.dot(hypothesis, theta[i]))
    else:
        hypothesis = np.dot(X, theta[0])
    return (hypothesis)

def getLoss(y, hypothesis, m, runClass, theta, thetaLossWeight):
    regLoss = (1 / (2 * m)) * (np.sum((theta[0] ** 2) * thetaLossWeight))

    if (runClass == 1):
        hypo_temp_flip = hypo_temp = np.zeros_like(hypothesis)
        
        hypo_temp[:] = hypothesis
        hypo_temp[y == 0] = 1
        hypo_temp = np.log(hypo_temp)
        
        hypo_temp_flip[:] = 1 - hypothesis 
        hypo_temp_flip[y == 1] = 1
        hypo_temp_flip = np.log(hypo_temp_flip)

        hypo_temp_combined = hypo_temp + hypo_temp_flip
        loss = (-1 / m) * (hypo_temp_combined) + regLoss
    else:
        loss = (1 / (2 * m)) * ((hypothesis - y) ** 2) + regLoss
    return loss

def getGradient(X, y, hypothesis, runClass, theta, thetaLossWeight):
    # works for both classification and regression, because maths
    m = X.shape[0]
    updateVector = hypothesis - y
    transX = X.transpose()
    gradient = (1 / m) * np.dot(transX, updateVector) + theta[0] * thetaLossWeight
    return (gradient)

def gradDescent(X, y, outcomes, runClass, lam, alpha, thetaRange, alphaReduce, thetaRangeReduce, costDiffCutoff, costDiffCount, gradEps, hiddenLayers, hiddenLayerLength):
    y = np.int8(y)
    m = X.shape[0]
    feat = X.shape[1]

    theta, thetaCandidate = thetaInit(feat, outcomes, thetaRange, hiddenLayers, hiddenLayerLength)
    
    thetaLossWeight = np.ones((feat, outcomes)) * lam
    thetaLossWeight[0: 1, :] = 0
    thetaInitAccepted = 0

    gradChecked = 0
    cost = math.inf
    stopCount = 0
    i = 0
    while (costDiffCount > stopCount):
        hypothesis = getHypothesis(X, thetaCandidate, runClass)
        loss = getLoss(y, hypothesis, m, runClass, thetaCandidate, thetaLossWeight)

        costCandidate =  np.sum(loss)

        rejectCandidate = 0
        costDiff = 0
        if (math.isnan(costCandidate) or math.isinf(costCandidate or costCandidate > cost)):
            if (thetaInitAccepted == 0):
                thetaRange = thetaRange * thetaRangeReduce
                theta, thetaCandidate = thetaInit(feat, outcomes, thetaRange, hiddenLayers, hiddenLayerLength)
            else:
                alpha = alpha * alphaReduce
                gradient = getGradient(X, y, hypothesis, runClass, theta, thetaLossWeight)
                thetaCandidate[0] = theta[0] - alpha * gradient
        else:
            thetaInitAccepted = 1

            costDiff = cost / (cost - costCandidate)
            if (costDiff >= costDiffCutoff):
                stopCount = stopCount + 1
            else:
                stopCount = 0

            theta[0] = thetaCandidate[0]
            gradient = getGradient(X, y, hypothesis, runClass, theta, thetaLossWeight)
            thetaCandidate[0] = theta[0] - alpha * gradient
            cost = costCandidate

            if (gradChecked == 0):
                gradChecked = 1
                theta_grad_plus = []
                theta_grad_less = []
                theta_grad_plus.append(np.zeros_like(theta[0]))
                theta_grad_less.append(np.zeros_like(theta[0]))
                theta_grad_plus[0][:] = theta[0]
                theta_grad_less[0][:] = theta[0]

                theta_grad_plus[0][0, 0] = theta[0][0, 0] * (1 + gradEps)
                theta_grad_less[0][0, 0] = theta[0][0, 0] * (1 - gradEps)
                hypothesis_grad_plus = getHypothesis(X, theta_grad_plus, runClass)
                hypothesis_grad_less = getHypothesis(X, theta_grad_less, runClass)

                gradient_plus = getGradient(X, y, hypothesis_grad_plus, runClass, theta_grad_plus, thetaLossWeight)
                gradient_less = getGradient(X, y, hypothesis_grad_less, runClass, theta_grad_less, thetaLossWeight)

                loss_grad_plus = getLoss(y, hypothesis_grad_plus, m, runClass, theta_grad_plus, thetaLossWeight)
                loss_grad_less = getLoss(y, hypothesis_grad_less, m, runClass, theta_grad_less, thetaLossWeight)
                
                cost_plus = np.sum(loss_grad_plus)
                cost_less = np.sum(loss_grad_less)

                cost_plus_0 = np.sum(loss_grad_plus[:, 0: 1])
                cost_less_0 = np.sum(loss_grad_less[:, 0: 1])

                cost_plus_1 = np.sum(loss_grad_plus[:, 1: 2])
                cost_less_1 = np.sum(loss_grad_less[:, 1: 2])

                gradientTest = (cost_plus - cost_less) / (2 * theta[0][0, 0] * gradEps)
                gradientTest_0 = (cost_plus_0 - cost_less_0) / (2 * theta[0][0, 0] * gradEps)
                gradientTest_1 = (cost_plus_1 - cost_less_1) / (2 * theta[0][0, 0] * gradEps)

                print("Gradient| Manual: %f | Standard: %f" % (gradientTest, gradient[0, 0]))
                print (gradient[0, 0] / gradientTest)
                print (gradientTest_0)
                input("Press Enter to continue...")
        i = i + 1
        print ("Iteration: %d | Cost: %f| Progress: %f" % (i, costCandidate, costDiff / costDiffCutoff))

    return (theta, cost)
        
###########
# Extract #
###########
with open(dataSource, encoding='utf-8') as data_file:
    jsonModel = json.loads(data_file.read())

if (jsonModel["type"] == "excel"):
    csv = pd.read_excel(jsonModel["fileName"], jsonModel["importSheet"], index_col = None, na_values = jsonModel["naArray"])

    if ("rowKeep" in jsonModel):
        keepLength = len(jsonModel["rowKeep"])
        for i in range(0, keepLength):
            csv = csv.drop(csv[csv[jsonModel["rowKeep"][i][0]] != jsonModel["rowKeep"][i][1]].index)

    if ("rowDrop" in jsonModel):
        dropLength = len(jsonModel["rowDrop"])
        for i in range(0, dropLength):
            csv = csv.drop(csv[csv[jsonModel["rowDrop"][i][0]] == jsonModel["rowDrop"][i][1]].index)

    if ("i_range" in jsonModel):
        if (jsonModel["i_range"] == 0):
            jsonModel["i_range"] = csv.shape[0]
    else:
        jsonModel["i_range"] = csv.shape[0]

    if ("j_range" in jsonModel):
        if (jsonModel["j_range"] == 0):
            jsonModel["j_range"] = csv.shape[1]
    else:
        jsonModel["j_range"] = csv.shape[1]

    if (feat == 0):
        feat = jsonModel["j_range"]

    for i in range(0, jsonModel["i_range"]):
        for j in range(0, 1 + jsonModel["j_range"] - feat):
            row = i * jsonModel["j_range"] + j
            dfTemp = np.array(csv.iloc[jsonModel["yOffset"] + i: jsonModel["yOffset"] + i + 1, jsonModel["xOffset"] + j - feat: jsonModel["xOffset"] + j])
            
            if ((i + j) == 0):
                newArray = np.array(dfTemp)
            else:
                newArray = np.insert(newArray, [1], dfTemp, axis = 0)

    df = pd.DataFrame(data = newArray)
    df = df.dropna()

    m = df.shape[0]

    X = np.array(df.iloc[:, 0: feat - 1], dtype = float)
    y = np.array(df.iloc[:, feat - 1: feat], dtype = float)

    oneCol = np.ones((m, 1))
    X = np.append(oneCol, X, axis = 1)

    if (runClass == 1):
        y_temp = np.zeros_like (y)
        y_temp[:] = y

        y_temp[y < yCutoff] = 1
        y_temp[y >= yCutoff] = 0

        y[:] = y_temp

        outcomes = int(np.amax(y, axis = 0) + 1)
    
        yArray = np.zeros((m, outcomes))
    
        for i in range(0, outcomes):
            yArray[:, i: i + 1][y == i] = 1

        y = np.zeros_like (yArray)
        y[:] = yArray

####################
# Training/CV/Test #
####################
np.random.shuffle(X)

trainCutoff = int(trainProp * m)
cvCutoff = int((cvProp + trainProp) * m)

X_train = X[0: trainCutoff, :]
X_cv = X[trainCutoff: cvCutoff, :]
X_test = X[cvCutoff: m, :]

y_train = y[0:trainCutoff, :]
y_cv = y[trainCutoff:cvCutoff, :]
y_test = y[cvCutoff: m, :]

##########################
# Polynomial calibration #
##########################
polynomial_d = 3

#poly_size = poly_size_f(feat, polynomial_d)

######################
# Lambda calibration #
######################
lam = 1

####################
# Theta estimation #
####################
X_prime = polynomial(X_train, polynomial_d)

x_max, x_min = minMax(X_prime)

X_prime = normalise(X_prime, x_max, x_min)

theta, cost = gradDescent(X_prime, y_train, outcomes, runClass, lam, alpha, thetaRange, alphaReduce, thetaRangeReduce, costDiffCutoff, costDiffCount, gradEps, hiddenLayers, hiddenLayerLength)

###########
# Predict #
###########
X_test = polynomial(X_test, polynomial_d)
X_test = normalise(X_test, x_max, x_min)

m_test = X_test.shape[0]

predict = getHypothesis(X_test, theta, runClass)

if (runClass == 1):
    predict =  np.int8(predict)
    y_test = np.int8(y_test)

    maxProb = np.argmax(predict, axis = 1)

    predict_temp = np.zeros_like (predict)
    for i in range(0, m_test):
        predict_temp[i: i + 1, maxProb[i]: maxProb[i] + 1] = 1
    predict[:] = predict_temp

    for i in range(0, outcomes):
        truePositive = np.sum(y_test[:, i: i + 1] * predict[:, i: i + 1])
        trueNegative = np.sum((1 - y_test[:, i: i + 1]) * (1 - predict[:, i: i + 1]))
        falsePositive = np.sum((1- y_test[:, i: i + 1]) * predict[:, i: i + 1])
        falseNegative = np.sum(y_test[:, i: i + 1] * (1 - predict[:, i: i + 1]))

        defaultPositive = np.sum(y_test[:, i: i + 1]) / m_test
        if (defaultPositive < 0.5):
            defaultPositive = 1 - defaultPositive

        accuracy = (truePositive + trueNegative) / m_test
        precision = truePositive / (truePositive + falsePositive)
        recall = truePositive / (truePositive + falseNegative)
        F1_score = 2 * precision * recall / (precision + recall)

        print ()
        print ("For answer: %d" % (i))
        print ("True positive: %d" % (truePositive))
        print ("True negative: %d" % (trueNegative))
        print ("False positive: %d" % (falsePositive))
        print ("False negative: %d" % (falseNegative))

        print ("Accuracy: %f%%" % (accuracy * 100))
        print ("Default accuracy: %f%%" % (defaultPositive * 100))
        print ("Sample size: %d" % (m_test))

        print ("Precision: %f%%" % (precision * 100))
        print ("Recall: %f%%" % (recall * 100))
        print ("F1 score: %f" % (F1_score))

else:
    resids = y_test - predict
    RSS = np.sum(resids ** 2)
    TSS = np.sum((y_test - np.mean(y_test)) ** 2)
    R2 = 1 - RSS / TSS
    print (R2)

    plt.plot(y_test, predict, 'o', label='Original data', markersize=10)
    #plt.plot(x, m*x + c, 'r', label='Fitted line')
    plt.legend()
    plt.show()

print ("theta")
print (theta[0])
