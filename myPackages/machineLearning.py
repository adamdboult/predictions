################
# Dependencies #
################
# Sci
import pandas as pd
import numpy as np

import pickle

# General
import math
import os
import string

# Preprocessing
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Cross validation
from sklearn.model_selection import GridSearchCV

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

###############################
# Classification / Regression #
###############################
def isClf(y):
    cutOff = 0.1

    sampleSize = len(y)
    print ("Sample size: " + str(sampleSize))

    uniques = len(np.unique(y))
    print ("Unique y values: " + str(uniques))

    ratio = uniques / sampleSize
    if ratio < cutOff:
        return True
    return False

############
# Training #
############
def trainClf(X, y):
    print ("Type: Classification")

    names = []
    classifiers = []
    hyperParameters = []

    ####
    # Decision tree 
    ####
    clf = DecisionTreeClassifier()

    maxDepthArray = np.arange(1,51,1)
    minSamplesSplitArray = np.arange(2,11,1)

    parameters = [{"max_depth":maxDepthArray, "min_samples_split": minSamplesSplitArray}]

    names.append("Decision tree")
    classifiers.append(clf)
    hyperParameters.append(parameters)

    ####
    # Random Forest (bagging)
    ####
    clf = RandomForestClassifier()

    maxDepthArray = np.arange(1,51,1)
    minSamplesSplitArray = np.arange(2,11,1)

    parameters = [{"max_depth":maxDepthArray, "min_samples_split": minSamplesSplitArray}]

    names.append("Random forest")
    classifiers.append(clf)
    hyperParameters.append(parameters)

    ####
    # Adaboost (boosting)
    ####
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 10, min_samples_split = 5))

    n_estimatorsArray = [1]

    parameters = [{"n_estimators": n_estimatorsArray}]

    names.append("Adaboost")
    classifiers.append(clf)
    hyperParameters.append(parameters)
    ####
    # Logistic regression
    ####
    clf = LogisticRegression()

    penaltyArray = ["l1", "l2"]

    parameters = [{}]

    names.append("Logistic regression")
    classifiers.append(clf)
    hyperParameters.append(parameters)

    ####
    # Naive bayes
    ####
    clf = GaussianNB()

    parameters = [{}]

    names.append("Naive bayes")
    classifiers.append(clf)
    hyperParameters.append(parameters)

    ####
    # K-nearest neighbours
    ####
    clf = KNeighborsClassifier()

    nNeighborsArray = [5]
    parameters = [{"n_neighbors": nNeighborsArray}]

    names.append("K-nearest neighbours")
    classifiers.append(clf)
    hyperParameters.append(parameters)

    ####
    # Support Vector Classifier
    ####
    clf = SVC()

    cArray = [1]
    degreeArray = [1]
    gammaArray = [0.3]
    kernelArray = ["poly"]

    parameters = [{"kernel": kernelArray, "degree": degreeArray, "gamma": gammaArray, "C": cArray}]

    names.append("Support vector classifier")
    classifiers.append(clf)
    hyperParameters.append(parameters)

    ####
    # Multi-Layer Perceptron
    ####
    clf = MLPClassifier(hidden_layer_sizes=(100,50,50))

    alphaArray = [1e-05]

    parameters = [{"alpha": alphaArray}]

    names.append("Multi-layer perceptron")
    classifiers.append(clf)
    hyperParameters.append(parameters)

    ####
    # Train
    ####
    for i in range(len(classifiers)):
        print ("\nTraining: " + str(classifiers[i]))
        bestParameters = crossValidate(X, y, classifiers[i], hyperParameters[i])

        classifiers[i].set_params(**bestParameters)
        classifiers[i].fit(X, y)

        score = classifiers[i].score(X, y)
        print (score)
        path = open("models/" + names[i] + ".sav", "wb")
        s = pickle.dump(classifiers[i], path)
    return

def trainReg(X, y):
    print ("Type: Regression")

    names = []
    regressors = []
    hyperParameters = []

    ####
    # Train
    ####
    for i in range(len(regressors)):
        print ("\nTraining: " + str(regressors[i]))
        bestParameters = crossValidate(X, y, regressors[i], hyperParameters[i])

        regressors[i].set_params(**bestParameters)
        regressors[i].fit(X, y)

        score = regressors[i].score(X, y)
        print (score)
        path = open("models/" + names[i] + ".sav", "wb")
        s = pickle.dump(regressors[i], path)
    return

def train(X, y):
    print ("\nIdentifying type of problem...")
    if isClf(y):
        trainClf(X, y)
        return
    else:
        print ("Type: Regression")
        trainReg(X, y)
        return

#def ensemblePredict(X, classifiers, weights):
#    normalWeights = []
#    totalWeight = sum(weights)
#    for weight in weights:
#         normalWeights.append(weight/totalWeight)
#    for i in range(len(classifiers)):
#        print (classifiers[i])
#        y_ind = classifiers[i].predict(X)
#        y = y + y_ind * normalWeights[i]
#    return y

def crossValidate(X, y, model, parameters):
    kSplits = 2
    grid_search = GridSearchCV(model, parameters, cv = kSplits)
    #print (parameters)
    #print (X)
    grid_search.fit(X, y)
    return grid_search.best_params_

################
# Read / Write #
################
def readCSV(projectName, fileName):
    fileName 
    print ("\nReading CSV...")
    path = os.path.join("raw", projectName, fileName)
    df = pd.read_csv(path)
    return df

def writeCSV(fileName, df):
    print ("\nWriting CSV...")
    df.to_csv(fileName, index = False, header = True)

############
# Cleaning #
############
def text_process(text):
    #print ("PROCESSING TEXT...")
    #print (text)	
    text = str(text)
    text = [char for char in text if char not in string.punctuation]
    text = "".join(text)

    text = text.lower()
    
    text = [word for word in text.split()]# if word not in stopWords]
    return text

def textExtraction(df, series):
    #print ("EXTRACTING TEXT...")
    vectorizer = CountVectorizer(analyzer = text_process, min_df = 0.1)
    df[series] = df[series].replace(np.nan, '', regex=True)
    vectorizer.fit_transform(df[series])
    vocab = vectorizer.get_feature_names()
    
    return vocab

def getVocab(df):
    print("\nGetting vocabulary...")
    vocabDict = {}
    for column in df.columns:
        cType = df[column].dtype
        if cType != np.float64 and cType != np.int64:
            #print (column + ": String")
            try: 
                vocab = textExtraction(df, column)
                vocabDict[column] = vocab
                print ("- \"" + column + "\" has a vocabulary\n--\t"+ str(vocab))

            except:
                vocabDict[column] = []
                print ("- \"" + column+ "\" does not have a vocabulary")

        else:
            print ("- \"" + column+ "\" is already numerical")
            #print (column + ": Number")

    return vocabDict

def processX(df, yColumn, vocab):
    print ("\nProcessing independent variables (X)...")
    # Exclude y
    if yColumn in df:
        df = df.drop([yColumn], axis = 1)

    # Text to vector
    for column in df.columns:

        if column in vocab:
            vectorizer = CountVectorizer(analyzer = text_process, vocabulary = vocab[column])
            if len(vectorizer.vocabulary) > 0:
                vector = vectorizer.fit_transform(df[column])
                i = 0
                vector = vector.toarray()
                for each in vector.T:
                    new_name = column + "_" + str(i)
                    df[new_name] = vector.T[i]
                    i = i + 1
            df = df.drop([column], axis = 1)

    # Impute
    imp = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
    values = imp.fit_transform(df.values)
    df = pd.DataFrame(values, index = df.index, columns = df.columns)

    return df

def processY(df, yColumn):
    print ("\nProcessing dependent variable (Y)...")
    y = df[yColumn]
    y = y.values.ravel()
    return y

##############
# Predicting #
##############
def predict(X):
    print ("Predicting...")

    for i in range(len(regressors)):
        print ("\nTraining: " + str(regressors[i]))
        bestParameters = crossValidate(X, y, regressors[i], hyperParameters[i])

        regressors[i].set_params(**bestParameters)
        regressors[i].fit(X, y)

        score = regressors[i].score(X, y)
        print (score)
        path = open("models/" + names[i] + ".sav", "wb")
        s = pickle.dump(regressors[i], path)
    return y


