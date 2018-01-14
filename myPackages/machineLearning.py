################
# Dependencies #
################
# Sci
import pandas as pd
import numpy as np

# General
import math
import os
import string
import pickle

# Workflow
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

# Preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB


################
# Master train #
################
def train(X, y, projectName):
    print ("\nIdentifying type of problem...")

    imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

    numberPipeline = Pipeline([
        ("numberFilter", GetNumbers()),
        ("imputer", imputer)
    ])

    textPipeline = Pipeline([
        ("textFilter", GetText()),
        ("vectoriser", MixedDict())
    ])

    transformPipeline = [
        ("feats", FeatureUnion([
            ("numberPipeline", numberPipeline),
            ("textPipeline", textPipeline)
        ])),
        ("scaler", StandardScaler()),
    ]

    if isClf(y):
        models, names = trainClf(X, y, projectName, transformPipeline)
    else:
        models, names = trainReg(X, y, projectName, transformPipeline)

    for i in range(len(models)):
        path = os.path.join("models", projectName, names[i] + ".sav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = open(path, "wb")

        pickle.dump(models[i], f)

    return

def stackedPredict(X, json_data, index):
    projectName = json_data["projectName"]
    model_dir = "models"
    basePath = os.path.join(model_dir, projectName)
    models = os.listdir(basePath)
    df = pd.DataFrame()

    skipName = "ensemble.sav"

    for model_name in models:
        if model_name != skipName:
            print ("95")
            path = os.path.join(basePath, model_name)

            model = pickle.load(open(path, "rb"))
            print (model_name.split(".")[0])

            y = model.predict(X)
            print ("done")
            df[model_name] = y

    path = os.path.join(basePath, skipName)
    model = pickle.load(open(path, "rb"))
    print ("alpha")
    y = model.predict(df)
    print ("beta")
    model_name = skipName
    output = pd.DataFrame(y, columns = [json_data["outputY"]], index = index.index)
    output[json_data["indexCol"]] = output.index
    output = output[[json_data["indexCol"], json_data["outputY"]]]
    writeCSV(json_data["outputFile"]+"_"+model_name+".csv", output, projectName)

    return

def stackedTrain(X, y, projectName):
    print ("\nTraining stacked...")
    model_dir = "models"
    basePath = os.path.join(model_dir, projectName)
    models = os.listdir(basePath)

    df = pd.DataFrame()#data=y, columns = ["y"])
    y = pd.DataFrame(data=y, columns = ["y"])
    skipName = "ensemble.sav"

    for model_name in models:
        print ("91")
        if model_name != skipName:
            print (model_name)
            path = os.path.join(basePath, model_name)

            model = pickle.load(open(path, "rb"))
            print (model_name.split(".")[0])

            y = model.predict(X)
            df[model_name] = y

    clf = RandomForestClassifier()
    print ("here!!!")
    print (df)
    print (y)
    clf.fit(df, y)

    path = os.path.join("models", projectName, skipName)
    f = open(path, "wb")
    pickle.dump(clf, f)
      
    return

################
# Transformers #
################
def isNumber(cType):
    if cType != np.float64 and cType != np.int64:
        return False
    return True

class GetText(BaseEstimator, TransformerMixin):
    def __init__(self):
        a = 1

    def transform(self, X, *_):
        for column in X.columns:
            cType = X[column].dtype
            if isNumber(cType):
                X = X.drop([column], axis = 1)
        return X

    def fit(self, X, *_):
        return self

class GetNumbers(BaseEstimator, TransformerMixin):
    def __init__(self):
        a = 1

    def transform(self, X, *_):
        for column in X.columns:
            cType = X[column].dtype
            if not isNumber(cType):
                X = X.drop([column], axis = 1)
        return X

    def fit(self, X, *_):
        return self

def text_process(text):
    text = str(text)
    text = [char for char in text if char not in string.punctuation]
    text = "".join(text)

    text = text.lower()
    
    text = [word for word in text.split()]# if word not in stopWords]
    return text

def textExtraction(df, series):
    vectorizer = CountVectorizer(analyzer = text_process, min_df = 0.1)
    df[series] = df[series].replace(np.nan, '', regex=True)
    vectorizer.fit_transform(df[series])
    vocab = vectorizer.get_feature_names()
    
    return vocab

class MixedDict(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocabDict = {}

    def transform(self, X, *_):
        for column in X.columns:
            if column in self.vocabDict:
                vectorizer = CountVectorizer(analyzer = text_process, vocabulary = self.vocabDict[column])
                if len(vectorizer.vocabulary) > 0:
                    vector = vectorizer.fit_transform(X[column])
                    i = 0
                    vector = vector.toarray()
                    for each in vector.T:
                        new_name = column + "_" + str(i)
                        X[new_name] = vector.T[i]
                        i = i + 1
                X = X.drop([column], axis = 1)

        return X

    def fit(self, X, *_):
        for column in X.columns:
            try: 
                vocab = textExtraction(X, column)
                self.vocabDict[column] = vocab
                #print ("- \"" + column + "\" has a vocabulary\n--\t"+ str(vocab))

            except:
                self.vocabDict[column] = []
                #print ("- \"" + column+ "\" does not have a vocabulary")

        return self

##############
# Predicting #
##############
def predict(X, json_data, index):
    print ("\nPredicting...")
    regressors = []
    model_dir = "models"
    basePath = os.path.join(model_dir, json_data["projectName"])
    models = os.listdir(basePath)

    skipName = "ensemble.sav"

    for model_name in models:
        if model_name != skipName:
        
            path = os.path.join(basePath, model_name)

            model = pickle.load(open(path, "rb"))
            print (model_name.split(".")[0])

            y = model.predict(X)
            #print ("bob")
            #print (index[0])
            #print (json_data["indexCol"])
            output = pd.DataFrame(y, columns = [json_data["outputY"]], index = index.index)
            output[json_data["indexCol"]] = output.index
            output = output[[json_data["indexCol"], json_data["outputY"]]]
            writeCSV(json_data["outputFile"]+"_"+model_name+".csv", output, json_data["projectName"])
    print ("DONE PREDICTING!")
    return

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

####################
# Train classifier #
####################
def trainClf(X, y, projectName, transformPipeline):
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
    # Bagging
    ####
    clf = BaggingClassifier()

    nEstimatorsArray = [10]

    parameters = [{"n_estimators": nEstimatorsArray}]

    names.append("Bagging")
    classifiers.append(clf)
    hyperParameters.append(parameters)

    ####
    # Random Forest (bagging+)
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
    # Gradient boosting
    ####
    clf = GradientBoostingClassifier()

    nEstimatorsArray = [10]

    parameters = [{"n_estimators": nEstimatorsArray}]

    names.append("Gradient boosting")
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
    pipelines = []
    for i in range(len(classifiers)):
        print ("\nTraining: " + str(classifiers[i]))

        # Get pipeline
        pipelineArray = transformPipeline[:]
        pipelineArray.append(("clf", classifiers[i]))
        pipeline = Pipeline(pipelineArray)

        kSplits = 2
        param_grid = {}
        for parameter in hyperParameters[i][0]:
            param_grid["clf__" + parameter]=hyperParameters[i][0][parameter]
        grid_search = GridSearchCV(pipeline, param_grid = param_grid, cv = kSplits)
        grid_search.fit(X, y)
        bestParameters = grid_search.best_params_

        pipeline.set_params(**bestParameters)

        pipeline.fit(X, y)
        pipelines.append(pipeline)

    return pipelines, names

####################
# Train regression #
####################
def trainReg(X, y, projectName, transformPipeline):
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

    return pipelines

################
# Read / Write #
################
def readCSV(projectName, fileName):
    print ("\nReading CSV...")
    path = os.path.join("raw", projectName, fileName)
    df = pd.read_csv(path)
    return df

def writeCSV(fileName, df, projectName):
    print ("\nWriting CSV...")
    path = os.path.join("output", projectName, fileName)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index = False, header = True)

##################
# Select columns #
##################
def getX(df, ignore):
    print ("\nSelect columns (X)...")
    # Exclude columns
    for column in ignore:
        if column in df:
            df = df.drop([column], axis = 1)

    return df

def getY(df, yColumn):
    print ("\nSelect columns (Y)...")
    y = df[yColumn]
    y = y.values.ravel()
    return y

