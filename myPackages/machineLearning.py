################
# Dependencies #
################
# Sci
import pandas as pd
import numpy as np
from scipy import stats

# General
import math
import os
import string
import pickle

# Workflow
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# Preprocessing
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Ensemble
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

# Support vector machines
from sklearn.svm import SVC

# Other classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import confusion_matrix

################
# Master train #
################
def getTransformPipeline():
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
    return transformPipeline

def train(X, y, projectName, scoring):
    print ("\nIdentifying type of problem...")

    #transformPipeline = getTransformPipeline()
    if isClf(y):
        models, names = trainClf(X, y, projectName, scoring)
    else:
        models, names = trainReg(X, y, projectName, scoring)

    for i in range(len(models)):
        # Save model
        path = os.path.join("models", projectName, names[i] + ".sav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = open(path, "wb")
        pickle.dump(models[i], f)

def stackedTrain(X, y, projectName, scoring):

    print ("\nTraining stacked...")
    model_dir = "models"
    basePath = os.path.join(model_dir, projectName)
    models = os.listdir(basePath)

    df = pd.DataFrame()
    #y = pd.DataFrame(data = y, columns = ["y"])
    skipName = "ensemble"

    for model_name in models:
        model_name_base = model_name.split(".")[0]
        suffix          = model_name.split(".")[1]
        if model_name_base != skipName and suffix == "sav":
            print ("\n" + model_name_base)
            
            path = os.path.join(basePath, model_name)
            model = pickle.load(open(path, "rb"))

            y_hat = model.predict(X)
            df[model_name_base] = y_hat

            tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
            
            n = tn + fp + fn + tp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / n
            f1 = stats.hmean([precision, recall])

            print (accuracy)
            
            path = os.path.join("models", projectName, model_name_base + ".txt")
            f = open(path, "w")
            
            f.write("N:\t\t" + str(n))
            f.write("\n\nTrue positive:\t" + str(tp) + "\t(" + str(tp/n) + ")")
            f.write("\nTrue negative:\t" + str(tn) + "\t(" + str(tn/n) + ")")
            f.write("\nFalse positive:\t" + str(fp) + "\t(" + str(fp/n) + ")")
            f.write("\nFalse negative:\t" + str(fn) + "\t(" + str(fn/n) + ")")
            
            f.write("\n\nAccuracy:\t" + str(accuracy))
            
            f.write("\n\nPrecision:\t" + str(precision))
            f.write("\nRecall:\t\t" + str(recall))
            f.write("\nF1:\t\t" + str(f1))
            
            f.close()

    kSplits = 2
    param_grid = {}
    model = RandomForestClassifier()            

    #transformPipeline = getTransformPipeline()
    #pipelineArray = transformPipeline[:]
    #pipelineArray.append(("clf", model))
    #pipeline = Pipeline(pipelineArray)
       
    grid_search = GridSearchCV(model, param_grid = param_grid, cv = kSplits, verbose = 2, scoring = scoring)
    grid_search.fit(df, y)
    bestParameters = grid_search.best_params_

    model.set_params(**bestParameters)

    model.fit(df, y)

    path = os.path.join("models", projectName, skipName + ".sav")
    f = open(path, "wb")
    pickle.dump(model, f)
    f.close()  
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
        suffix = model_name.split(".")[1]
        if model_name != skipName and suffix == "sav":
            print (model_name.split(".")[0])
            path = os.path.join(basePath, model_name)
            model_name_base = model_name.split(".")[0]
            model = pickle.load(open(path, "rb"))

            y = model.predict(X)
            output = pd.DataFrame(y, columns = [json_data["outputY"]], index = index.index)
            output[json_data["indexCol"]] = output.index
            output = output[[json_data["indexCol"], json_data["outputY"]]]
            writeCSV(json_data["outputFile"] + "_" + model_name_base + ".csv", output, json_data["projectName"])

    return

def stackedPredict(X, json_data, index):
    print ("Stacked predicting...")
    projectName = json_data["projectName"]
    model_dir = "models"
    csv_dir = "output"
    basePath = os.path.join(csv_dir, projectName)
    modelPath = os.path.join(model_dir, projectName)

    CSVs = os.listdir(basePath)
    df = pd.DataFrame()

    baseFileName = json_data["outputFile"].split(".")[0]

    skipName = baseFileName + "_ensemble.csv"

    for csv_name in CSVs:
        if csv_name != skipName:
            path = os.path.join(basePath, csv_name)
            csv = pd.read_csv(path)

            #print (csv)
            #y = model.predict(X)
            df[csv_name] = csv[json_data["outputY"]]

    model_name = "ensemble.sav"

    path = os.path.join(modelPath, model_name)
    model = pickle.load(open(path, "rb"))
    y = model.predict(df)

    output = pd.DataFrame(y, columns = [json_data["outputY"]], index = index.index)
    output[json_data["indexCol"]] = output.index
    output = output[[json_data["indexCol"], json_data["outputY"]]]
    writeCSV(skipName, output, projectName)

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
def trainClf(X, y, projectName, scoring):
    transformPipeline = getTransformPipeline()
    
    print ("Type: Classification")

    names = []
    classifiers = []
    hyperParameters = []

    ####
    # Decision tree 
    ####
    clf = DecisionTreeClassifier()

    maxDepthArray = [20]
    minSamplesSplitArray = [4]

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

    maxDepthArray = [20]
    minSamplesSplitArray = [2]

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
            param_grid["clf__" + parameter] = hyperParameters[i][0][parameter]

        grid_search = GridSearchCV(pipeline, param_grid = param_grid, cv = kSplits, verbose = 2, scoring = scoring)
        grid_search.fit(X, y)
        bestParameters = grid_search.best_params_

        pipeline.set_params(**bestParameters)

        pipeline.fit(X, y)
        pipelines.append(pipeline)

    return pipelines, names

####################
# Train regression #
####################
def trainReg(X, y, projectName):
    transformPipeline = transformPipeline()
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

