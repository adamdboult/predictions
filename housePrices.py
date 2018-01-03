################
# Dependencies #
################
import pandas as pd
import numpy as np

import myPackages.machineLearning as myPack

from sklearn.linear_model import LinearRegression

########
# Main #
########
def main():
    #############
    # Variables #
    #############
    projectName = "HousePrices"

    trainFile = "train.csv"
    testFile = "test.csv"
    outputFile = "submission.csv"
    
    indexCol = "Id"

    inputY =  "SalePrice"
    outputY = "SalePrice"

    cleanArr = []

    xColumns = []

    method = "MLP"

    #################
    # Training data #
    #################
    # Import
    df = myPack.readCSV(trainFile, projectName)

    # Clean
    vocab = myPack.getVocab(df)
    X = myPack.processX(df, inputY, vocab)
    y = myPack.processY(df, [inputY])

    y = y.values.ravel()

    # Train
    model = LinearRegression()

    fitInterceptArray = [True, False]
    normalizeArray = [True, False]

    parameters = [{"normalize":normalizeArray, "fit_intercept": fitInterceptArray}]

    bestParameters = myPack.crossValidate(X, y, model, parameters)

    model.set_params(**bestParameters)
    model.fit(X, y)

    score = model.score(X, y)

    #############
    # Test data #
    #############
    # Import
    df = myPack.readCSV(testFile, projectName)
    
    # Clean
    X = myPack.processX(df, inputY, vocab)

    # Predict
    y = model.predict(X)
    
    # Output    
    output = pd.DataFrame(y, columns = [outputY])

    if indexCol in df:
        output[indexCol] = df[indexCol]
    else:
        output.index += 1
        output[indexCol] = output.index
    output = output[[indexCol, outputY]]

    myPack.writeCSV(outputFile, output)
    
############
# Run main #
############
if __name__ == "__main__":
    main()

