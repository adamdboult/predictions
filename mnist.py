################
# Dependencies #
################
import pandas as pd
import numpy as np

import myPackages.machineLearning as myPack

########
# Main #
########
def main():
    #############
    # Variables #
    #############
    projectName = "MNIST"

    trainFile = "train.csv"
    testFile = "test.csv"
    outputFile = "submission.csv"
    
    indexCol = "ImageId"

    inputY =  "label"
    outputY = "Label"

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

    # Train
    model = myPack.train(X, y, method)

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

