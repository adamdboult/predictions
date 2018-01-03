################
# Dependencies #
################
import myPackages.machineLearning as myPack

########
# Main #
########
def main():
    #############
    # Variables #
    #############
    projectName = "Titanic"

    trainFile = "train.csv"
    testFile = "test.csv"

    indexCol = "PassengerId"
    inputY =  "Survived"

    outputFile = "submission.csv"
    outputY = "Survived"

    #method = "SVM"

    #################
    # Training data #
    #################
    # Import
    df = myPack.readCSV(projectName, trainFile)

    # Clean
    vocab = myPack.getVocab(df)
    X = myPack.processX(df, inputY, vocab)
    y = myPack.processY(df, [inputY])
    
    # Train
    myPack.train(X, y)

    #############
    # Test data #
    #############
    # Import
    df = myPack.readCSV(projectName, testFile)
    
    # Clean
    X = myPack.processX(df, inputY, vocab)

    # Predict
    y = myPack.predict(X)

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

