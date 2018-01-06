################
# Dependencies #
################
import myPackages.machineLearning as myPack
import json
import sys

########
# Main #
########
def main():
    #############
    # Variables #
    #############
    json_path = sys.argv[1]
    json_file = open(json_path).read()
    json_data = json.loads(json_file)

    run_train = True
    run_test = False
    if len(sys.argv) > 2:
        test_train = sys.argv[2]
        if test_train == "train":
            run_test = False

        elif test_train == "test":
            run_train = False
    
    #################
    # Training data #
    #################
    # Import
    df = myPack.readCSV(json_data["projectName"], json_data["trainFile"])

    # Clean
    vocab = myPack.getVocab(df)
    X = myPack.processX(df, json_data["inputY"])
    y = myPack.processY(df, [json_data["inputY"]])
    
    # Train
    myPack.train(X, y)

    #############
    # Test data #
    #############
    # Import
    df = myPack.readCSV(json_data["projectName"], json_data["testFile"])
    
    # Clean
    X = myPack.processX(df, json_data["inputY"])

    # Predict
    y = myPack.predict(X)

    # Output    
    output = pd.DataFrame(y, columns = [json_data["outputY"]])

    if json_data["indexCol"] in df:
        output[json_data["indexCol"]] = df[json_data["indexCol"]]
    else:
        output.index += 1
        output[json_data["indexCol"]] = output.index
    output = output[[json_data["indexCol"], json_data["outputY"]]]

    myPack.writeCSV(json_data["outputFile"], output)
    
############
# Run main #
############
if __name__ == "__main__":
    main()

