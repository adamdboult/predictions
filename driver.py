################
# Dependencies #
################
import myPackages.machineLearning as myPack

import json
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

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

    projectName = json_data["projectName"]

    ignore = [json_data["indexCol"], json_data["inputY"]]

    ########################
    # What are we running? #
    ########################
    run_train = True
    run_test = True
    if len(sys.argv) > 2:
        test_train = sys.argv[2]
        if test_train == "train":
            run_test = False

        elif test_train == "test":
            run_train = False

    #########
    # Train #
    #########
    if run_train:
        # Import
        df = myPack.readCSV(projectName, json_data["trainFile"])

        # Select columns
        X = myPack.getX(df, ignore)
        y = myPack.getY(df, json_data["inputY"])

        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size = 0.25)

        # Train
        if "scoring" in json_data:
            scoring = json_data["scoring"]
        else:
            scoring = "accuracy"
        
        myPack.train(X_train, y_train, projectName, scoring)

        # Stacked ensemble
        myPack.stackedTrain(X_holdout, y_holdout, projectName, scoring)

    ###########
    # Predict #
    ###########
    if run_test:
        # Import
        df = myPack.readCSV(projectName, json_data["testFile"])
    
        # Select columns
        X = myPack.getX(df, ignore)

        if json_data["indexCol"] in df:
            index = pd.DataFrame(data=df[json_data["indexCol"]], columns = [json_data["indexCol"]])
            index.index = df[json_data["indexCol"]]
            index = index.drop([json_data["indexCol"]], axis = 1)
        else:
            index = pd.DataFrame(index=df.index)
            index.index += 1
        #print (index)
        # Predict
        myPack.predict(X, json_data, index)
        myPack.stackedPredict(X, json_data, index)
    
############
# Run main #
############
if __name__ == "__main__":
    main()

