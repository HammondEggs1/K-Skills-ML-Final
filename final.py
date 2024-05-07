import argparse
import json
import pandas as pd
import time
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
from helperfunctions import saveClf, loadClf
from sklearn.naive_bayes import GaussianNB



def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest, target, fileName = -1):
    """
    Given a sklearn classifier and a parameter grid to search,
    choose the optimal parameters from pgrid using Grid Search CV
    and train the model using the training dataset and evaluate the
    performance on the test dataset.

    Parameters
    ----------
    clf : sklearn.ClassifierMixin
        The sklearn classifier model 
    pgrid : dict
        The dictionary of parameters to tune for in the model
    xTrain : nd-array with shape (n, d)
        Training data
    yTrain : 1d array with shape (n, )
        Array of labels associated with training data
    xTest : nd-array with shape (m, d)
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.
    target: the integer value associated with the job title
        that roc scores are measured based on
    fileName: the name of the file that you want the classifier
    to be saved to

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 4 keys,
        "AUC", "AUPRC", "F1", "Time" and the values are the floats
        associated with them for the test set.
    roc : dict
        A Python dictionary with 2 keys, fpr, and tpr, where
        each of the values are 1-d numpy arrays of the fpr and tpr
        associated with different thresholds. You should be able to use 
        this to plot the ROC for the model performance on the test curve.
    bestParams: dict
        A Python dictionary with the best parameters chosen by your
        GridSearch. The values in the parameters should be something
        that was in the original pgrid.
    """
    start = time.time()
    roc = {}
    bestParams = {}

    grid_search = GridSearchCV(clf, pgrid, cv = 3, verbose=10, n_jobs=-1)
    print("Begin GS")
    grid_search.fit(xTrain, yTrain) 
    bestParams = grid_search.best_params_
    final_clf = grid_search.best_estimator_
    reg = final_clf.fit(xTrain, yTrain)
    ypred = final_clf.predict(xTest)

    # The main idea of the OvM fpr and tpr is that the target
    # predictions are turned to 1 while all others are turned
    # 0, allowing metrics.roc_curve to interpret the multi
    # class classification as though it were binary
    roc_yTest = [1 if x == target else 0 for x in yTest]
    roc_ypred = [1 if x == target else 0 for x in ypred]
    fpr, tpr, thresholds = metrics.roc_curve(roc_yTest, roc_ypred)
    roc["fpr"] = fpr
    roc["tpr"] = tpr

    time_lr = time.time() - start
    resultDict = {'Accuracy': accuracy_score(ypred, yTest),
        'Time': time_lr}
    # resultDict = {
    #     'AUPRC': average_precision_score(yTest, ypred),
    #     'F1': f1_score(yTest, ypred),
    #     'Time': time_lr}
    
    if(fileName!=-1):
        saveClf(reg, fileName)
    return resultDict, roc, bestParams

def eval_model(clf, xTrain, yTrain, xTest, yTest, target, fileName = -1):
    start = time.time()
    roc = {}
    bestParams = {}
    clf.fit(xTrain, yTrain)
    ypred = clf.predict(xTest)
    bin_yTest = [1 if x == target else 0 for x in yTest]
    bin_ypred = [1 if x == target else 0 for x in ypred]
    fpr, tpr, thresholds = metrics.roc_curve(bin_yTest, bin_ypred)
    roc["fpr"] = fpr
    roc["tpr"] = tpr

    if(fileName!=-1):
        saveClf(clf, fileName)

    time_lr = time.time() - start
    resultDict = {
         'AUPRC': average_precision_score(bin_yTest, bin_ypred),
         'F1': f1_score(bin_yTest, bin_ypred),
         'Time': time_lr}
    
    return resultDict, roc, bestParams

def main():

    bruh = pd.read_csv("further_processed_dataset.csv")
    bruh = bruh.dropna()
    soft_skills = bruh["job_skills"]
    job_types = bruh["job_title"]
    label_encoder = LabelEncoder()
    job_types= label_encoder.fit_transform(job_types)
    # In order to later decode back to jobtitles,
    # the label encoder must be saved
    saveClf(label_encoder, "label_encoder_fpd.joblib")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(soft_skills)
    vectorizer = vectorizer.fit(soft_skills)
    saveClf(vectorizer, "vectorizer_fpd.joblib")
    xTrain, xTest, yTrain, yTest = train_test_split(X, job_types, test_size=0.3)
    
    #K Nearest Neighbors

    parameters = {}
    parameters["n_neighbors"] = [5,10,50,100,200]

    # For grid searching
    # knnClf = KNeighborsClassifier()
    knnClf = KNeighborsClassifier(n_neighbors=5)
    # perfDict, rocDF, bestParamDict = eval_gridsearch(knnClf, parameters, xTrain, yTrain, xTest, yTest, 0)
    perfDict, rocDF, bestParamDict = eval_model(knnClf, xTrain, yTrain, xTest, yTest, 0, "KNN_fpd.joblib")
    print("KNN")
    print(perfDict)
    print(rocDF)
    print(bestParamDict)
   
    # search grid
    params_grid = {
        'learning_rate': [0.1, 0.25],
        'max_depth':[2, 3],
        "n_estimators": [50, 75, 100]
    }

    classes = max(job_types)+1
    # for grid searching
    # xgclf = xgb.XGBRegressor(objective='multi:softmax', num_class=classes)
    xgclf = xgb.XGBRegressor(objective='multi:softmax', n_estimators=100, max_depth=3, num_class=classes)
    setattr(xgclf, 'verbosity', 2)
    #perfDict, rocDF, bestParamDict = eval_gridsearch(xgclf, params_grid, xTrain, yTrain, xTest, yTest, 0, "XGB_fpd.joblib")
    perfDict, rocDF, bestParamDict = eval_model(xgclf, xTrain, yTrain, xTest, yTest, 0, "XGB_fpd.joblib")
    print(perfDict)
    print(rocDF)
    print(bestParamDict)


    # #Decision Tree
    """
    parameters = {}
    parameters["criterion"] = ["gini", "entropy"]
    parameters["max_depth"] = [1,5,10]
    parameters["min_samples_leaf"] = [5,10,15]

    dtclf = DecisionTreeClassifier()
    perfDict, rocDF, bestParamDict = eval_gridsearch(dtclf, parameters, xTrain, yTrain, xTest, yTest, 0, "dt_fpd.joblib")
    print("Decision Tree")
    print(perfDict)
    print(rocDF)
    print(bestParamDict)
    """

    # #Neural Networks
    # parameters = {}
    # parameters["activation"] = ['identity']
    # parameters["max_iter"] = [10]

    # mlpclf = MLPClassifier(max_iter=10)
    # perfDict, rocDF, bestParamDict = eval_gridsearch(mlpclf, parameters, xTrain, yTrain, xTest, yTest, 54)
    # print("Neural Networks")
    # print(perfDict)
    # print(rocDF)
    # print(bestParamDict)


    #Naive Bayes
    """
    parameters = {}
    parameters['var_smoothing'] = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    gnbclf = GaussianNB()
    perfDict, rocDF, bestParamDict = eval_gridsearch(gnbclf, parameters, xTrain, yTrain, xTest, yTest, 0, "nb_bot50.joblib")
    print("Naive Bayes")
    print(perfDict)
    print(rocDF)
    print(bestParamDict)
    """


    
   
    #We changed the labels depending on which method we were testing
    plt.plot(rocDF["fpr"], rocDF["tpr"], label="KNN")
    plt.legend(loc=0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.show()
    

if __name__ == "__main__":
    main()
