import argparse
import json
import pandas as pd
import time
import numpy as np
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


def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
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

    grid_search = GridSearchCV(clf, pgrid, cv = 5) 
    grid_search.fit(xTrain, yTrain) 
    bestParams = grid_search.best_params_

    final_clf = grid_search.best_estimator_
    reg = final_clf.fit(xTrain, yTrain)
    ypred = final_clf.predict(xTest)

    # new_ypred = []
    # for i in ypred:
    #     new_ypred.append(i)
    
    # for i in range(len(new_ypred)):
    #     if new_ypred[i] >= 0.5:
    #         new_ypred[i] = int(1)
    #     else:
    #         new_ypred[i] = int(0)
    # new_ypred = np.array(new_ypred)

    #fpr, tpr, thresholds = metrics.roc_curve(yTest, ypred)
    roc["fpr"] = 1
    roc["tpr"] = 2

    time_lr = time.time() - start
    resultDict = {'Accuracy': accuracy_score(ypred, yTest),
        'Best Parameters': bestParams,
        'Time': time_lr}
    # resultDict = {
    #     'AUPRC': average_precision_score(yTest, ypred),
    #     'F1': f1_score(yTest, ypred),
    #     'Time': time_lr}
    
    return resultDict, roc, bestParams



def main():

    bruh = pd.read_csv("first_1000_samples.csv")
    bruh = bruh.dropna()
    soft_skills = bruh["job_skills"]
    job_types = bruh["job_title"]
    label_encoder = LabelEncoder()
    job_types= label_encoder.fit_transform(job_types)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(soft_skills)
    xTrain, xTest, yTrain, yTest = train_test_split(X, job_types, test_size=0.3)

    #K Nearest Neighbors
    parameters = {}
    parameters["n_neighbors"] = [5,10,50,100,200]

    knnClf = KNeighborsClassifier()
    #perfDict, rocDF, bestParamDict = eval_gridsearch(knnClf, parameters, xTrain, yTrain, xTest, yTest)
    print("KNN")
    print(perfDict)
    print(rocDF)
    print(bestParamDict)

    #Decision Tree
    parameters = {}
    parameters["criterion"] = ["gini", "entropy"]
    parameters["max_depth"] = [1,5,10]
    parameters["min_samples_leaf"] = [5,10,15]

    dtclf = DecisionTreeClassifier()
    perfDict, rocDF, bestParamDict = eval_gridsearch(dtclf, parameters, xTrain, yTrain, xTest, yTest)
    print("Decision Tree")
    print(perfDict)
    print(rocDF)
    print(bestParamDict)

    #Neural Networks
    parameters = {}
    parameters["activation"] = ['identity', 'logistic', 'tanh']
    parameters["max_iter"] = [100,200,300]

    mlpclf = MLPClassifier(max_iter=10)
    #perfDict, rocDF, bestParamDict = eval_gridsearch(mlpclf, parameters, xTrain, yTrain, xTest, yTest)
    print("Neural Networks")
    print(perfDict)
    print(rocDF)
    print(bestParamDict)



                            

if __name__ == "__main__":
    main()