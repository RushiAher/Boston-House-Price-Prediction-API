import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def preProcessing():
    df = datasets.load_boston()
    data = pd.DataFrame(df.data)
    data.columns = df.feature_names
    target = pd.DataFrame(df.target)
    target.columns = ['price']
    return (data, target)


def splitDataset(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)
    return (x_train, x_test, y_train, y_test)
    
def buildModel(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model
    
def makePrediction(test_data):
    data, target = preProcessing()
    x_train, x_test, y_train, y_test = splitDataset(data, target)
    trained_model = buildModel(x_train, y_train)
    # test_data = [[1.50234e+01, 0.00000e+00, 1.81000e+01, 0.00000e+00, 6.14000e-01,
    #     5.30400e+00, 9.73000e+01, 2.10070e+00, 2.40000e+01, 6.66000e+02,
    #     2.02000e+01, 3.49480e+02, 2.49100e+01]]
    result = trained_model.predict(test_data)
    return result
