import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pydataset import data
from scipy import stats



#--------------------------------------------------------------------------------------------------

def plot_rolling(train, val, yhat):

    plt.figure(figsize=(10,5))
    plt.plot(train[train.index.year >1880]['avg_temp'], label='Train')
    plt.plot(val.avg_temp, label='Validate')
    plt.plot(yhat.avg_temp, label='Predicted',  )
    plt.ylabel('Temperature (F)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

#--------------------------------------------------------------------------------------------------

def plot_test(train, val, test):

    plt.figure(figsize=(10,5))
    plt.plot(train[train.index.year >1880]['avg_temp'], label='Train')
    plt.plot(val.avg_temp, label='Validate')
    plt.plot(test_df.avg_temp, label='Test Rolling Avg')
    plt.ylabel('Temperature (F)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()