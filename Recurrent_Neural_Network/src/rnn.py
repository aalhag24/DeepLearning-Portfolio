# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:39:44 2019

@author: Ahmed Alhag
"""

# Recurrent Neural Network

# Import Default Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import Keras libraries Material

# Check that the gpu/cpu is being used properly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Import datasets
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
TrainingData = BinPath + '/Google_Stock_Price_Test.csv';
TestingData = BinPath + '/Google_Stock_Price_Train.csv';

dataset_train = pd.read_csv(TrainingData)
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)