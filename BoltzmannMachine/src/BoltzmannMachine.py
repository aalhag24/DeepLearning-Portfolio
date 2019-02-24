# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:49:54 2019

@author: Ahmed Alhag
"""

# Boltzmann Machine - Movie Recommender System
# Dataset from Grouplens 100k and 1m dataset

# Check that the gpu/cpu is being used properly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Import Default Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the Keras libraries and packages
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# List of file paths
BinPath = os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'bin'))
MovesData = BinPath + '/ml-1m/movies.dat'
UsersData = BinPath + '/ml-1m/users.dat'
RatingsData = BinPath + '/ml-1m/ratings.dat'

TrainingData = BinPath + '/ml-100k/u1.base'
TestingData = BinPath + '/ml-100k/u1.test'

# Import datasets
movies = pd.read_csv(MovesData, sep ='::', header = None, engine = 'python', encoding ='latin-1')
users = pd.read_csv(UsersData, sep ='::', header = None, engine = 'python', encoding ='latin-1')
ratings = pd.read_csv(RatingsData, sep ='::', header = None, engine = 'python', encoding ='latin-1')

# Preparing the training and test sets
training_set = pd.read_csv(TrainingData, delimiter='\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv(TestingData, delimiter='\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_c(self, x):
        wx torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_h = torch.sigmoid(activation)
        return p_h_given_h, torch.bernoulli(p_h_given_h)
        