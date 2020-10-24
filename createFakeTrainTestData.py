# -*- coding: utf-8 -*-
"""
# test creating numpy arrays, stacking, and combining into tuples

# AIM IS TO CREATE: tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
"""

import numpy as np

numpix = 5
npos = 3
nneg = 4
ntest = 2

# creating data - positive instances
xpos = np.arange(npos*numpix*numpix*3).reshape(npos, 3, numpix, numpix)
ypos = np.ones(npos)

# creating data - negative instances
xneg = np.arange(nneg*numpix*numpix*3).reshape(nneg, 3, numpix, numpix)
yneg = np.zeros(nneg)

# create full data
x = np.vstack((xpos, xneg)) # stack using vstack - must have same shape for all but first axis
y = np.concatenate((ypos, yneg))
x.shape, y.shape

# split train and test - 70% train - by randomly picking indices
ntot = y.shape[0]
indices = np.random.permutation(ntot)
cutoffIndex = round(ntot*0.7)
training_idx, test_idx = indices[:cutoffIndex], indices[cutoffIndex:]
training, test = x[training_idx,:], x[test_idx,:]
x_train = x[training_idx, :,:,]
x_test = x[test_idx, :,:,]
y_train = y[training_idx]
y_test = y[test_idx]

# Check - first dimension should add to num across train + test
x_train.shape, y_train.shape
x_test.shape, y_test.shape

(x_train, y_train), (x_test, y_test)




