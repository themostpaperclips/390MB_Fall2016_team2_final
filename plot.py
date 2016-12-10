# -*- coding: utf-8 -*-
# %%---------------------------------------------------------------------------
#
#		                          Imports
#
# -----------------------------------------------------------------------------

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation

from util import slidingWindow, reorient, reset_vars
from features import *
import os
import scipy

# %%---------------------------------------------------------------------------
#
#		                      Initialization
#
# -----------------------------------------------------------------------------

plt.figure() # always call plt.figure() unless you want to plot points on an existing plot

# %%---------------------------------------------------------------------------
#
#		      Import Data from sample-data.csv
#
# -----------------------------------------------------------------------------

x = dict()
y = dict()
x[0.0] = []
x[1.0] = []
y[0.0] = []
y[1.0] = []
data_file = os.path.join('data', 'sample-data.csv')
data = np.genfromtxt(data_file, delimiter=',')

for i,window_with_timestamp_and_label in slidingWindow(data, 100):
    window = window_with_timestamp_and_label[:,1:-1]
    label = scipy.stats.mstats.mode(window_with_timestamp_and_label[:, 4])[0]
    # Add feature extraction here
    x[label[0]] += []
    y[label[0]] += []

stationary = plt.scatter(x[0], y[0], label='Stationary', color='red')
walking = plt.scatter(x[1], y[1], label='Walking', color='green')

# %%---------------------------------------------------------------------------
#
#		                   Format and Show Plot
#
# -----------------------------------------------------------------------------
plt.xlabel('Z entropy')
plt.ylabel('Z zero crossing rate')
plt.title('Z entropy and z zero crossing rate data samples for various activities')
plt.legend(handles=[stationary, walking])
plt.show()
# call plt.savefig(filename) to save to disk
