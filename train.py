# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import numpy as np
from extract_features import extract_features, extract_labels
from window import Window
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pickle

from sklearn.tree import DecisionTreeClassifier

# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
magnetometer_file = os.path.join('data', 'magnetometer_data.csv')
magnetometerData = np.genfromtxt(magnetometer_file, delimiter=',')
barometer_file = os.path.join('data', 'barometer_data.csv')
barometerData = np.genfromtxt(barometer_file, delimiter=',')
light_file = os.path.join('data', 'light_data.csv')
lightData = np.genfromtxt(light_file, delimiter=',')
data = {'magnetometer': magnetometerData, 'barometer': barometerData, 'light': lightData}
print("Loaded data")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 2000

print("Extracting features and labels for window size {}".format(window_size))
sys.stdout.flush()


# TODO make it so this isn't manual
n_features = 24

X = np.zeros((0,n_features))
y = np.zeros(0,)

total = 0

while(sum(map(lambda x: len(x), data.values())) != 0):

    window = Window(window_size)

    data = window.push_slices(data)
    if (window.allCheck()):

        X = np.append(X, np.transpose(extract_features(window).reshape(-1, 1)), axis=0)
        # append label:
        y = np.append(y, [extract_labels(window)])
    else:
        total += 1

print total
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

class_names = ['indoors', 'outdoors']

n = len(y)
n_classes = len(class_names)

clf = DecisionTreeClassifier(max_depth=3, max_features=5)

cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)

accuracyList = []
precisionList = []
recallList = []

for i, (train_indexes, test_indexes) in enumerate(cv):
    print("Fold {}".format(i))

    clf.fit(X[train_indexes], y[train_indexes])
    conf = confusion_matrix(clf.predict(X[test_indexes]), y[test_indexes], labels=range(0, n_classes))

    accuracy = sum(sum(np.multiply(conf, np.eye(n_classes)))) / sum(sum(conf))
    accuracyList += [accuracy]

    precision = [conf[i, i] / sum(conf[:, i]) for i in range(0, n_classes)]
    precisionList += [precision]

    recall = [conf[i, i] / sum(conf[i, :]) for i in range(0, n_classes)]
    recallList += [recall]

print "average accuracy:"
print np.nanmean(accuracyList)

print "average precision:"
print np.nanmean(precisionList, axis=0)

print "average recall:"
print np.nanmean(recallList, axis=0)

# when ready, set this to the best model you found, trained on all the data:
best_classifier = clf
with open('classifier.pickle', 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)
