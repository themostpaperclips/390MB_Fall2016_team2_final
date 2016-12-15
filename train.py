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

print("Loading data, windowing data, and extracting features...")
sys.stdout.flush()

window_size = 2000
n_features = 24

X = np.zeros((0,n_features))
y = np.zeros(0,)

rejected = 0

# For each of the directories in the data directory
for dirname in os.listdir('data'):

    # Extract each of the data sets
    magnetometerData = np.genfromtxt(os.path.join('data', dirname,'magnetometer_data.csv'), delimiter=',')
    barometerData = np.genfromtxt(os.path.join('data', dirname, 'barometer_data.csv'), delimiter=',')
    lightData = np.genfromtxt(os.path.join('data', dirname,'light_data.csv'), delimiter=',')

    # Combine the data into a dictionary
    data = {'magnetometer': magnetometerData, 'barometer': barometerData, 'light': lightData}

    # The data will equal false when extraction is done
    while data:

        # Create a window
        window = Window(window_size)

        # Update data so it doesn't have the data in the window
        data = window.push_slices(data)

        # If the window has enough data
        if (window.allCheck()):

            # Append the window features to X
            X = np.append(X, np.transpose(extract_features(window).reshape(-1, 1)), axis=0)

            # Append the window label to y
            y = np.append(y, [extract_labels(window)])
        else:
            rejected += 1
print("Rejected {} windows, {} percent of windows".format(rejected, (rejected / (rejected + len(X))) * 100))
print("Loaded data and extracted features over {} windows".format(len(X)))
print("Unique labels found: {}".format(list(set(y))))
sys.stdout.flush()

print("Training with 10-fold Cross Validation...")

class_names = ['indoors', 'outdoors']

n = len(y)
n_classes = len(class_names)

# Create the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3, max_features=5)

cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)

accuracyList = []
precisionList = []
recallList = []

# Train the classifier
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

# Save the classifier to disk
with open('classifier.pickle', 'wb') as f:
    pickle.dump(clf, f)
