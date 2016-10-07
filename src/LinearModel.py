from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

def unpickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            return data['train_labels'], data['valid_labels'], data['test_labels'], data['train_dataset'], data['valid_dataset'], data['test_dataset']
    except Exception as e:
        print(e)

def accuracy(predictions, sanitized=False):
    correct = 0
    labels = valid_labels
    if sanitized:
        labels = valid_labels_sanitized
    for index, prediction in enumerate(predictions):
        if prediction == labels[index]:
            correct += 1
    print(correct / len(predictions) * 100)

train_labels, valid_labels, test_labels, train_dataset, valid_dataset, test_dataset = unpickle('notMNIST.pickle')
_, valid_labels_sanitized, test_labels_sanitized, _, valid_dataset_sanitized, test_dataset_sanitized = unpickle('notMNIST_sanitized.pickle')

model_50, model_100, model_1000, model_5000, model_50_sanitized, model_100_sanitized, model_1000_sanitized, model_5000_sanitized = LogisticRegression(), LogisticRegression(), LogisticRegression(), LogisticRegression(), LogisticRegression(), LogisticRegression(), LogisticRegression(), LogisticRegression()

print('Fitting 50 sample model')
nsamples, nx, ny = train_dataset[0:50].shape
twod_train_dataset = train_dataset[0:50].reshape((nsamples,nx*ny))
model_50.fit(twod_train_dataset, train_labels[0:50])

print('Fitting 100 sample model')
nsamples, nx, ny = train_dataset[0:100].shape
twod_train_dataset = train_dataset[0:100].reshape((nsamples,nx*ny))
model_100.fit(twod_train_dataset, train_labels[0:100])

print('Fitting 1000 sample model')
nsamples, nx, ny = train_dataset[0:1000].shape
twod_train_dataset = train_dataset[0:1000].reshape((nsamples,nx*ny))
model_1000.fit(twod_train_dataset, train_labels[0:1000])

print('Fitting 5000 sample model')
nsamples, nx, ny = train_dataset[0:5000].shape
twod_train_dataset = train_dataset[0:5000].reshape((nsamples,nx*ny))
model_5000.fit(twod_train_dataset, train_labels[0:5000])

nsamples, nx, ny = valid_dataset.shape
twod_valid_dataset = valid_dataset.reshape((nsamples,nx*ny))

nsamples, nx, ny = valid_dataset_sanitized.shape
twod_valid_dataset_sanitized = valid_dataset_sanitized.reshape((nsamples,nx*ny))

print()
accuracy(model_50.predict(twod_valid_dataset))
accuracy(model_100.predict(twod_valid_dataset))
accuracy(model_1000.predict(twod_valid_dataset))
accuracy(model_5000.predict(twod_valid_dataset))
print()
accuracy(model_50.predict(twod_valid_dataset_sanitized), sanitized=True)
accuracy(model_100.predict(twod_valid_dataset_sanitized), sanitized=True)
accuracy(model_1000.predict(twod_valid_dataset_sanitized), sanitized=True)
accuracy(model_5000.predict(twod_valid_dataset_sanitized), sanitized=True)