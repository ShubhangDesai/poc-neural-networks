from __future__ import print_function
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

def sanitize(valid_labels, test_labels, train_dataset, valid_dataset, test_dataset):
    remove_test = 0
    remove_valid = 0

    for train_index, train_image in enumerate(train_dataset):
        print(train_index)

        loop_count = 0
        for index in np.argwhere(np.all(test_dataset == train_image, axis=(1, 2))):
            test_dataset = np.delete(test_dataset, index[0] - loop_count, axis=0)
            test_labels = np.delete(test_labels, index[0] - loop_count, axis=0)
            print('Removing image from test dataset')
            loop_count += 1
            remove_test += 1

        loop_count = 0
        for index in np.argwhere(np.all(valid_dataset == train_image, axis=(1, 2))):
            valid_dataset = np.delete(valid_dataset, index[0] - loop_count, axis=0)
            valid_labels = np.delete(valid_labels, index[0] - loop_count, axis=0)
            print('Removing image from validation dataset')
            loop_count += 1
            remove_valid += 1
    print(remove_test)
    print(remove_valid)
    return valid_labels, test_labels, valid_dataset, test_dataset

train_labels, valid_labels, test_labels, train_dataset, valid_dataset, test_dataset = unpickle('notMNIST.pickle')
print(np.shape(train_dataset))
print(np.shape(test_dataset))
print(np.shape(valid_dataset))
valid_labels, test_labels, valid_dataset, test_dataset = sanitize(valid_labels, test_labels, train_dataset, valid_dataset, test_dataset)
pickle_file = 'notMNIST_sanitized.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
  print('Pickling complete')
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

print('Finished sanitation')