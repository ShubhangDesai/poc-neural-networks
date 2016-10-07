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
import tensorflow as tf

def unpickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            return data['train_labels'], data['valid_labels'], data['test_labels'], data['train_dataset'], data['valid_dataset'], data['test_dataset']
    except Exception as e:
        print(e)

train_labels, valid_labels, test_labels, train_dataset, valid_dataset, test_dataset = unpickle('notMNIST.pickle')

image_size = 28
num_labels = 10
hidden_1 = 500
hidden_2 = 300
hidden_3 = 100

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.

  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_1]))
  weights_2 = tf.Variable(tf.truncated_normal([hidden_1, hidden_2]))
  weights_3 = tf.Variable(tf.truncated_normal([hidden_2, hidden_3]))
  weights_4 = tf.Variable(tf.truncated_normal([hidden_3, num_labels]))
  biases_1 = tf.Variable(tf.zeros([hidden_1]))
  biases_2 = tf.Variable(tf.zeros([hidden_2]))
  biases_3 = tf.Variable(tf.zeros([hidden_3]))
  biases_4 = tf.Variable(tf.zeros([num_labels]))
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.5, global_step, 500, -15)

  # Training computation.
  layer_1 = tf.nn.tanh(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights_2) + biases_2)
  layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights_3) + biases_3)
  logits = tf.matmul(layer_3, weights_4) + biases_4
  ##logits = tf.nn.relu(tf.matmul(tf_train_dataset, weights) + biases)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.matmul(tf_valid_dataset, weights_1) + biases_1, weights_2) + biases_2)
  test_prediction = tf.nn.softmax(tf.matmul(tf.matmul(tf.matmul(tf.matmul(tf_test_dataset, weights_1) + biases_1, weights_2) + biases_2, weights_3) + biases_3, weights_4) + biases_4)

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  l_previous = 0
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      ##print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    if (step > 2500) and (l - l_previous > 0.2):
      print(step)
      print(l_previous)
      print(l)
      break
    l_previous = l
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))