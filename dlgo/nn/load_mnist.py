import six.moves.cPickle as cPickle
import gzip
import numpy as np


def encode_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# we one-hot encode indicies to vectors of length 10


def shape_data(data):
    # We flatten the imput images to feature vectors of length 784
    features = [np.reshape(x, (784, 1)) for x in data[0]]
    # All labels are one-hot encoded
    labels = [encode_label(y) for y in data[1]]
    # Then we create pairs of features and labels.
    return list(zip(features, labels))


def load_data_impl():
    # file retrieved by:
    #   wget https://s3.amazonaws.com/img-datasets/mnist.npz -O code/dlgo/nn/mnist.npz
    # code based on:
    #   site-packages/keras/datasets/mnist.py
    path = 'mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def load_data():
    # Unzipping and loading the MNIST data yields three data sets.
    train_data, test_data = load_data_impl()
    # We discard validation data here and reshape the other two data sets.
    return shape_data(train_data), shape_data(test_data)
