from __future__ import print_function
import numpy as np


def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)


def sigmoid_prime_double(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)


class Layer(object):
    # Layers are stacked to build a sequential neural network
    def __init__(self):
        self.params = []

        # a layer knows its predecessor
        self.previous = None
        # and its successor
        self.next = None

        # each later can persist data flowing into and out of it in the forward pass
        self.input_data = None
        self.output_data = None

        # Analogously, a layer holds input and output data for the backward pass
        self.input_delta = None
        self.output_delta = None

    def connect(self, layer):
        # this method connects a leyer to its direct neighbours in the sequential network
        self.previous = layer
        layer.next = self

    def forward(self):
        # each layer implementation has to provide a function to feed input data forward
        raise NotImplementedError

    def get_forward_input(self):
        # input_data is reserved for the first layer, all opthers get their input from the previous output.
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):
        # layers have to implement backpropagation of error terms, that is a way to feed input errors backward theough the network
        raise NotImplementedError

    def get_backward_input(self):
        # input delta is reserved for the last layer, all other layers get their error terms from their successor
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta

    def clear_deltas(self):
        # we compute and accumulate deltas per mini-batch, after which we need to reset these deltas
        pass

    def update_params(self, learning_rate):
        # update layer parameters according to current deltas, using the specified learning_rate
        pass

    def descrie(self):
        # layer implementations can print their properties


class ActivationLayer(Layer):
    # this activation layer uses the sigmoid function to activate neurons
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        data = self.get_forward_input()
