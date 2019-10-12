from __future__ import print_function
from six.moves import range
import random
import numpy as np

# We use mean squared error as our loss function


class MSE:
    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions, labels):
        diff = predictions - labels
        # By defining MSE as 0.5 times the square difference between predictions and labels
        return 0.5 * sum(diff * diff)[0]

    @staticmethod
    def loss_derivative(predictions, labels):
        #  the loss derivative is simply predictions - labels
        return predictions - labels


# In a sequential neural network we stack layers sequentially
class SequentialNetwork:
    def __init__(self, loss=None):
        print("Initializing network...")
        self.layers = []
        # If no loss is provided we use MSE
        if loss is None:
            self.loss = MSE()

    def add(self, layer):
        # Whenever we add a layer we connect it to its predecessor and and let it describe itself
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(training_data)
        # To train our network, we pass over data for as many times as there are epochs
        for epoch in range(epochs):
            random.shuffle(training_data)
            # We shuffle data and create mini batches
            mini_batches = [
                training_data[k:k + mini_batch_size] for
                k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                # for each mini batch we train our network
                self.train_batch(mini_batch, learning_rate)
            if test_data:
                n_test = len(test_data)
                # in case we provided test data, we evaluate our network on it after each epoch.
                print("Epoch {0}: {1} / {2}"
                      .format(epoch, self.evaluate(test_data), n_test))

    def train_batch(self, mini_batch, learning_rate):
        # To train the network on a mini-batch, we compute feed-forward and backward pass
        self.forward_backward(mini_batch)
        # and then update model parameters accordingly
        self.update(mini_batch, learning_rate)

    def update(self, mini_batch, learning_rate):
        # a common technique is to normailze the learning rate by the mini-batch size
        learning_rate = learning_rate / len(mini_batch)
        for layer in self.layers:
            # we then update parameters for all layers
            layer.update_params(learning_rate)
        for layer in self.layers:
            # afterwards we clear all deltas in each layer
            layer.clear_deltas()

    def forward_backward(self, mini_batch):
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:
                # for each sample in the mini batch feed the features forward layer by layer
                layer.forward()
            # we compute the loss derivative for the output data
            self.layers[-1].input_delta = \
                self.loss.loss_derivative(self.layers[-1].output_data, y)
            for layer in reversed(self.layers):
                # finally we do layer-by-layer backpropagation of error terms.
                layer.backward()

    def single_forward(self, x):
        # pass a single sample forward and return the result
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data):
        # compute accuracy on test data
        test_results = [(
            np.argmax(self.single_forward(x)),
            np.argmax(y)
        ) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)
