from matplotlib import pyplot as plt
import numpy as np
from dlgo.nn.load_mnist import load_data
from dlgo.nn.layers import sigmoid_double


def average_digit(data, digit):
    # We compute the average over all samples in our data representing a given digit.
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


train, test = load_data()
# We use the average eight as parameters for a simple model to detect eights.
avg_eight = average_digit(train, 8)


img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()

# training sample at index 2 is a "4"
x_3 = train[2][0]
# training sample at index 17 is a"8
x_18 = train[17][0]

W = np.transpose(avg_eight)
# this evaluates to about 20.1
np.dot(W, x_3)
# this term is mich bigger, about 54.2
np.dot(W, x_18)


def predict(x, W, b):
    # A simple prediction is defined by applying sigmoid to the output of np.doc(W, x) + b.
    return sigmoid_double(np.dot(W, x) + b)


# Based on the examples computed so far we set the bias term to -45.
b = -45

# The prediction for the example with a "4" is close to zero
print(predict(x_3, W, b))
# The prediction for an '8' is 0.96 here. We seem to be onto something with our heuristic.
print(predict(x_18, W, b))


def evaluate(data, digit, threshold, W, b):
    # as evaluating metric we choose accuracy, the ratio of correct predictions among all.
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        # predicting an instance of an eight as "8" is a correct prediction.
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:
            correct_predictions += 1
        # if the prediction is below our threshold and the sample is not an "8", we also predicted correctly.
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:
            correct_predictions += 1
    return correct_predictions / total_samples


# accuracy on training data of our simple model is 78%
evaluate(data=train, digit=8, threshold=0.5, W=W, b=b)
# accuracy on test is sloghtly lower, at 77%
evaluate(data=test, digit=8, threshold=0.5, W=W, b=b)

eight_test = [x for x in test if np.argmax(x[1]) == 8]
# Evaluating only on the set of eights in the test set only results in 67% accuracy
evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b)
