import time
import numpy
import random


class PLA(object):

    def __init__(self, num_weights, seed=None):
        if seed is None:
            seed = int(time.time())
        numpy.random.seed(seed)
        self.num_weights = num_weights
        # self.weights = numpy.asmatrix(numpy.zeros(self.num_weights)).reshape([1, self.num_weights])
        self.weights = numpy.asmatrix(numpy.random.rand(
            self.num_weights).reshape([1, self.num_weights]))
        self.bias = random.uniform(0, 1)

    def __str__(self):
        return '<PLA w0={1} w={0}>'.format(self.weights, self.bias)

    def test(self, inp):
        sop = self.weights * inp.T  # + self.bias
        return -1 if sop < 0 else 1

    def train(self, inp, expected_output):
        actual_output = self.test(inp)
        if actual_output == expected_output:
            return True
        update = expected_output * inp
        self.weights += update
        return False
