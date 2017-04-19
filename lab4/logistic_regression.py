import numpy


class LogisticRegression:

    def __init__(self, shape, learning_rate=0.0005):
        self.weights = (numpy.random.random(shape) - 0.5) * 0.001
        self.learning_rate = learning_rate

    def test(self, data_point):
        z = self.weights * data_point.T
        sigma = LogisticRegression.sigmoid(z)
        try:
            return sigma.T
        except:
            return sigma

    def train_old(self, x, y):
        actual_value = self.test(x)
        update = (y - actual_value) * actual_value * (1 - actual_value) * x
        self.weights += self.learning_rate * update

    def train(self, x, y, stop_delta=0.1):
        y_ = self.test(x)
        update = self.learning_rate * (y - y_).T * x
        change_norm = numpy.linalg.norm(update)
        self.weights += self.learning_rate * update
        return change_norm > stop_delta

    def calculate_error(self, x, y, hard=False):
        actual_value = self.test(x)
        if not hard:
            actual_class = numpy.where(actual_value > 0.5, 1, 0)
        else:
            actual_class = actual_value
        error = numpy.absolute(actual_class - y)
        ret = sum(error)[0, 0]/float(len(y))
        return ret

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + numpy.exp(-x))
