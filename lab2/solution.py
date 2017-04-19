import json
import sys
import numpy

from matplotlib import pyplot

sys.path.append('.')

from lab2.load_wine_data import load_wine_data


def get_train_test(dataset):
    return dataset[:, :-1], dataset[:, -1]


def get_weights(train_set):
    train_x, train_y = get_train_test(train_set)
    return numpy.linalg.pinv(train_x) * train_y


def calculate_error(weights, test_set):
    test_x, test_y = get_train_test(test_set)
    y_actual = test_x * weights
    return numpy.linalg.norm(test_y - y_actual)**2/len(test_set)


def calculate_plot_data(data):
    plot_data = []
    weight_array = []
    # test_set = data[4000:]
    test_set = data[4501:]
    # for partition in range(500, 4001, 500):
    for partition in range(500, 4501, 1000):
        mid = partition
        train_set = data[:mid]
        weights = get_weights(train_set)
        e_out = calculate_error(weights, test_set)
        plot_data.append((partition, e_out))  # math.log(e_out)))
        weight_array.append(weights)
    return plot_data, weight_array


def non_linear_transform(data, degree):
    x, y = get_train_test(data)
    new = [i.T for i in x.T]
    for i in range(degree - 1):
        shape = len(new)
        newer = []
        for j in range(shape):
            for k in range(j, shape):
                newer.append(numpy.multiply(new[j], new[k]))
        new = newer
    st = numpy.stack([x.T for x in new] + [y.T,]).T
    return st


def main():
    data = load_wine_data(sys.argv[1])
    numpy.random.seed(1234)  # int(time.time()))
    numpy.random.shuffle(data)
    for i in range(1, 3):
        transformed_data = non_linear_transform(data, i)
        plot_data, _ = calculate_plot_data(transformed_data)
        pyplot.plot(*zip(*plot_data), label='Nonlinear Transform of Degree {0}'.format(i))
        with open('plot_data{0}.json'.format(i), 'w') as f:
            json.dump(plot_data, f)
    pyplot.legend(loc='best')
    pyplot.xlabel('Train Partition Size')
    pyplot.ylabel('Error on Test Partition')
    pyplot.title('Partition Size vs. Error for Single Perceptron Regression')
    pyplot.grid(True, which='both')
    pyplot.show()


if __name__ == '__main__':
    main()
