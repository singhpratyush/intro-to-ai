import random
import sys
import time
import numpy

from matplotlib import pyplot

from lab1.pla import PLA

train_plot_data_start = []
train_plot_data_end = []
test_plot_data_start = []
test_plot_data_end = []


def plot():
    plot_data = zip(*train_plot_data_start)
    # p1 = pyplot.plot(*plot_data, c='r', label='Error in training set before iterations')
    plot_data = zip(*train_plot_data_end)
    p2 = pyplot.plot(*plot_data, c='g', label='Error in training set after iterations')
    plot_data = zip(*test_plot_data_start)
    # p3 = pyplot.plot(*plot_data, c='b', label='Error in testing set before iterations')
    plot_data = zip(*test_plot_data_end)
    p4 = pyplot.plot(*plot_data, c='y', label='Error in testing set after iterations')
    pyplot.title('Error vs Iterations for {0} partition'.format(input()))
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Error')
    pyplot.legend(loc='best')
    pyplot.grid(True)
    pyplot.show()
    pyplot.clf()


def train_pla(pla, train_set, test_set, max_iterations=float('inf')):
    complete = False
    iterations = 0
    # while not complete and iterations < max_iterations:
    while iterations < max_iterations:
        iterations += 1
        complete = True

        # Calculate initial errors
        test_err = test_pla(pla, test_set)
        train_err = test_pla(pla, train_set)

        # Add to plot data
        train_plot_data_start.append((iterations, train_err))
        test_plot_data_start.append((iterations, test_err))

        # Print to console
        # print('Iteration {0} start. Training Error {1}. Testing Error {2}'.format(
        #     iterations,
        #     train_err,
        #     test_err
        # ))
        for data in train_set:
            complete = complete and pla.train(data[0][0, 0:-1], data[0][0, -1])

        # Calculate errors after iteration
        test_err = test_pla(pla, test_set)
        train_err = test_pla(pla, train_set)

        # Add to plot data
        train_plot_data_end.append((iterations, train_err))
        test_plot_data_end.append((iterations, test_err))

        # Print to console
    #     print('Iteration {0} end. Training Error {1}. Testing Error {2}\n'.format(
    #         iterations,
    #         train_err,
    #         test_err
    #     ))
    # print(iterations, pla)
    return pla


def test_pla(pla, test_set):
    errors = 0
    size = len(test_set)
    for data in test_set:
        inp = data[0][0,0:-1]
        out = data[0][0,-1]
        act_out = pla.test(inp)
        if act_out != out:
            errors += 1
    return errors / float(size)


def load_iris_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = map(numpy.array, [list(map(float, line.replace('\r\n', '').split(
        '\t'))) for line in lines])
    data = [numpy.asmatrix(i.reshape([1, i.size])) for i in data]
    return data


def main():
    filename = sys.argv[1]
    try:
        iterations = int(sys.argv[2])
        print('Iterations - {0}'.format(iterations))
    except (IndexError, ValueError):
        iterations = None

    fractions = [0.3, 0.4, 0.5, 0.6, 0.7]
    data = load_iris_data(filename)
    random.seed(1234)
    random.shuffle(data)

    data_size = len(data)
    for fraction in fractions:
        fraction = 0.3
        mid_index = int(data_size * fraction)
        train_data = data[:mid_index]
        test_data = data[mid_index+1:]
        pla = PLA(data[0].size - 1)
        print(pla)
        if iterations is None:
            pla = train_pla(pla, train_data, test_data)
        else:
            pla = train_pla(pla, train_data, test_data, iterations)
        e_in = test_pla(pla, train_data)
        e_out = test_pla(pla, test_data)
        print('Partition {0} | E_in = {1} | E_out = {2}'.format(fraction, e_in,
                                                                e_out))
        plot()
        break


if __name__ == '__main__':
    main()
