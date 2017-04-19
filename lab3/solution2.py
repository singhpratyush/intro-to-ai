import sys

from matplotlib import pyplot

sys.path.append('.')

from lab2.load_wine_data import load_wine_data
from lab2.solution import calculate_plot_data
from lab1.pla import PLA
from lab1.solution2 import test_pla, train_pla


def main():
    data = load_wine_data(sys.argv[1])
    data[:, -1] = data[:, -1] * 2 - 1
    plot_data, weight_array = calculate_plot_data(data)

    # Perform Classification
    pla = PLA(len(data[0]) - 1)
    classification_error_o = []
    classification_error_i = []
    test_data = data[4501:]

    runs = 50
    for iteration in range(runs):
        index = 0
        print('Iterations: {0}'.format(iteration))
        for i in range(500, 4501, 1000):
            train_data = data[:i]
            pla.weights = weight_array[index].T  # weight_array is calculated weight using regression
            pla = train_pla(pla, train_data, test_data, iteration)
            error_o = test_pla(pla, test_data)
            classification_error_o.append((i, error_o))

            error_i = test_pla(pla, train_data)
            classification_error_i.append((i, error_i))

            tr_len = len(train_data)
            te_len = len(test_data)
            print(
                'Training Misses: {1}/{2} | Testing Misses: {3}/{4}'.format(
                    i, round(error_i * tr_len), tr_len, round(error_o * te_len),
                    te_len))

            index += 1

    memory_i = {}
    for i, j in classification_error_i:
        if i in memory_i:
            memory_i[i] = min([memory_i[i], j])
        else:
            memory_i[i] = j
    memory_o = {}
    for i, j in classification_error_o:
        if i in memory_o:
            memory_o[i] = min([memory_o[i], j])
        else:
            memory_o[i] = j

    plot_i = sorted([(i, memory_i[i]) for i in memory_i])
    plot_o = sorted([(i, memory_o[i]) for i in memory_o])

    pyplot.plot(*zip(*plot_i), label='E-in')
    pyplot.plot(*zip(*plot_o), label='E-out')
    pyplot.legend(loc='best')
    pyplot.xlabel('Train Partition Size')
    pyplot.ylabel('Error on Test Partition')
    pyplot.title('Partition Size vs. Error for Single Perceptron Regression weights Applied to Classification with Iterations and Pocket Algorithm ({0} runs)'.format(runs))
    pyplot.xticks(range(0, 5001, 500))
    pyplot.grid(True, which='both')
    pyplot.show()


if __name__ == '__main__':
    main()
