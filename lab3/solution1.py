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
    plot_data, _ = calculate_plot_data(data)
    pyplot.plot(*zip(*plot_data))
    pyplot.xlabel('Training Data Size')
    pyplot.ylabel('Error')
    pyplot.title('Training Data vs. Error for Single Perceptron Linear Regression')
    pyplot.grid(True)
    pyplot.show()


if __name__ == '__main__':
    main()
