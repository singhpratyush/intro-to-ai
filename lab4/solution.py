import sys
import pickle

sys.path.append('.')

from lab4.logistic_regression import LogisticRegression
from lab2.load_wine_data import load_wine_data

iterations = 1000


def regression_plot_data(train_x, train_y, test_x, test_y, max_iterations,
                         hard=True):
    log_regressor = LogisticRegression(train_x[0].shape)
    plot_data = []
    for j in range(1, max_iterations + 1):
        for i in range(len(train_x)):
            log_regressor.train_old(train_x[i], train_y[i])
        plot_data.append((j, log_regressor.calculate_error(test_x, test_y,
                                                           hard=hard)))
    return plot_data


def regression_plot_data_new(train_x, train_y, test_x, test_y):
    log_regressor = LogisticRegression(train_x[0].shape)
    plot_data = []
    iteration = 0
    while log_regressor.train(train_x, train_y, stop_delta=0.1):
        iteration += 1
        plot_data.append((iteration, log_regressor.calculate_error(test_x, test_y)))
    return plot_data


def main():
    data = load_wine_data(sys.argv[1])
    train_data = data[:4500]
    test_data = data[4501:]

    train_x, train_y = train_data[:, :-1], train_data[:, -1]
    test_x, test_y = test_data[:, :-1], test_data[:, -1]

    data_size_plot = {}
    for i in range(500, 4501, 2000):
        # plot_data = regression_plot_data(train_x[:i], train_y[:i], test_x,
        #                                  test_y, iterations)
        plot_data = regression_plot_data_new(train_x[:i], train_y[:i], test_x,
                                         test_y)
        data_size_plot[i] = plot_data

    with open(sys.argv[2], 'wb') as f:
        pickle.dump(data_size_plot, f)


if __name__ == '__main__':
    main()
