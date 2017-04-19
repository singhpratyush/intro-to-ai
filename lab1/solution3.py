import random
import sys
import time

from matplotlib import pyplot

sys.path.append('.')

from lab1.pla import PLA
from lab1 import solution2


def main():
    data = solution2.load_iris_data(sys.argv[1])
    size = len(data)
    pyplot.show()
    points = {}
    points_i = {}
    runs = 10
    for _ in range(runs):
        for iterations in range(25+1):
            random.seed(time.time())
            random.shuffle(data)
            mid = int(size * 0.3)
            train_set = data[:mid]
            test_set = data[mid + 1:]
            pla = PLA(data[0].size - 1)
            pla = solution2.train_pla(pla, train_set, test_set,
                                      max_iterations=iterations)
            error = solution2.test_pla(pla, test_set)
            error_i = solution2.test_pla(pla, train_set)
            try:
                points[iterations] += error
            except KeyError:
                points[iterations] = error
            try:
                points_i[iterations] += error_i
            except KeyError:
                points_i[iterations] = error_i
            print('Run {0} | Iteration {1} | Error {2}'.format(_, iterations, error))
            del pla
    point_lot = []
    point_lot_i = []
    for i in points:
        points[i] /= runs
        points_i[i] /= runs
        point_lot.append((i, points[i]))
        point_lot_i.append((i, points_i[i]))
    pyplot.plot(*zip(*point_lot), label='E_out')
    pyplot.plot(*zip(*point_lot_i), label='E_in')
    pyplot.legend(loc='best')
    pyplot.title('E_in vs E_out - Average of {0} runs'.format(runs))
    pyplot.xlabel('Number of Iterations')
    pyplot.ylabel('Error')
    pyplot.grid(True)
    pyplot.pause(10000)


if __name__ == '__main__':
    main()
