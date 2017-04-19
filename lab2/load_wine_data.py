import numpy


def load_wine_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    arr = [list(map(float, ['1',] + x.replace('\n', '').split(';')))
           for x in lines[1:]]
    data = map(numpy.asmatrix, arr)
    data = list(data)
    return numpy.stack(data)
