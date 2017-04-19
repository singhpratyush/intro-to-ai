import sys
import pickle

from matplotlib import pyplot


with open(sys.argv[1], 'rb') as f:
    plot_data = pickle.load(f)

for i in [500, 2500, 4500]:
    pyplot.plot(*zip(*plot_data[i]), label='Training Size - {0}'.format(i))
pyplot.grid(True)
pyplot.legend(loc='best')
pyplot.title("Iterations vs. Testing Error with Learning Rate 0.0005")#sys.argv[2])
pyplot.xlabel("Number of Iterations")#sys.argv[3])
pyplot.ylabel("Testing Error Ratio")#sys.argv[4])
pyplot.show()
