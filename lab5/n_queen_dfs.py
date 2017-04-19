import numpy
import sys
sys.path.append('.')

from Lab5.dfs import DepthFirstSearch


class NQueenDFS(DepthFirstSearch):

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.dim = [n, n]

    def is_final(self, node):
        node = numpy.array(node).reshape(self.dim)
        if sum(node.diagonal()) > 1:
            return False
        for i in range(self.n):
            if sum(node[:, i]) > 1:
                return False
            if sum(node[i, :]) > 1:
                return False
        return True

    def get_adjacent(self, node):
        node = numpy.array(node).reshape(self.dim)
        queen_pos = numpy.where(node == 1)
        for i in range(self.n):
            curr_queen = queen_pos[0][i], queen_pos[1][i]
            # Left
            if curr_queen[0] - 1 >= 0:
                temp = node.copy()
                new = curr_queen[0] - 1, curr_queen[1]
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    yield tuple(temp.ravel().tolist())
            # Left-Bottom
            if curr_queen[0] - 1 >= 0 and curr_queen[1] + 1 < self.n:
                temp = node.copy()
                new = curr_queen[0] - 1, curr_queen[1] + 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    yield tuple(temp.ravel().tolist())
            # Bottom
            if curr_queen[1] + 1 < self.n:
                temp = node.copy()
                new = curr_queen[0], curr_queen[1] + 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    yield tuple(temp.ravel().tolist())
            # Right-Bottom
            if curr_queen[0] + 1 < self.n and curr_queen[1] + 1 < self.n:
                temp = node.copy()
                new = curr_queen[0] + 1, curr_queen[1] + 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    yield tuple(temp.ravel().tolist())
            # Right
            if curr_queen[0] + 1 < self.n:
                temp = node.copy()
                new = curr_queen[0] + 1, curr_queen[1]
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    yield tuple(temp.ravel().tolist())
            # Right-Top
            if curr_queen[0] + 1 < self.n and curr_queen[1] - 1 >= 0:
                temp = node.copy()
                new = curr_queen[0] + 1, curr_queen[1] - 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    yield tuple(temp.ravel().tolist())
            # Top
            if curr_queen[1] - 1 >= 0:
                temp = node.copy()
                new = curr_queen[0], curr_queen[1] - 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    yield tuple(temp.ravel().tolist())
            # Left-Top
            if curr_queen[0] - 1 >= 0 and curr_queen[1] - 1 >= 0:
                temp = node.copy()
                new = curr_queen[0] - 1, curr_queen[1] - 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    yield tuple(temp.ravel().tolist())


def main():
    n = 5
    nqp = NQueenDFS(n)
    root = tuple([1]*n + [0]*(n-1)*n)
    nqp.add_node(root)
    result = nqp.traverse()
    print(result[0])
    print(result[2])


if __name__ == '__main__':
    main()
