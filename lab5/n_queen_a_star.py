import numpy
import sys
sys.path.append('.')

from Lab5.a_star_search import AStarSearch


class NQueenAStar(AStarSearch):

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.dim = [n, n]

    def get_score(self, node):
        if len(node) == 2:
            node = node[1]
        node = numpy.array(node).reshape(self.dim)
        score = 0
        sum_diag = sum(node.diagonal())
        if sum_diag > 1:
            score += sum_diag - 1
        for i in range(self.n):
            sum_row = sum(node[i, :])
            score += sum_row - 1 if sum_row > 1 else 0
            sum_col = sum(node[:, i])
            score += sum_col - 1 if sum_col > 1 else 0
        return score

    def is_final(self, node):
        return self.get_score(node) == 0

    def get_adjacent(self, node):
        node = numpy.array(node[1]).reshape(self.dim)
        queen_pos = numpy.where(node == 1)
        for i in range(self.n):
            curr_queen = queen_pos[0][i], queen_pos[1][i]
            # Left
            if curr_queen[0] - 1 >= 0:
                temp = node.copy()
                new = curr_queen[0] - 1, curr_queen[1]
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    state = tuple(temp.ravel().tolist())
                    score = self.get_score(state)
                    yield (score, state)
            # Left-Bottom
            if curr_queen[0] - 1 >= 0 and curr_queen[1] + 1 < self.n:
                temp = node.copy()
                new = curr_queen[0] - 1, curr_queen[1] + 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    state = tuple(temp.ravel().tolist())
                    score = self.get_score(state)
                    yield (score, state)
            # Bottom
            if curr_queen[1] + 1 < self.n:
                temp = node.copy()
                new = curr_queen[0], curr_queen[1] + 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    state = tuple(temp.ravel().tolist())
                    score = self.get_score(state)
                    yield (score, state)
            # Right-Bottom
            if curr_queen[0] + 1 < self.n and curr_queen[1] + 1 < self.n:
                temp = node.copy()
                new = curr_queen[0] + 1, curr_queen[1] + 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    state = tuple(temp.ravel().tolist())
                    score = self.get_score(state)
                    yield (score, state)
            # Right
            if curr_queen[0] + 1 < self.n:
                temp = node.copy()
                new = curr_queen[0] + 1, curr_queen[1]
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    state = tuple(temp.ravel().tolist())
                    score = self.get_score(state)
                    yield (score, state)
            # Right-Top
            if curr_queen[0] + 1 < self.n and curr_queen[1] - 1 >= 0:
                temp = node.copy()
                new = curr_queen[0] + 1, curr_queen[1] - 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    state = tuple(temp.ravel().tolist())
                    score = self.get_score(state)
                    yield (score, state)
            # Top
            if curr_queen[1] - 1 >= 0:
                temp = node.copy()
                new = curr_queen[0], curr_queen[1] - 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    state = tuple(temp.ravel().tolist())
                    score = self.get_score(state)
                    yield (score, state)
            # Left-Top
            if curr_queen[0] - 1 >= 0 and curr_queen[1] - 1 >= 0:
                temp = node.copy()
                new = curr_queen[0] - 1, curr_queen[1] - 1
                if node[new] == 0:
                    temp[curr_queen], temp[new] = 0, 1
                    state = tuple(temp.ravel().tolist())
                    score = self.get_score(state)
                    yield (score, state)


def main():
    n = 5
    nqp = NQueenAStar(n)
    root = tuple([1]*n + [0]*(n-1)*n)
    score = nqp.get_score(root)
    nqp.add_node((score, root))
    result = nqp.traverse()
    print(result[0])
    print(result[2])

if __name__ == '__main__':
    main()